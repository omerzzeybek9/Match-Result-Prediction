import copy
import json
import tempfile
import unittest
from pathlib import Path

from match_predictor.api_football import ApiFootballClient, normalize_fixture_snapshot
from match_predictor.context import assess_context, availability_rows, latest_fixture_context
from match_predictor.dashboard import player_dashboard
from match_predictor.history_import import import_fixture_history
from match_predictor.data import API_LEAGUES, LEAGUES, load_matches


def snapshot():
    return {"fixture_id": 1, "captured_at": "2026-01-01T10:00:00Z", "responses": {
        "fixture": {"response": [{"fixture": {"id": 1, "date": "2026-01-01T12:00:00Z", "status": {"short": "NS"}},
            "league": {"id": 203, "season": 2025},
            "teams": {"home": {"id": 10, "name": "Home"}, "away": {"id": 20, "name": "Away"}}}]},
        "lineups": {"response": [{"team": {"id": team}, "startXI": [{"player": {"id": team * 100 + n, "name": str(n)}} for n in range(11)]} for team in [10, 20]]},
        "injuries": {"response": [{"team": {"id": 10}, "player": {"id": 50, "name": "Unavailable", "type": "Missing Fixture", "reason": "Suspended"}}]}}}


class ContextTests(unittest.TestCase):
    def test_unknown_status_and_fixture_mismatch_fail_closed(self):
        raw = snapshot()
        raw["responses"]["fixture"]["response"][0]["fixture"]["status"] = {}
        self.assertFalse(normalize_fixture_snapshot(raw)["prematch_safe"])
        raw["fixture_id"] = 99
        with self.assertRaises(ValueError):
            normalize_fixture_snapshot(raw)

    def test_temporal_and_completeness_checks(self):
        record = normalize_fixture_snapshot(snapshot())
        self.assertTrue(assess_context(record, "2026-01-01T11:00:00Z")["ready"])
        self.assertFalse(assess_context(record, "2026-01-01T12:00:00Z")["ready"])
        self.assertFalse(assess_context(record, "2026-01-01T09:00:00Z")["ready"])
        self.assertFalse(assess_context(None)["ready"])
        self.assertFalse(assess_context(record, "2026-01-01T11:00:00Z")["model_uses_context"])

    def test_stale_snapshot_rejected_before_kickoff(self):
        record = normalize_fixture_snapshot(snapshot())
        record["captured_at"] = "2025-12-31T10:00:00Z"
        self.assertFalse(assess_context(record, "2026-01-01T11:00:00Z")["ready"])

    def test_postmatch_live_naive_and_unknown_times_not_safe(self):
        for captured in ["2026-01-01T12:00:00Z", "2026-01-01T13:00:00Z", None, "bad", "2026-01-01T10:00:00"]:
            raw = snapshot()
            raw["captured_at"] = captured
            self.assertFalse(normalize_fixture_snapshot(raw)["prematch_safe"])
        raw = snapshot()
        raw["responses"]["fixture"]["response"][0]["fixture"]["status"]["short"] = "1H"
        self.assertFalse(normalize_fixture_snapshot(raw)["prematch_safe"])

    def test_partial_and_duplicate_lineups_not_confirmed(self):
        raw = snapshot()
        raw["responses"]["lineups"]["response"][0]["startXI"] = [{"player": {"id": 1}}] * 11
        self.assertFalse(normalize_fixture_snapshot(raw)["lineups"]["home"]["confirmed"])
        raw["responses"]["lineups"]["response"][0]["startXI"] = [{"player": {"id": 1}}]
        self.assertFalse(normalize_fixture_snapshot(raw)["lineups"]["home"]["confirmed"])

    def test_empty_injuries_are_unknown_not_healthy(self):
        raw = snapshot()
        raw["responses"]["injuries"] = {"response": []}
        record = normalize_fixture_snapshot(raw)
        self.assertEqual(record["endpoint_status"]["injuries"], "empty")
        self.assertFalse(assess_context(record, "2026-01-01T11:00:00Z")["ready"])

    def test_optional_endpoint_failure_preserves_other_data(self):
        class Fake(ApiFootballClient):
            def get(self, endpoint, params):
                if endpoint == "injuries":
                    raise RuntimeError("quota")
                key = {"fixtures": "fixture", "fixtures/lineups": "lineups"}.get(endpoint)
                return snapshot()["responses"].get(key, {"response": []})
        raw = Fake("fake").fixture_snapshot(1, requested_at="2026-01-01T10:00:00Z")
        record = normalize_fixture_snapshot(raw)
        self.assertEqual(record["endpoint_status"]["injuries"], "error")
        self.assertEqual(record["home_team"], "Home")

    def test_asof_selection_does_not_read_future(self):
        first = normalize_fixture_snapshot(snapshot())
        later = {**first, "captured_at": "2026-01-01T11:30:00Z"}
        self.assertEqual(latest_fixture_context([first, later], 1, "2026-01-01T11:00:00Z"), first)
        self.assertIsNone(latest_fixture_context([first], 1, "2026-01-01T09:00:00Z"))

    def test_injury_details_preserve_reason_and_time(self):
        rows = availability_rows(normalize_fixture_snapshot(snapshot()))
        self.assertEqual(rows[0]["Reason"], "Suspended")
        self.assertEqual(rows[0]["Team"], "Home")
        self.assertEqual(rows[0]["Captured at"], "2026-01-01T10:00:00Z")

    def test_actual_fixture_player_schema(self):
        raw = snapshot()
        raw["responses"]["player_stats"] = {"response": [{"team": {"id": 10}, "players": [
            {"player": {"id": 7, "name": "Starter"}, "statistics": [{"games": {"minutes": 90, "substitute": False, "rating": "7.3"}}]},
            {"player": {"id": 8, "name": "Sub"}, "statistics": [{"games": {"minutes": 20, "substitute": True}}]},
            {"player": {"id": 9, "name": "Unused"}, "statistics": [{"games": {"minutes": None, "substitute": True}}]}]}]}
        record = normalize_fixture_snapshot(raw)
        self.assertFalse(record["prematch_safe"])
        self.assertEqual([p["starts"] for p in record["player_stats"]["home"]], [1, 0, 0])
        self.assertEqual([p["appearances"] for p in record["player_stats"]["home"]], [1, 1, 0])

    def test_later_context_refresh_does_not_erase_stats(self):
        with tempfile.TemporaryDirectory() as directory:
            record = normalize_fixture_snapshot(snapshot())
            record["player_stats"]["home"] = [{"player_id": 7, "player_name": "Name", "minutes": 90}]
            later = copy.deepcopy(record)
            later["captured_at"] = "2026-01-01T11:00:00Z"
            later["player_stats"]["home"] = []
            for index, row in enumerate([record, later]):
                Path(directory, f"{index}.normalized.json").write_text(json.dumps(row))
            players, _ = player_dashboard("Home", directory, league_id=203)
            self.assertEqual(players.loc[players.player_id == 7, "minutes"].iloc[0], 90)
            empty, _ = player_dashboard("Home", directory, league_id=39)
            self.assertTrue(empty.empty)


class HistoryImportTests(unittest.TestCase):
    def test_all_ten_leagues_trainable_from_imported_results(self):
        self.assertTrue(set(API_LEAGUES).issubset(LEAGUES))
        with tempfile.TemporaryDirectory() as directory:
            raw = snapshot()["responses"]["fixture"]
            fixture = raw["response"][0]
            fixture["fixture"]["status"]["short"] = "FT"
            fixture["score"] = {"fulltime": {"home": 2, "away": 1}}
            path = Path(directory, "listing.json")
            path.write_text(json.dumps({"response": raw}))
            output = Path(directory, "history")
            self.assertEqual(len(import_fixture_history(path, "super_lig", output)), 1)
            matches = load_matches("super_lig", output)
            self.assertEqual(matches.iloc[0].home_goals, 2)
            with self.assertRaises(FileExistsError):
                import_fixture_history(path, "super_lig", output)

    def test_unplayed_wrong_league_and_partial_pages_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory, "listing.json")
            raw = snapshot()["responses"]["fixture"]
            path.write_text(json.dumps(raw))
            with self.assertRaises(ValueError):
                import_fixture_history(path, "super_lig", directory)
            raw["paging"] = {"total": 2}
            path.write_text(json.dumps(raw))
            with self.assertRaises(ValueError):
                import_fixture_history(path, "super_lig", directory)


if __name__ == "__main__":
    unittest.main()
