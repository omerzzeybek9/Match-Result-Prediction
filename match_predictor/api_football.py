"""API-Football pre-match snapshots and a provider-neutral normalized context.

The collector intentionally stores raw responses before normalization. This makes
the exact information available at each timestamp auditable and prevents a later
parser change from silently rewriting historical inputs.
"""
from __future__ import annotations

import hashlib
import json
import os
import re
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Mapping

import numpy as np

from .data import API_LEAGUES


API_BASE_URL = "https://v3.football.api-sports.io"

# API-Football's stable league IDs for the dashboard's ten-league catalog.
LEAGUE_IDS = {key: value[1] for key, value in API_LEAGUES.items()}

SNAPSHOT_ENDPOINTS = {
    "fixture": "fixtures",
    "lineups": "fixtures/lineups",
    "injuries": "injuries",
    "odds": "odds",
}


def _utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def _safe_timestamp(value: str) -> str:
    return re.sub(r"[^0-9TZ-]", "", value.replace(":", ""))


def _first_response(payload: Mapping[str, Any]) -> dict[str, Any]:
    response = payload.get("response")
    if not isinstance(response, list) or not response:
        return {}
    first = response[0]
    return first if isinstance(first, dict) else {}


def _as_float(value: Any) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if np.isfinite(number) else None


def _team_id(team: Mapping[str, Any] | None) -> int | None:
    value = team.get("id") if isinstance(team, Mapping) else None
    try:
        return int(value) if value is not None else None
    except (TypeError, ValueError):
        return None


@dataclass(frozen=True)
class ApiFootballClient:
    """Small dependency-free client with injectable opener for deterministic tests."""

    api_key: str
    base_url: str = API_BASE_URL
    timeout: float = 30.0
    opener: Callable[..., Any] | None = None

    @classmethod
    def from_environment(cls, api_key: str | None = None, **kwargs: Any) -> "ApiFootballClient":
        key = api_key or os.environ.get("API_FOOTBALL_KEY")
        if not key:
            raise ValueError("Set API_FOOTBALL_KEY before collecting API-Football data")
        return cls(key, **kwargs)

    def get(self, endpoint: str, params: Mapping[str, Any]) -> dict[str, Any]:
        if endpoint.startswith("/"):
            endpoint = endpoint[1:]
        query = urllib.parse.urlencode({k: v for k, v in params.items() if v is not None})
        url = f"{self.base_url.rstrip('/')}/{endpoint}"
        if query:
            url += f"?{query}"
        request = urllib.request.Request(url, headers={
            "x-apisports-key": self.api_key,
            "Accept": "application/json",
            "User-Agent": "MatchResultPrediction/3.4",
        })
        opener = self.opener or urllib.request.urlopen
        try:
            with opener(request, timeout=self.timeout) as response:
                raw = response.read()
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="replace")[:500]
            raise RuntimeError(f"API-Football HTTP {exc.code}: {detail}") from exc
        except urllib.error.URLError as exc:
            raise RuntimeError(f"API-Football connection failed: {exc.reason}") from exc
        try:
            payload = json.loads(raw)
        except json.JSONDecodeError as exc:
            raise RuntimeError("API-Football returned invalid JSON") from exc
        if not isinstance(payload, dict):
            raise RuntimeError("API-Football returned an unexpected response")
        errors = payload.get("errors")
        if errors and errors != {} and errors != []:
            raise RuntimeError(f"API-Football error: {errors}")
        return payload

    def fixture_snapshot(self, fixture_id: int, requested_at: str | None = None,
                         include_odds: bool = True, include_player_stats: bool = False) -> dict[str, Any]:
        """Fetch a timestamped context snapshot for one fixture.

        Fixture metadata, lineups, injuries and odds are kept as separate raw
        responses. Player statistics are optional because they are post-match
        data and must never enter a pre-match model feature set.
        """
        if int(fixture_id) <= 0:
            raise ValueError("fixture_id must be a positive integer")
        requests = {
            "fixture": (SNAPSHOT_ENDPOINTS["fixture"], {"id": int(fixture_id)}),
            "lineups": (SNAPSHOT_ENDPOINTS["lineups"], {"fixture": int(fixture_id)}),
            "injuries": (SNAPSHOT_ENDPOINTS["injuries"], {"fixture": int(fixture_id)}),
        }
        if include_odds:
            requests["odds"] = (SNAPSHOT_ENDPOINTS["odds"], {"fixture": int(fixture_id)} )
        if include_player_stats:
            requests["player_stats"] = ("fixtures/players", {"fixture": int(fixture_id)})
        responses = {}
        endpoint_errors = {}
        for name, (endpoint, params) in requests.items():
            try:
                responses[name] = self.get(endpoint, params)
            except RuntimeError:
                if name == "fixture":
                    raise
                endpoint_errors[name] = "Endpoint unavailable; check coverage, quota and access."
        # Timestamp when all responses are available, not before slow requests.
        captured_at = requested_at or _utc_now()
        return {"provider": "api-football", "fixture_id": int(fixture_id),
                "captured_at": captured_at, "responses": responses,
                "endpoint_errors": endpoint_errors,
                "player_stats_requested": include_player_stats}

    def fixtures(self, league: str | int, season: int, date_from: str | None = None,
                 date_to: str | None = None) -> dict[str, Any]:
        """List fixtures for a league/season and an optional date window."""
        league_id = LEAGUE_IDS.get(league, league) if isinstance(league, str) else league
        try:
            league_id, season = int(league_id), int(season)
        except (TypeError, ValueError) as exc:
            raise ValueError("league and season must identify a valid competition") from exc
        if league_id <= 0 or season < 1900:
            raise ValueError("Invalid API-Football league or season")
        params = {"league": league_id, "season": season, "from": date_from, "to": date_to}
        return self.get(SNAPSHOT_ENDPOINTS["fixture"], params)


class SnapshotStore:
    """Immutable JSON snapshots plus a compact index for later feature builds."""

    def __init__(self, root: str | Path):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self.index_path = self.root / "index.jsonl"

    def save(self, snapshot: Mapping[str, Any]) -> Path:
        fixture_id = int(snapshot["fixture_id"])
        captured_at = str(snapshot["captured_at"])
        day = captured_at[:10]
        directory = self.root / day
        directory.mkdir(parents=True, exist_ok=True)
        filename = f"fixture_{fixture_id}_{_safe_timestamp(captured_at)}.json"
        path = directory / filename
        payload = json.dumps(snapshot, ensure_ascii=False, sort_keys=True, indent=2).encode("utf-8")
        if path.exists() and path.read_bytes() != payload:
            raise FileExistsError(f"Snapshot path already contains different data: {path}")
        if not path.exists():
            temp = path.with_suffix(".json.tmp")
            temp.write_bytes(payload)
            temp.replace(path)
            entry = {"path": str(path.relative_to(self.root)), "fixture_id": fixture_id,
                     "captured_at": captured_at, "sha256": hashlib.sha256(payload).hexdigest()}
            with self.index_path.open("a", encoding="utf-8") as stream:
                stream.write(json.dumps(entry, ensure_ascii=False) + "\n")
        return path

    def save_normalized(self, record: Mapping[str, Any]) -> Path:
        """Save the parser output beside raw data without changing raw history."""
        fixture_id = int(record["fixture_id"])
        captured_at = str(record["captured_at"])
        day = captured_at[:10]
        directory = self.root / day
        directory.mkdir(parents=True, exist_ok=True)
        path = directory / f"fixture_{fixture_id}_{_safe_timestamp(captured_at)}.normalized.json"
        payload = json.dumps(record, ensure_ascii=False, sort_keys=True, indent=2).encode("utf-8")
        if path.exists() and path.read_bytes() != payload:
            raise FileExistsError(f"Normalized path already contains different data: {path}")
        if not path.exists():
            temp = path.with_suffix(".json.tmp")
            temp.write_bytes(payload)
            temp.replace(path)
        return path

    def save_listing(self, payload: Mapping[str, Any], captured_at: str | None = None) -> Path:
        """Save a fixture-list response used to discover fixture IDs."""
        captured_at = captured_at or _utc_now()
        directory = self.root / "lists" / captured_at[:10]
        directory.mkdir(parents=True, exist_ok=True)
        path = directory / f"fixtures_{_safe_timestamp(captured_at)}.json"
        encoded = json.dumps({"provider": "api-football", "captured_at": captured_at,
                              "response": payload}, ensure_ascii=False, sort_keys=True, indent=2).encode("utf-8")
        if path.exists() and path.read_bytes() != encoded:
            raise FileExistsError(f"Fixture-list path already contains different data: {path}")
        if not path.exists():
            temp = path.with_suffix(".json.tmp")
            temp.write_bytes(encoded)
            temp.replace(path)
        return path


def _lineup_summary(payload: Mapping[str, Any], team_id: int | None) -> dict[str, Any]:
    rows = payload.get("response") if isinstance(payload, Mapping) else None
    for row in rows or []:
        if not isinstance(row, Mapping) or _team_id(row.get("team")) != team_id:
            continue
        starters = row.get("startXI") if isinstance(row.get("startXI"), list) else []
        substitutes = row.get("substitutes") if isinstance(row.get("substitutes"), list) else []
        starter_ids = [entry.get("player", {}).get("id") for entry in starters if isinstance(entry, Mapping)]
        return {"confirmed": len(starter_ids) == 11 and None not in starter_ids and len(set(starter_ids)) == 11,
                "starting_xi_count": len(starters),
                "substitute_count": len(substitutes), "formation": row.get("formation"),
                "coach_id": _team_id(row.get("coach")),
                "players": [entry.get("player", {}) for entry in starters if isinstance(entry, Mapping)]}
    return {"confirmed": False, "starting_xi_count": 0, "substitute_count": 0,
            "formation": None, "coach_id": None, "players": []}


def _injury_summary(payload: Mapping[str, Any], home_id: int | None, away_id: int | None) -> dict[str, Any]:
    counts = {"home": 0, "away": 0}
    players: list[dict[str, Any]] = []
    for row in payload.get("response", []) if isinstance(payload, Mapping) else []:
        if not isinstance(row, Mapping):
            continue
        team = _team_id(row.get("team"))
        side = "home" if team == home_id else "away" if team == away_id else None
        if side is None:
            continue
        counts[side] += 1
        player = row.get("player") if isinstance(row.get("player"), Mapping) else {}
        players.append({"team": side, "player_id": player.get("id"),
                        "player_name": player.get("name"), "type": player.get("type"),
                        "reason": player.get("reason")})
    return {"home_count": counts["home"], "away_count": counts["away"], "players": players}


def _odds_summary(payload: Mapping[str, Any]) -> dict[str, Any]:
    values: dict[str, list[float]] = {"home": [], "draw": [], "away": []}
    aliases = {"home": "home", "draw": "draw", "away": "away"}
    for row in payload.get("response", []) if isinstance(payload, Mapping) else []:
        bookmakers = row.get("bookmakers", []) if isinstance(row, Mapping) else []
        for bookmaker in bookmakers:
            for market in bookmaker.get("bets", []) if isinstance(bookmaker, Mapping) else []:
                name = str(market.get("name", "")).lower()
                if name not in {"match winner", "fulltime result", "1x2"}:
                    continue
                for outcome in market.get("values", []) if isinstance(market, Mapping) else []:
                    label = str(outcome.get("value", "")).strip().lower()
                    side = aliases.get(label)
                    number = _as_float(outcome.get("odd"))
                    if side and number is not None and number > 1:
                        values[side].append(number)
    medians = {side: float(np.median(numbers)) if numbers else None for side, numbers in values.items()}
    valid = all(value is not None for value in medians.values())
    probabilities = None
    if valid:
        raw = np.array([1 / medians["home"], 1 / medians["draw"], 1 / medians["away"]], dtype=float)
        raw /= raw.sum()
        probabilities = {"home": float(raw[0]), "draw": float(raw[1]), "away": float(raw[2])}
    return {"bookmaker_samples": {side: len(numbers) for side, numbers in values.items()},
            "median_decimal": medians, "normalized_probability": probabilities}


def _player_stats_summary(payload: Mapping[str, Any], home_id: int | None,
                          away_id: int | None) -> dict[str, list[dict[str, Any]]]:
    result = {"home": [], "away": []}
    for team_row in payload.get("response", []) if isinstance(payload, Mapping) else []:
        if not isinstance(team_row, Mapping):
            continue
        team = _team_id(team_row.get("team"))
        side = "home" if team == home_id else "away" if team == away_id else None
        if side is None:
            continue
        for row in team_row.get("players", []) if isinstance(team_row.get("players"), list) else []:
            if not isinstance(row, Mapping):
                continue
            player = row.get("player") if isinstance(row.get("player"), Mapping) else {}
            statistics = row.get("statistics") if isinstance(row.get("statistics"), list) else []
            stats = statistics[0] if statistics and isinstance(statistics[0], Mapping) else {}
            games = stats.get("games") if isinstance(stats.get("games"), Mapping) else {}
            goals = stats.get("goals") if isinstance(stats.get("goals"), Mapping) else {}
            shots = stats.get("shots") if isinstance(stats.get("shots"), Mapping) else {}
            passes = stats.get("passes") if isinstance(stats.get("passes"), Mapping) else {}
            cards = stats.get("cards") if isinstance(stats.get("cards"), Mapping) else {}
            result[side].append({"player_id": player.get("id"), "player_name": player.get("name"),
                "position": games.get("position"), "minutes": games.get("minutes") or 0,
                "appearances": int((_as_float(games.get("minutes")) or 0) > 0),
                "starts": int(games.get("substitute") is False),
                "rating": _as_float(games.get("rating")), "goals": goals.get("total") or 0,
                "assists": goals.get("assists") or 0, "shots": shots.get("total") or 0,
                "shots_on": shots.get("on") or 0, "key_passes": passes.get("key") or 0,
                "passes": passes.get("total") or 0, "yellow": cards.get("yellow") or 0,
                "red": cards.get("red") or 0})
    return result


def normalize_fixture_snapshot(snapshot: Mapping[str, Any]) -> dict[str, Any]:
    """Normalize context with conservative temporal and endpoint provenance."""
    responses = snapshot.get("responses", {})
    fixture = _first_response(responses.get("fixture", {}))
    teams = fixture.get("teams", {}) if isinstance(fixture, Mapping) else {}
    home = teams.get("home", {}) if isinstance(teams, Mapping) else {}
    away = teams.get("away", {}) if isinstance(teams, Mapping) else {}
    home_id, away_id = _team_id(home), _team_id(away)
    if (fixture.get("fixture", {}).get("id") != int(snapshot["fixture_id"])
            or home_id is None or away_id is None or home_id == away_id):
        raise ValueError("Missing or mismatched fixture/team identities in provider response")
    kickoff = fixture.get("fixture", {}).get("date") if isinstance(fixture.get("fixture"), Mapping) else None
    status = fixture.get("fixture", {}).get("status", {}).get("short")
    before_kickoff = False
    try:
        captured = datetime.fromisoformat(str(snapshot.get("captured_at")).replace("Z", "+00:00"))
        starts = datetime.fromisoformat(str(kickoff).replace("Z", "+00:00"))
        before_kickoff = captured.tzinfo is not None and starts.tzinfo is not None and captured < starts
    except (ValueError, TypeError):
        pass
    endpoint_errors = snapshot.get("endpoint_errors", {})
    endpoint_status = {name: "error" if name in endpoint_errors else
                       "available" if responses.get(name, {}).get("response") else
                       "empty" if name in responses else "not_requested"
                       for name in ["fixture", "lineups", "injuries", "odds", "player_stats"]}
    result = {"provider": snapshot.get("provider", "api-football"),
              "fixture_id": int(snapshot["fixture_id"]), "captured_at": snapshot.get("captured_at"),
              "kickoff_utc": kickoff, "league_id": fixture.get("league", {}).get("id") if isinstance(fixture.get("league"), Mapping) else None,
              "fixture_status": status, "endpoint_status": endpoint_status,
              "endpoint_errors": endpoint_errors,
              "season": fixture.get("league", {}).get("season") if isinstance(fixture.get("league"), Mapping) else None,
              "home_team_id": home_id, "home_team": home.get("name"),
              "away_team_id": away_id, "away_team": away.get("name"),
              "lineups": {"home": _lineup_summary(responses.get("lineups", {}), home_id),
                          "away": _lineup_summary(responses.get("lineups", {}), away_id)},
              "injuries": _injury_summary(responses.get("injuries", {}), home_id, away_id),
              "odds": _odds_summary(responses.get("odds", {})),
              "xg": {"home": None, "away": None, "source": None},
              "player_stats": _player_stats_summary(responses.get("player_stats", {}), home_id, away_id),
              "prematch_safe": bool(before_kickoff and status == "NS"
                                    and not snapshot.get("player_stats_requested") and "player_stats" not in responses)}
    return result


def collect_fixture(client: ApiFootballClient, fixture_id: int, store: SnapshotStore,
                    requested_at: str | None = None, include_odds: bool = True,
                    include_player_stats: bool = False) -> tuple[Path, dict[str, Any]]:
    snapshot = client.fixture_snapshot(fixture_id, requested_at=requested_at,
                                       include_odds=include_odds,
                                       include_player_stats=include_player_stats)
    path = store.save(snapshot)
    return path, normalize_fixture_snapshot(snapshot)
