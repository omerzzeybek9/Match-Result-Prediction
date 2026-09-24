"""Pure data preparation helpers for the English Streamlit dashboard."""
from __future__ import annotations

import json
import math
import unicodedata
from pathlib import Path
from typing import Any

import pandas as pd

from .data import ROOT, load_matches


def _key(value: Any) -> str:
    text = unicodedata.normalize("NFKD", str(value or ""))
    return "".join(char for char in text.casefold() if not unicodedata.combining(char))


def team_history(league: str, team: str, limit: int = 20, data_dir: str | Path | None = None) -> pd.DataFrame:
    """Return recent completed fixtures from the selected team's perspective."""
    matches = load_matches(league, data_dir)
    selected = matches.loc[(matches.home_team == team) | (matches.away_team == team)].copy()
    selected = selected.sort_values("date", ascending=False).head(limit)
    if selected.empty:
        return pd.DataFrame(columns=["date", "opponent", "venue", "score", "result", "points", "goals_for", "goals_against"])
    rows = []
    for row in selected.itertuples(index=False):
        home = row.home_team == team
        goals_for, goals_against = (row.home_goals, row.away_goals) if home else (row.away_goals, row.home_goals)
        result = "W" if goals_for > goals_against else "D" if goals_for == goals_against else "L"
        rows.append({"date": row.date, "opponent": row.away_team if home else row.home_team,
                     "venue": "Home" if home else "Away", "score": f"{goals_for}-{goals_against}",
                     "result": result, "points": 3 if result == "W" else 1 if result == "D" else 0,
                     "goals_for": int(goals_for), "goals_against": int(goals_against),
                     "goal_difference": int(goals_for - goals_against),
                     "shots_for": getattr(row, "home_shots" if home else "away_shots", math.nan),
                     "shots_against": getattr(row, "away_shots" if home else "home_shots", math.nan),
                     "sot_for": getattr(row, "home_sot" if home else "away_sot", math.nan),
                     "sot_against": getattr(row, "away_sot" if home else "home_sot", math.nan)})
    return pd.DataFrame(rows)


def team_summary(history: pd.DataFrame, window: int = 5) -> dict[str, Any]:
    recent = history.head(window).copy()
    if recent.empty:
        return {"matches": 0, "record": "—", "points": 0, "points_per_game": None,
                "goals_for": None, "goals_against": None, "clean_sheet_rate": None,
                "form": "—"}
    wins = int((recent.result == "W").sum())
    draws = int((recent.result == "D").sum())
    losses = int((recent.result == "L").sum())
    return {"matches": len(recent), "record": f"{wins}-{draws}-{losses}",
            "points": int(recent.points.sum()), "points_per_game": float(recent.points.mean()),
            "goals_for": int(recent.goals_for.sum()), "goals_against": int(recent.goals_against.sum()),
            "clean_sheet_rate": float((recent.goals_against == 0).mean()),
            "form": " ".join(recent.result.tolist()[::-1])}


def league_table(league: str, season: int | None = None, data_dir: str | Path | None = None) -> pd.DataFrame:
    """Build a points table from completed match results only."""
    matches = load_matches(league, data_dir)
    if season is None:
        season = int(matches.season.max())
    matches = matches.loc[matches.season == season]
    table: dict[str, dict[str, int]] = {}
    for row in matches.itertuples(index=False):
        for team in (row.home_team, row.away_team):
            table.setdefault(team, {"played": 0, "wins": 0, "draws": 0, "losses": 0,
                                    "goals_for": 0, "goals_against": 0, "points": 0})
        home, away = table[row.home_team], table[row.away_team]
        home["played"] += 1; away["played"] += 1
        home["goals_for"] += int(row.home_goals); home["goals_against"] += int(row.away_goals)
        away["goals_for"] += int(row.away_goals); away["goals_against"] += int(row.home_goals)
        if row.home_goals > row.away_goals:
            home["wins"] += 1; home["points"] += 3; away["losses"] += 1
        elif row.home_goals < row.away_goals:
            away["wins"] += 1; away["points"] += 3; home["losses"] += 1
        else:
            home["draws"] += 1; away["draws"] += 1; home["points"] += 1; away["points"] += 1
    result = pd.DataFrame([{"team": team, **values, "goal_difference": values["goals_for"] - values["goals_against"]}
                           for team, values in table.items()])
    if result.empty:
        return result
    return result.sort_values(["points", "goal_difference", "goals_for"], ascending=False).reset_index(drop=True).assign(position=lambda x: x.index + 1)


def _context_files(data_root: str | Path | None = None) -> list[Path]:
    root = Path(data_root or ROOT / "data/api_football")
    if not root.exists():
        return []
    return sorted(root.glob("**/*.normalized.json"))


def load_context_records(data_root: str | Path | None = None) -> list[dict[str, Any]]:
    records = []
    for path in _context_files(data_root):
        try:
            record = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        if isinstance(record, dict):
            records.append(record)
    return records


def player_dashboard(team: str, data_root: str | Path | None = None,
                     league_id: int | None = None, window: int = 10) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Aggregate player snapshots, preferring completed-match stats when present."""
    wanted = _key(team)
    latest_by_fixture: dict[tuple[Any, str], dict[str, Any]] = {}
    for record in load_context_records(data_root):
        if league_id is not None and record.get("league_id") != league_id:
            continue
        side = None
        if _key(record.get("home_team")) == wanted:
            side = "home"
        elif _key(record.get("away_team")) == wanted:
            side = "away"
        if side is None:
            continue
        identity = (record.get("fixture_id"), side)
        current = latest_by_fixture.get(identity)
        def priority(item):
            return (bool(item.get("player_stats", {}).get(side)), str(item.get("captured_at", "")))
        if current is None or priority(record) > priority(current):
            latest_by_fixture[identity] = record
    latest_by_fixture = dict(sorted(latest_by_fixture.items(),
        key=lambda item: str(item[1].get("kickoff_utc") or ""), reverse=True)[:window])
    players: dict[tuple[Any, str], dict[str, Any]] = {}
    stats_available = False
    snapshots = 0
    for record in latest_by_fixture.values():
        snapshots += 1
        side = "home" if _key(record.get("home_team")) == wanted else "away"
        stats = record.get("player_stats", {}).get(side, []) if isinstance(record.get("player_stats"), dict) else []
        lineup = record.get("lineups", {}).get(side, {}) if isinstance(record.get("lineups"), dict) else {}
        source_rows = stats if stats else lineup.get("players", [])
        if stats:
            stats_available = True
        for entry in source_rows if isinstance(source_rows, list) else []:
            if not isinstance(entry, dict):
                continue
            player_id = entry.get("player_id") or entry.get("id")
            name = entry.get("player_name") or entry.get("name")
            if player_id is None and not name:
                continue
            key = (player_id, "") if player_id is not None else (_key(name), _key(name))
            row = players.setdefault(key, {"player": name or str(player_id), "player_id": player_id,
                                           "position": entry.get("position") or entry.get("pos"),
                                           "snapshots": 0, "lineup_starts": 0, "starts": 0, "appearances": 0, "minutes": 0,
                                           "goals": 0, "assists": 0, "shots": 0, "shots_on": 0,
                                           "key_passes": 0, "rating_sum": 0.0, "rating_count": 0,
                                           "injury_mentions": 0})
            row["snapshots"] += 1
            if stats:
                if entry.get("position"):
                    row["position"] = entry["position"]
                for field in ["starts", "appearances", "minutes", "goals", "assists", "shots", "shots_on", "key_passes"]:
                    row[field] += int(entry.get(field) or 0)
                rating = entry.get("rating")
                if rating is not None:
                    row["rating_sum"] += float(rating); row["rating_count"] += 1
            else:
                row["lineup_starts"] += 1
    for record in latest_by_fixture.values():
        side = "home" if _key(record.get("home_team")) == wanted else "away"
        injuries = record.get("injuries", {}).get("players", []) if isinstance(record.get("injuries"), dict) else []
        for injury in injuries:
            if not isinstance(injury, dict) or injury.get("team") != side:
                continue
            player_id, name = injury.get("player_id"), injury.get("player_name")
            if player_id is None and not name:
                continue
            key = (player_id, "") if player_id is not None else (_key(name), _key(name))
            if key not in players:
                players[key] = {"player": injury.get("player_name") or str(injury.get("player_id")),
                                "player_id": injury.get("player_id"), "position": None, "snapshots": 0,
                                "lineup_starts": 0, "starts": 0, "appearances": 0, "minutes": 0, "goals": 0, "assists": 0,
                                "shots": 0, "shots_on": 0, "key_passes": 0, "rating_sum": 0.0,
                                "rating_count": 0, "injury_mentions": 0}
            players[key]["injury_mentions"] += 1
    result = pd.DataFrame(players.values())
    if result.empty:
        return result, {"snapshots": snapshots, "stats_available": stats_available}
    result["average_rating"] = result.apply(lambda row: row.rating_sum / row.rating_count if row.rating_count else None, axis=1)
    result = result.drop(columns=["rating_sum", "rating_count"]).sort_values(["minutes", "starts", "snapshots"], ascending=False)
    return result.reset_index(drop=True), {"snapshots": snapshots, "stats_available": stats_available}
