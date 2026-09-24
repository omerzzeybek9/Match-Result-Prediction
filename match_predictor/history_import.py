"""Import saved API fixture lists into the existing chronological training flow."""
import json
from pathlib import Path

import pandas as pd

from .data import API_LEAGUES, validate_matches


def import_fixture_history(input_path, league, output_dir):
    payload = json.loads(Path(input_path).read_text(encoding="utf-8"))
    # SnapshotStore.save_listing wraps the provider payload once.
    if isinstance(payload.get("response"), dict):
        payload = payload["response"]
    if payload.get("errors"):
        raise ValueError("Cannot import a failed provider response")
    paging = payload.get("paging", {})
    if paging.get("total", 1) > 1:
        raise ValueError("Import a complete fixture list, not a partial page")
    rows = []
    for item in payload.get("response", []):
        if item.get("league", {}).get("id") != API_LEAGUES[league][1]:
            continue
        fixture = item.get("fixture", {})
        # FT only: exclude abandoned matches and extra-time/penalty results.
        if fixture.get("status", {}).get("short") != "FT":
            continue
        score = item.get("score", {}).get("fulltime") or item.get("goals", {})
        if score.get("home") is None or score.get("away") is None:
            raise ValueError("Completed fixture is missing its full-time score")
        rows.append({"date": fixture["date"], "season": item["league"]["season"],
                     "home_team": item["teams"]["home"]["name"],
                     "away_team": item["teams"]["away"]["name"],
                     "home_goals": score["home"], "away_goals": score["away"]})
    if not rows:
        raise ValueError("No completed regulation-time fixtures for the selected league")
    frame = validate_matches(pd.DataFrame(rows))
    if (frame.date > pd.Timestamp.now(tz="UTC").normalize()).any():
        raise ValueError("Source contains future results")
    directory = Path(output_dir)
    targets = [(directory / f"{league}_{season}.csv", part) for season, part in frame.groupby("season")]
    if any(path.exists() for path, _ in targets):
        raise FileExistsError("Season already exists; import to a new directory to avoid mixing provider identities")
    directory.mkdir(parents=True, exist_ok=True)
    for path, part in targets:
        part.to_csv(path, index=False)
    return [str(path) for path, _ in targets]
