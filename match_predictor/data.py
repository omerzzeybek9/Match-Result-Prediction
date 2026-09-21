"""Read only dated results, never season-end standings or target-match statistics."""
from __future__ import annotations

import hashlib
import io
import json
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
LEAGUES = {
    "premier_league": ("Premier League", "E0", "pl"),
    "bundesliga": ("Bundesliga", "D1", "bundesliga"),
    "laliga": ("La Liga", "SP1", "laliga"),
    "seriea": ("Serie A", "I1", "seriea"),
    "ligue1": ("Ligue 1", "F1", "ligue1"),
    "eredivise": ("Eredivisie", "N1", "eredivise"),
    "cl": ("Champions League · legacy data", None, "cl"),
}
OPTIONAL_STATS = {"HS": "home_shots", "AS": "away_shots",
                  "HST": "home_sot", "AST": "away_sot"}
MARKET_ODDS = ["market_home_odds", "market_draw_odds", "market_away_odds"]
ODDS_TRIPLETS = [
    ("AvgCH", "AvgCD", "AvgCA"),  # market-average closing prices
    ("AvgH", "AvgD", "AvgA"),     # market-average earlier prices
    ("B365CH", "B365CD", "B365CA"),
    ("B365H", "B365D", "B365A"),
]


def validate_matches(frame: pd.DataFrame) -> pd.DataFrame:
    required = ["date", "home_team", "away_team", "home_goals", "away_goals"]
    missing = set(required) - set(frame.columns)
    if missing:
        raise ValueError(f"Missing match fields: {sorted(missing)}")
    df = frame.copy()
    # Future knockout slots can have neither date nor teams yet. Omit only rows
    # with no result at all, before validating the identities of completed matches.
    df = df.loc[~df[["home_goals", "away_goals"]].isna().all(axis=1)].copy()
    df["date"] = pd.to_datetime(df["date"], utc=True, errors="raise").dt.normalize()
    if df["date"].isna().any() or df[["home_team", "away_team"]].isna().any().any():
        raise ValueError("Missing match date or team")
    for column in ["home_team", "away_team"]:
        df[column] = df[column].str.strip()
        if df[column].eq("").any():
            raise ValueError("Empty team name")
    # Entirely unplayed fixtures are omitted; partial/invalid results are rejected.
    df = df.loc[~df[["home_goals", "away_goals"]].isna().all(axis=1)].copy()
    for column in ["home_goals", "away_goals"]:
        df[column] = pd.to_numeric(df[column], errors="raise")
        v = df[column].to_numpy()
        if not np.isfinite(v).all() or (v < 0).any() or (v != np.floor(v)).any():
            raise ValueError(f"Invalid goals in {column}")
        df[column] = df[column].astype(int)
    if df["home_team"].eq(df["away_team"]).any():
        raise ValueError("A team cannot play itself")
    for column in OPTIONAL_STATS.values():
        if column not in df:
            df[column] = np.nan
        df[column] = pd.to_numeric(df[column], errors="coerce")
        if (df[column].dropna() < 0).any():
            raise ValueError(f"Negative count in {column}")
    for column in MARKET_ODDS:
        if column not in df:
            df[column] = np.nan
        df[column] = pd.to_numeric(df[column], errors="coerce")
    valid_odds = df[MARKET_ODDS].notna().all(axis=1) & df[MARKET_ODDS].gt(1.0).all(axis=1)
    df.loc[~valid_odds, MARKET_ODDS] = np.nan
    key = ["date", "home_team", "away_team"]
    if df.duplicated(key).any():
        raise ValueError("Duplicate match identities; resolve source conflicts before training")
    if "season" not in df:
        df["season"] = df.date.dt.year - (df.date.dt.month < 7).astype(int)
    df["season"] = df["season"].astype(int)
    return df.sort_values(key, kind="stable").reset_index(drop=True)


def parse_football_data(payload: bytes, season: int) -> pd.DataFrame:
    source = pd.read_csv(io.BytesIO(payload), encoding="utf-8-sig")
    rename = {"HomeTeam": "home_team", "AwayTeam": "away_team",
              "FTHG": "home_goals", "FTAG": "away_goals", **OPTIONAL_STATS}
    if not {"Date", "HomeTeam", "AwayTeam", "FTHG", "FTAG"}.issubset(source.columns):
        raise ValueError("Response is not a football-data result CSV")
    source = source.loc[~source[["FTHG", "FTAG"]].isna().all(axis=1)].copy()
    source["date"] = pd.to_datetime(source["Date"], dayfirst=True, format="mixed", utc=True)
    source["season"] = season
    source[MARKET_ODDS] = np.nan
    # Take a complete price set from one snapshot/provider. Mixing individual
    # columns would create probabilities that never coexisted in the market.
    remaining = pd.Series(True, index=source.index)
    for columns in ODDS_TRIPLETS:
        if not all(column in source for column in columns):
            continue
        values = source[list(columns)].apply(pd.to_numeric, errors="coerce")
        usable = remaining & values.notna().all(axis=1) & values.gt(1.0).all(axis=1)
        source.loc[usable, MARKET_ODDS] = values.loc[usable].to_numpy()
        remaining &= ~usable
    source = source.rename(columns=rename)
    keep = ["date", "season", "home_team", "away_team", "home_goals", "away_goals"]
    keep += [c for c in OPTIONAL_STATS.values() if c in source]
    keep += MARKET_ODDS
    return validate_matches(source[keep])


def download_history(leagues, start=2018, end=2025, directory=None):
    """Refresh explicitly requested seasons. Individual failures remain visible."""
    directory = Path(directory or ROOT / "data/history")
    directory.mkdir(parents=True, exist_ok=True)
    manifest_path = directory / "manifest.json"
    manifest = json.loads(manifest_path.read_text()) if manifest_path.exists() else {}
    if start > end:
        raise ValueError("start must not exceed end")

    def fetch(league, year):
        code = LEAGUES[league][1]
        url = f"https://www.football-data.co.uk/mmz4281/{year % 100:02d}{(year + 1) % 100:02d}/{code}.csv"
        error = None
        for attempt in range(2):
            try:
                request = urllib.request.Request(url, headers={"User-Agent": "MatchResultResearch/3.0"})
                with urllib.request.urlopen(request, timeout=35) as response:
                    payload = response.read()
                df = parse_football_data(payload, year)
                if df.empty:
                    raise ValueError("No completed matches")
                # Reject future results instead of quietly training on bad source dates.
                today = pd.Timestamp.now(tz="UTC").normalize()
                if (df.date > today).any():
                    raise ValueError("Source contains results dated in the future")
                name = f"{league}_{year}.csv"
                temp = directory / (name + ".tmp")
                df.to_csv(temp, index=False)
                temp.replace(directory / name)
                return name, {"source": url, "downloaded_at": datetime.now(timezone.utc).isoformat(),
                              "source_sha256": hashlib.sha256(payload).hexdigest(),
                              "normalized_sha256": hashlib.sha256((directory / name).read_bytes()).hexdigest(),
                              "matches": len(df), "last_match": str(df.date.max().date())}
            except Exception as exc:
                error = exc
                if attempt == 0:
                    time.sleep(0.5)
        raise RuntimeError(f"{league} {year}: {error}")

    errors = []
    with ThreadPoolExecutor(max_workers=4) as pool:
        jobs = [pool.submit(fetch, league, year) for league in leagues
                if LEAGUES[league][1] for year in range(start, end + 1)]
        for task in as_completed(jobs):
            try:
                name, entry = task.result()
                manifest[name] = entry
                print(f"Downloaded {name}: {entry['matches']} matches", flush=True)
            except Exception as exc:
                errors.append(str(exc))
                print(f"FAILED: {exc}", flush=True)
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    return errors


def load_matches(league, directory=None):
    if league not in LEAGUES:
        raise ValueError(f"Unknown league: {league}")
    directory = Path(directory or ROOT / "data/history")
    files = sorted(directory.glob(f"{league}_*.csv"))
    if files:
        # Do not mix provider team names with legacy aliases.
        return validate_matches(pd.concat([pd.read_csv(f) for f in files], ignore_index=True))
    legacy = ROOT / "Fixture/data" / f"fixtures_{LEAGUES[league][2]}.csv"
    df = pd.read_csv(legacy).rename(columns={"Date": "date", "Home Team": "home_team",
        "Away Team": "away_team", "Home Score": "home_goals", "Away Score": "away_goals"})
    return validate_matches(df)
