"""Stateful feature engine: snapshot a whole day before observing any result."""
from __future__ import annotations

from collections import defaultdict, deque
from dataclasses import dataclass, field

import numpy as np
import pandas as pd


@dataclass
class TeamState:
    elo: float = 1500.0
    matches: int = 0
    history: deque = field(default_factory=lambda: deque(maxlen=20))
    home: deque = field(default_factory=lambda: deque(maxlen=12))
    away: deque = field(default_factory=lambda: deque(maxlen=12))
    last_date: object = None
    ew_gf: float = 1.35
    ew_ga: float = 1.35


class FeatureEngine:
    def __init__(self):
        self.teams = defaultdict(TeamState)
        self.season = None
        self.home_goals = 30.0
        self.away_goals = 24.0
        self.games = 20
        self.latest_date = None

    def prepare_season(self, season):
        if self.season is not None and season != self.season:
            for state in self.teams.values():
                state.elo = 1500 + 0.8 * (state.elo - 1500)
        self.season = season

    @staticmethod
    def _mean(records, key, prior, strength=3):
        values = [r[key] for r in records if np.isfinite(r[key])]
        return (sum(values) + strength * prior) / (len(values) + strength)

    def _team_features(self, name, venue, date):
        s = self.teams[name]
        mean_goal = (self.home_goals + self.away_goals) / (2 * self.games)
        result = {"elo": s.elo, "experience": min(s.matches, 100),
                  "rest_days": min((date - s.last_date).days, 30) if s.last_date is not None else 14,
                  "ew_gf": s.ew_gf, "ew_ga": s.ew_ga}
        for window in [5, 10, 20]:
            history = list(s.history)[-window:]
            for key, prior in [("gf", mean_goal), ("ga", mean_goal), ("points", 1.35)]:
                result[f"{key}_{window}"] = self._mean(history, key, prior)
        history = getattr(s, venue)
        for key, prior in [("gf", mean_goal), ("ga", mean_goal), ("points", 1.35)]:
            result[f"venue_{key}"] = self._mean(history, key, prior, strength=5)
        recent = list(s.history)[-10:]
        for key, prior in [("shots", 12.0), ("sot", 4.0), ("sot_against", 4.0)]:
            result[f"{key}_10"] = self._mean(recent, key, prior)
        result["shot_history_count"] = sum(np.isfinite(r["sot"]) for r in recent)
        return result

    def snapshot(self, home, away, date):
        if home == away:
            raise ValueError("Home and away teams must be different")
        hf = self._team_features(home, "home", date)
        af = self._team_features(away, "away", date)
        row = {"home_team": home, "away_team": away,
               **{f"home_{k}": v for k, v in hf.items()},
               **{f"away_{k}": v for k, v in af.items()}}
        for key in ["elo", "points_5", "points_10", "gf_10", "ga_10", "rest_days", "sot_10"]:
            row[f"diff_{key}"] = hf[key] - af[key]
        row["league_home_goals"] = self.home_goals / self.games
        row["league_away_goals"] = self.away_goals / self.games
        return row

    def observe(self, row):
        home, away = self.teams[row.home_team], self.teams[row.away_team]
        hg, ag = row.home_goals, row.away_goals
        expectation = 1 / (1 + 10 ** ((away.elo - home.elo - 65) / 400))
        actual = 1.0 if hg > ag else 0.5 if hg == ag else 0.0
        delta = 20 * (1 + np.log1p(abs(hg - ag)) / 2) * (actual - expectation)
        home.elo += delta
        away.elo -= delta
        pairs = [(home, "home", hg, ag, row.home_shots, row.home_sot, row.away_sot),
                 (away, "away", ag, hg, row.away_shots, row.away_sot, row.home_sot)]
        for state, venue, gf, ga, shots, sot, sot_against in pairs:
            record = {"gf": gf, "ga": ga, "points": 3 if gf > ga else 1 if gf == ga else 0,
                      "shots": shots, "sot": sot, "sot_against": sot_against}
            state.history.append(record)
            getattr(state, venue).append(record)
            state.matches += 1
            state.last_date = row.date
            state.ew_gf = 0.85 * state.ew_gf + 0.15 * gf
            state.ew_ga = 0.85 * state.ew_ga + 0.15 * ga
        self.home_goals += hg
        self.away_goals += ag
        self.games += 1
        self.latest_date = row.date


def build_features(matches):
    engine = FeatureEngine()
    rows = []
    for date, group in matches.sort_values("date", kind="stable").groupby("date", sort=True):
        if group.season.nunique() != 1:
            raise ValueError("Mixed seasons on a single date")
        engine.prepare_season(int(group.season.iloc[0]))
        for row in group.itertuples(index=False):
            rows.append({"date": date, "season": row.season,
                         "home_goals": row.home_goals, "away_goals": row.away_goals,
                         **engine.snapshot(row.home_team, row.away_team, date)})
        for row in group.itertuples(index=False):
            engine.observe(row)
    return pd.DataFrame(rows), engine


def feature_columns(frame):
    return [c for c in frame.columns if c not in ["date", "season", "home_goals", "away_goals"]]


def result_labels(frame):
    """Fixed class order: 0=home win, 1=draw, 2=away win."""
    return np.where(frame.home_goals > frame.away_goals, 0,
                    np.where(frame.home_goals == frame.away_goals, 1, 2))
