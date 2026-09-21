from __future__ import annotations

from copy import deepcopy

import numpy as np
import pandas as pd

from .models import GOALS, blend_market, outcomes, temperature_scale
from .selection import double_chance_pick, policy_verdict


def predict_match(bundle, home, away, date=None, decimal_odds=None):
    if bundle.get("schema_version") != 3:
        raise ValueError("Unsupported model artifact; retrain with the current version")
    if home == away:
        raise ValueError("Choose two different teams")
    unknown = {home, away} - set(bundle["teams"])
    if unknown:
        raise ValueError(f"No historical data for: {', '.join(sorted(unknown))}")
    latest = pd.Timestamp(bundle["last_match_date"], tz="UTC")
    date = pd.Timestamp(date) if date is not None else pd.Timestamp.now(tz="UTC")
    date = date.tz_localize("UTC") if date.tzinfo is None else date.tz_convert("UTC")
    date = date.normalize()
    if date <= latest:
        raise ValueError(f"This model includes results through {latest.date()}; choose a later date. Historical forecasts require a pre-date refit.")
    engine = deepcopy(bundle["engine"])
    season = date.year - (date.month < 7)
    engine.prepare_season(season)
    features = engine.snapshot(home, away, date)
    frame = pd.DataFrame([features])[bundle["columns"]]
    statistical_grid = temperature_scale(sum(w*bundle["models"][name].predict_grid(frame)
                                              for name,w in bundle["weights"].items()), bundle["temperature"])
    statistical_p = outcomes(statistical_grid)[0]
    market_p = None
    mode = "stats"
    if decimal_odds is not None:
        odds=np.asarray(decimal_odds,dtype=float)
        if odds.shape != (3,) or not np.isfinite(odds).all() or (odds <= 1).any():
            raise ValueError("Home, draw and away decimal odds must all be finite and greater than 1.00")
        market_p=1/odds;market_p/=market_p.sum()
        frame[["market_home","market_draw","market_away"]]=market_p
        grid=blend_market(statistical_grid,frame,bundle["market_weight"])[0]
        mode="assisted"
    else:
        grid=statistical_grid[0]
    p = outcomes(grid[None])[0]
    positions = np.argsort(grid.ravel())[-5:][::-1]
    scores = [{"home": int(i//len(GOALS)), "away": int(i%len(GOALS)), "probability": float(grid.ravel()[i])} for i in positions]
    warnings = []
    age = (date - latest).days
    if age > 30:
        warnings.append(f"Latest result is {age} days before this match date. Refresh data and retrain.")
    for team in [home, away]:
        if engine.teams[team].matches < 10:
            warnings.append(f"{team}: fewer than 10 historical matches.")
        if team not in bundle["recent_teams"]:
            warnings.append(f"{team}: not present in the latest available season.")
    total = GOALS[:, None] + GOALS[None, :]
    policies=bundle.get("selective_policies",{})
    verdict=policy_verdict(policies.get(mode),float(p.max()),engine.teams[home].matches,
                           engine.teams[away].matches,age,agreement=int(p.argmax())==int(statistical_p.argmax()),
                           experimental=bundle["league"]=="cl")
    labels=[home,"Beraberlik",away]
    double_chance=double_chance_pick(p)
    double_chance["label"]={
        "1X":f"{home} veya Beraberlik",
        "X2":f"Beraberlik veya {away}",
        "12":f"{home} veya {away} (beraberlik yok)",
    }[double_chance["code"]]
    return {"home_team":home,"away_team":away,"date":str(date.date()),
            "probabilities":{"home":float(p[0]),"draw":float(p[1]),"away":float(p[2])},
            "statistical_probabilities":{"home":float(statistical_p[0]),"draw":float(statistical_p[1]),"away":float(statistical_p[2])},
            "market_probabilities":None if market_p is None else {"home":float(market_p[0]),"draw":float(market_p[1]),"away":float(market_p[2])},
            "prediction_mode":mode,"predicted_outcome":labels[int(p.argmax())],"selection":verdict,
            "double_chance":double_chance,
            "expected_goals":{"home":float((grid.sum(axis=1)*GOALS).sum()),
                              "away":float((grid.sum(axis=0)*GOALS).sum())},
            "top_scores":scores,"over_2_5":float(grid[total>=3].sum()),
            "both_teams_score":float(grid[1:,1:].sum()),"warnings":warnings,
            "last_match_date":bundle["last_match_date"],
            "form":{"home_points_last5":features["home_points_5"],"away_points_last5":features["away_points_5"],
                    "home_elo":features["home_elo"],"away_elo":features["away_elo"]}}
