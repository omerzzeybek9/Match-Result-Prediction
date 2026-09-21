"""Expanding-window selection, untouched season test, then production refit."""
from __future__ import annotations

import hashlib
import json
import platform
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import sklearn
from scipy.optimize import minimize
from sklearn.metrics import log_loss

from .data import ROOT, load_matches
from .features import build_features, feature_columns, result_labels
from .models import GOALS, MatchModel, outcomes, temperature_scale

CANDIDATES = ["league_mean", "rolling_poisson", "poisson_1", "poisson_0.1",
              "logistic_0.05", "logistic_0.5", "boosted_poisson", "random_forest"]


def chronological_partitions(frame):
    counts = frame.groupby("season").size()
    seasons = sorted(counts.index)
    if len(seasons) >= 4:
        substantial = [s for s in seasons[3:] if counts[s] >= 200]
        if not substantial:
            raise ValueError("No season with at least 200 matches for an independent test")
        test_season = substantial[-1]
        test = frame.index[frame.season == test_season].to_numpy()
        development = frame.index[frame.season < test_season].to_numpy()
        validation_dates = frame.loc[frame.season.isin([test_season - 2, test_season - 1]), "date"].unique()
        protocol = f"Untouched season {test_season}/{test_season + 1}; 3 expanding validation folds in preceding 2 seasons"
    else:
        dates = np.sort(frame.date.unique())
        if len(dates) < 20 or len(frame) < 120:
            raise ValueError("At least 120 completed matches and 20 distinct dates are required")
        test_start = dates[int(len(dates) * 0.8)]
        validation_start = dates[int(len(dates) * 0.5)]
        test = frame.index[frame.date >= test_start].to_numpy()
        development = frame.index[frame.date < test_start].to_numpy()
        validation_dates = frame.loc[(frame.date >= validation_start) & (frame.date < test_start), "date"].unique()
        protocol = "Small-data fallback: final 20% of dates held out; 3 expanding folds over preceding 30%"
    folds = []
    for dates in np.array_split(np.sort(validation_dates), 3):
        if len(dates) == 0:
            raise ValueError("Insufficient validation dates")
        train = frame.index[frame.date < dates[0]].to_numpy()
        valid = frame.index[frame.date.isin(dates)].to_numpy()
        if len(train) < 40 or len(valid) < 10:
            raise ValueError("Insufficient data for chronological validation")
        folds.append((train, valid))
    return development, test, folds, protocol


def metrics(frame, grid):
    y = result_labels(frame)
    p = outcomes(grid)
    predicted = p.argmax(axis=1)
    correct = predicted == y
    confidence = p.max(axis=1)
    n = len(y)
    accuracy = float(correct.mean())
    z = 1.96
    centre = (accuracy + z*z/(2*n)) / (1+z*z/n)
    radius = z*np.sqrt(accuracy*(1-accuracy)/n + z*z/(4*n*n))/(1+z*z/n)
    mean_home = (grid.sum(axis=2) * GOALS).sum(axis=1)
    mean_away = (grid.sum(axis=1) * GOALS).sum(axis=1)
    top_score = grid.reshape(n, -1).argmax(axis=1)
    score_home, score_away = np.unravel_index(top_score, grid.shape[1:])
    buckets = []
    ece = 0.0
    for low, high in zip([0, .4, .5, .6, .7, .8, .9], [.4, .5, .6, .7, .8, .9, 1.000001]):
        mask = (confidence >= low) & (confidence < high)
        if mask.any():
            observed, expected = float(correct[mask].mean()), float(confidence[mask].mean())
            ece += mask.mean() * abs(observed - expected)
            buckets.append({"range": f"{low:.0%}–{min(high,1):.0%}", "matches": int(mask.sum()),
                            "mean_probability": expected, "accuracy": observed})
    return {"matches": n, "accuracy": accuracy, "accuracy_95_interval": [centre-radius, centre+radius],
        "log_loss": float(log_loss(y, p, labels=[0, 1, 2])),
        "brier_multiclass": float(np.mean(np.sum((p-np.eye(3)[y])**2, axis=1))),
        "goal_mae": float(np.mean(np.abs(np.column_stack([mean_home, mean_away]) - frame[["home_goals", "away_goals"]].to_numpy()))),
        "exact_score_accuracy": float(np.mean((score_home == frame.home_goals.to_numpy()) & (score_away == frame.away_goals.to_numpy()))),
        "calibration_error": float(ece), "confidence_buckets": buckets}


def select_ensemble(grids, y):
    names = list(grids)
    probabilities = np.stack([outcomes(grids[name]) for name in names], axis=1)
    true_probabilities = probabilities[np.arange(len(y)), :, y]
    losses = {name: float(-np.log(np.maximum(true_probabilities[:, i], 1e-12)).mean())
              for i, name in enumerate(names)}
    def objective(weights):
        return -np.log(np.maximum(true_probabilities @ weights, 1e-12)).mean() + 0.01 * np.sum(weights**2)
    fit = minimize(objective, np.full(len(names), 1/len(names)), method="SLSQP",
                   bounds=[(0, 1)]*len(names), constraints={"type": "eq", "fun": lambda w: w.sum()-1},
                   options={"maxiter": 200, "ftol": 1e-10})
    weights = fit.x if fit.success else np.eye(len(names))[np.argmin(list(losses.values()))]
    weights[weights < 0.001] = 0
    weights /= weights.sum()
    blend = sum(weights[i]*grids[name] for i, name in enumerate(names))
    best_single = min(losses, key=losses.get)
    if log_loss(y, outcomes(blend), labels=[0,1,2]) > losses[best_single]:
        weights = np.array([float(name == best_single) for name in names])
        blend = grids[best_single]
    # Coarse bounded calibration on validation predictions only.
    temperatures = [0.85, 1.0, 1.15, 1.3, 1.5]
    temperature = min(temperatures, key=lambda t: log_loss(y, outcomes(temperature_scale(blend, t)), labels=[0,1,2]))
    return {name: float(w) for name, w in zip(names, weights) if w > 0}, temperature, losses


def _period(frame):
    return {"from": str(frame.date.min().date()), "to": str(frame.date.max().date()), "matches": len(frame)}


def train_league(league, output_dir=None, data_dir=None):
    output_dir = Path(output_dir or ROOT / "artifacts")
    output_dir.mkdir(parents=True, exist_ok=True)
    matches = load_matches(league, data_dir)
    frame, engine = build_features(matches)
    columns = feature_columns(frame)
    development, test_indices, folds, protocol = chronological_partitions(frame)
    predictions = {name: [] for name in CANDIDATES}
    validations = []
    fold_details = []
    for number, (train, valid) in enumerate(folds, 1):
        print(f"{league}: validation fold {number}/3 ({len(train)} train, {len(valid)} validation)", flush=True)
        training, validation = frame.loc[train], frame.loc[valid]
        validations.append(validation)
        fold_details.append({"train": _period(training), "validation": _period(validation)})
        for name in CANDIDATES:
            model = MatchModel(name, columns).fit(training)
            predictions[name].append(model.predict_grid(validation))
    grids = {name: np.concatenate(values) for name, values in predictions.items()}
    validation = pd.concat(validations)
    weights, temperature, losses = select_ensemble(grids, result_labels(validation))
    print(f"{league}: selected {weights}; temperature={temperature}", flush=True)
    test = frame.loc[test_indices]
    # Freeze selection before evaluating the test season. Features update only with earlier results.
    fitted = {name: MatchModel(name, columns).fit(frame.loc[development]) for name in set(weights) | {"league_mean", "rolling_poisson"}}
    selected_grid = temperature_scale(sum(w*fitted[name].predict_grid(test) for name,w in weights.items()), temperature)
    baseline_grids = {name: fitted[name].predict_grid(test) for name in ["league_mean", "rolling_poisson"]}
    report = {"league": league, "protocol": protocol, "selection_metric": "validation 1X2 log loss",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "data": _period(frame), "seasons": sorted(int(s) for s in frame.season.unique()),
        "data_sha256": hashlib.sha256(matches.to_csv(index=False).encode()).hexdigest(),
        "feature_count": len(columns), "features": columns, "folds": fold_details,
        "test_period": _period(test), "test_model_training_period": _period(frame.loc[development]),
        "validation_log_loss": losses, "weights": weights, "temperature": temperature,
        "selected": metrics(test, selected_grid),
        "baselines": {name: metrics(test, grid) for name,grid in baseline_grids.items()},
        "versions": {"python": platform.python_version(), "sklearn": sklearn.__version__, "numpy": np.__version__, "pandas": pd.__version__},
        "limitations": ["No lineups, injuries, transfers or xG inputs.",
            "Features update after each match day; model coefficients are fixed throughout the test season.",
            "Reported test results belong to the pre-test fit; production model is subsequently refitted on all available results.",
            "Rest/form reflect this competition only, not other competitions.",
            "No comparison against bookmaker probabilities; betting profitability is not established."]}
    # Paired day-block bootstrap against the stronger rolling reference.
    y = result_labels(test)
    p = outcomes(selected_grid)
    reference = outcomes(baseline_grids["rolling_poisson"])
    delta = -np.log(np.maximum(p[np.arange(len(y)), y],1e-12)) + np.log(np.maximum(reference[np.arange(len(y)),y],1e-12))
    daily = pd.DataFrame({"date": test.date.to_numpy(), "delta": delta}).groupby("date").delta.agg(["sum","count"])
    rng = np.random.default_rng(42)
    samples = rng.integers(0, len(daily), size=(2000, len(daily)))
    estimates = daily["sum"].to_numpy()[samples].sum(axis=1)/daily["count"].to_numpy()[samples].sum(axis=1)
    report["log_loss_difference_vs_rolling"] = {"difference": float(delta.mean()),
        "day_bootstrap_95_interval": [float(v) for v in np.quantile(estimates,[.025,.975])], "negative_is_better": True}
    if len(matches) < 1000:
        report["limitations"].append("Small historical sample: exploratory results, not strong evidence of generalization.")
    report_path = output_dir / f"{league}_report.json"
    report_path.write_text(json.dumps(report, indent=2, ensure_ascii=False)+"\n")
    prediction_frame = test[["date","home_team","away_team","home_goals","away_goals"]].copy()
    prediction_frame[["p_home","p_draw","p_away"]] = p
    prediction_frame["actual_class"] = y
    prediction_frame["predicted_class"] = p.argmax(axis=1)
    prediction_frame.to_csv(output_dir / f"{league}_test_predictions.csv",index=False)
    print(f"{league}: test accuracy={report['selected']['accuracy']:.3f}, log loss={report['selected']['log_loss']:.3f}; production refit", flush=True)
    production = {name: MatchModel(name, columns).fit(frame) for name in weights}
    bundle = {"schema_version": 2, "league": league, "models": production, "weights": weights,
              "temperature": temperature, "engine": engine, "columns": columns, "report": report,
              "last_match_date": str(matches.date.max().date()),
              "teams": sorted(set(matches.home_team) | set(matches.away_team)),
              "recent_teams": sorted(set(matches.loc[matches.season == matches.season.max(),"home_team"]) |
                                     set(matches.loc[matches.season == matches.season.max(),"away_team"]))}
    path = output_dir / f"{league}.joblib"
    temp = path.with_suffix(".joblib.tmp")
    joblib.dump(bundle, temp, compress=3)
    temp.replace(path)
    return report
