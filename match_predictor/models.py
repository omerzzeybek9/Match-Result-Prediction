"""Goal distributions and outcome models with one coherent score grid."""
from __future__ import annotations

import numpy as np
from scipy.stats import poisson
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import LogisticRegression, PoissonRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from .features import result_labels

MAX_GOALS = 30  # With lambda capped at 8, excluded Poisson mass is negligible.
GOALS = np.arange(MAX_GOALS + 1)
HOME_MASK = GOALS[:, None] > GOALS[None, :]
DRAW_MASK = GOALS[:, None] == GOALS[None, :]
AWAY_MASK = GOALS[:, None] < GOALS[None, :]
MASKS = [HOME_MASK, DRAW_MASK, AWAY_MASK]


def goal_grid(means, rho=0.0):
    means = np.clip(np.asarray(means, dtype=float), 0.05, 8.0)
    h, a = means[:, 0], means[:, 1]
    grid = poisson.pmf(GOALS[None, :], h[:, None])[:, :, None] * poisson.pmf(
        GOALS[None, :], a[:, None])[:, None, :]
    # Optional low-score adjustment; default training leaves rho at zero.
    grid[:, 0, 0] *= np.maximum(0.01, 1 - h * a * rho)
    grid[:, 0, 1] *= np.maximum(0.01, 1 + h * rho)
    grid[:, 1, 0] *= np.maximum(0.01, 1 + a * rho)
    grid[:, 1, 1] *= max(0.01, 1 - rho)
    return grid / grid.sum(axis=(1, 2), keepdims=True)


def outcomes(grid):
    return np.column_stack([grid[:, mask].sum(axis=1) for mask in MASKS])


def reconcile_outcomes(grid, probabilities):
    """Preserve within-outcome score rankings while matching classifier probabilities."""
    base = outcomes(grid)
    result = grid.copy()
    for k, mask in enumerate(MASKS):
        result[:, mask] *= (probabilities[:, k] / np.maximum(base[:, k], 1e-15))[:, None]
    return result / result.sum(axis=(1, 2), keepdims=True)


def temperature_scale(grid, temperature):
    p = np.maximum(outcomes(grid), 1e-12) ** (1.0 / temperature)
    p /= p.sum(axis=1, keepdims=True)
    return reconcile_outcomes(grid, p)


def _transformer(columns, categorical=True):
    numeric = [c for c in columns if c not in ["home_team", "away_team"]]
    transforms = [("numbers", StandardScaler(), numeric)]
    if categorical:
        transforms.append(("teams", OneHotEncoder(handle_unknown="ignore", sparse_output=False),
                           ["home_team", "away_team"]))
    return ColumnTransformer(transforms)


class MatchModel:
    def __init__(self, name, columns):
        self.name = name
        self.columns = list(columns)

    def fit(self, frame):
        x = frame[self.columns]
        targets = frame[["home_goals", "away_goals"]].to_numpy()
        days_old = (frame.date.max() - frame.date).dt.days.to_numpy()
        weights = np.exp(-np.log(2) * days_old / 730.0)
        self.mean = np.average(targets, axis=0, weights=weights)
        self.goal_models = []
        if self.name in ["league_mean", "rolling_poisson"]:
            return self
        for side in range(2):
            if self.name.startswith("poisson") or self.name.startswith("logistic"):
                alpha = 0.1 if self.name == "poisson_0.1" else 1.0
                estimator = PoissonRegressor(alpha=alpha, max_iter=500, tol=1e-6)
                categorical = True
            elif self.name == "boosted_poisson":
                estimator = HistGradientBoostingRegressor(loss="poisson", max_iter=160,
                    learning_rate=0.045, max_leaf_nodes=7, min_samples_leaf=40,
                    l2_regularization=8, early_stopping=False, random_state=42)
                categorical = False
            elif self.name == "random_forest":
                estimator = RandomForestRegressor(n_estimators=160, min_samples_leaf=15,
                    max_depth=10, max_features=0.7, n_jobs=1, random_state=42)
                categorical = False
            else:
                raise ValueError(f"Unknown model {self.name}")
            model = make_pipeline(_transformer(self.columns, categorical), estimator)
            model.fit(x, targets[:, side], **{f"{model.steps[-1][0]}__sample_weight": weights})
            self.goal_models.append(model)
        if self.name.startswith("logistic"):
            c = float(self.name.split("_")[1])
            self.classifier = make_pipeline(_transformer(self.columns),
                LogisticRegression(C=c, max_iter=1000, random_state=42))
            self.classifier.fit(x, result_labels(frame), logisticregression__sample_weight=weights)
        return self

    def means(self, x):
        if self.name == "league_mean":
            return np.tile(self.mean, (len(x), 1))
        if self.name == "rolling_poisson":
            h = (x.home_venue_gf.to_numpy() + x.away_venue_ga.to_numpy()) / 2
            a = (x.away_venue_gf.to_numpy() + x.home_venue_ga.to_numpy()) / 2
            # Shrunk venue rates need a small home adjustment.
            return np.column_stack([h * 1.06, a * 0.94])
        return np.clip(np.column_stack([m.predict(x[self.columns]) for m in self.goal_models]), .05, 8)

    def predict_grid(self, frame, rho=0.0):
        grid = goal_grid(self.means(frame), rho)
        if self.name.startswith("logistic"):
            raw = self.classifier.predict_proba(frame[self.columns])
            p = np.full((len(frame), 3), 1e-9)
            p[:, self.classifier.classes_.astype(int)] = raw
            p /= p.sum(axis=1, keepdims=True)
            grid = reconcile_outcomes(grid, p)
        return grid
