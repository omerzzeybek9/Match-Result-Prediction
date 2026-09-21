# Match Result Prediction — v2

A reproducible pre-match football forecasting application. Predicts **home win / draw / away win probabilities**, expected goals and the five most likely scores. Includes dated match history, causal features, chronological model selection and an independent test season.

## Start here

Python **3.11 or 3.12** is recommended. Run commands from the repository root.

```bash
python -m venv .venv
# Windows: .venv\Scripts\activate
# macOS/Linux: source .venv/bin/activate
python -m pip install -r requirements.txt
python -m match_predictor download --start 2018
python -m match_predictor train
python -m streamlit run streamlit/user_interface.py
```

The download command includes the current season by default. It uses public CSVs and requires no API key. A prepared release ZIP includes the downloaded history and trained artifacts, so **download/train can be skipped** to inspect that snapshot. The Git repository excludes binary artifacts and downloaded data; a fresh clone needs those steps. Only load joblib files you trust. To use bundled artifacts, install the exact tested versions in `requirements-tested.txt`, or retrain with your installed scikit-learn version.

Train one league or request a prediction:

```bash
python -m match_predictor train --leagues premier_league
python -m match_predictor predict --league premier_league --home Arsenal --away Chelsea --date 2026-09-21
python -m unittest discover -s tests -v
```

Use exact provider team names (the UI supplies dropdowns). A forecast date must be later than the last completed result included in its artifact. Historic predictions require retraining with an earlier data cutoff; using a current artifact for an old match is rejected.

## What changed

- Replaced season-end standings joined to every old fixture with **match-day snapshots** built only from previously completed dates.
- Added 5/10/20-match form, venue history, exponentially weighted goals, Elo ratings, rest within the competition, experience, league scoring levels, and lagged shots/shots on target when supplied by the source.
- A whole day's features are computed **before** any of that day's results are observed. Dates are intentionally normalized to days; same-day result reuse is conservatively excluded.
- Added eight candidates: league-average Poisson, rolling-goal Poisson, two regularized Poisson regressions, two regularized multinomial logistic regressions, Poisson gradient boosting, and Random Forest goal regression.
- Team categories are one-hot encoded with unseen-category handling for linear models. Tree models use numeric pre-match strength/form features.
- Weights and probability temperature are selected from three expanding validation folds. The final test period never enters model selection, encoding fit, weight fitting or calibration.
- Score distributions and 1X2 probabilities remain consistent. Logistic candidates reweight within-outcome score distributions; top scores and totals all come from the resulting joint grid.
- Replaced machine-specific Windows paths and free-text teams with portable paths and dropdowns. Both teams feed the **same requested fixture**.
- Added measured-performance and confidence-bucket views, stale-data notices and small-history notices.
- Removed embedded API credentials from legacy scripts; they now require `FOOTBALL_DATA_API_KEY` in the environment. The new CSV downloader does not need it.

## Data and reproducibility

Source: [Football-Data historical CSVs](https://www.football-data.co.uk/data.php). The downloader uses paths such as `https://www.football-data.co.uk/mmz4281/2425/E0.csv`. Source files may be revised; the local manifest records URL, retrieval time, original-content and normalized-content SHA-256 hashes, match count and last match date. Consult the source's terms before redistributing or using its data commercially.

Supported historical leagues: Premier League (`E0`), Bundesliga (`D1`), La Liga (`SP1`), Serie A (`I1`), Ligue 1 (`F1`), Eredivisie (`N1`). Internal `eredivise` spelling is kept for compatibility. The CSV history replaces, rather than mixes with, legacy provider data to avoid inconsistent team names.

Champions League (`cl`) still uses the repository's limited 2024/25 fixture data and is **experimental**. It can be trained explicitly with `--leagues cl`. Six domestic leagues are trained by default. No domestic-to-European team identity mapping is assumed.

- `data/history/`: normalized dated results and download manifest.
- `match_predictor/`: loading, causal features, models, training, prediction and CLI.
- `artifacts/<league>.joblib`: production model and historical feature state.
- `artifacts/<league>_report.json`: validation and held-out test metrics, dates, versions, dataset hash and caveats.
- `artifacts/<league>_test_predictions.csv`: each held-out fixture and its predicted probabilities.
- `reports/benchmark.md`: human-readable measured results for this release.
- `tests/`: leakage, temporal separation, input validation and probability consistency checks.

Legacy notebooks and models remain as historical work; they are **not** used by v2. Their old random-split MSE figures are not comparable with the new out-of-time results.

## Evaluation protocol

For multi-season data, the latest season containing at least 200 completed matches is held out. This is a **substantial season**, not necessarily a finished one: the report gives its exact dates and count. The preceding two seasons supply three expanding-window validation folds; older matches form the initial training history. A newer season with fewer than 200 matches is excluded from this test but included in the subsequent production refit.

Candidate selection minimizes **1X2 log loss**, not raw accuracy. Non-negative ensemble weights are fitted with a small regularizer on validation predictions, compared with the best individual candidate, and followed by a coarse probability-temperature search on the same validation predictions. None of these choices use final-test results.

For the test season, models are trained on all prior seasons and held fixed. Feature state advances after each match day, so this tests sequential upcoming-match forecasts, not predictions for an entire season made on opening day. After evaluation, the chosen recipe is refitted on every available result for the UI. Reported test scores are therefore **not** scores of the all-data production fit.

Reports include accuracy, approximate 95% Wilson interval, log loss, multiclass Brier score (sum over classes), expected-goal MAE, exact-score accuracy, confidence buckets and a paired day-block bootstrap for the log-loss difference against rolling-goal Poisson. These quantify different aspects of performance; a method can improve log loss without improving accuracy. The small-data fallback uses the last 20% of distinct dates as test and the preceding 30% for expanding validation.

## Practical limits

This is a working statistical forecasting system, not a promise of highly accurate individual scores. Inputs do not include current injuries, starting lineups, transfers or xG. Form and rest are competition-specific; cross-competition congestion is missing. Unknown teams are rejected rather than assigned fabricated evidence. Shot fields are used only after their match date, but historical vendor corrections cannot be reconstructed as originally published.

Goal distributions use a bounded independent-Poisson starting model; outcome reweighting adds flexibility but does not remove all score-distribution assumptions. The optional low-score correction is disabled in this release. Totals and both-teams-to-score probabilities are derived outputs; they have not undergone separate calibration/evaluation. There is no bookmaker-odds benchmark or demonstrated betting profitability.

Refreshing results regularly and adding dated lineup/injury/xG data are the next substantive extensions. Future experiments should use new validation periods and preserve a fresh final holdout rather than repeatedly tuning against this release's test season.

**Credential follow-up:** a legacy source file contained a hard-coded API credential. It has been removed from this snapshot, but the repository's earlier public history may still contain it. Revoke/rotate the old credential with its provider; this update does not rewrite GitHub history or revoke keys.

## Updating

```bash
python -m match_predictor download --start 2026 --end 2026
python -m match_predictor train
```

Use the actual season start year. Download failures are reported with a nonzero exit code; existing successful files are retained. Duplicate fixture identities and partial/invalid scores fail validation. The app visibly flags stale results. No external publication or GitHub push is performed by these commands.

MIT license applies to project code. Data remains subject to its provider's terms.
