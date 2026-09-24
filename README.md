# Match Result Prediction — v3.4

## v3.4 readiness and availability update

- English **Injuries & Lineups** view across the ten-league API catalog, with fixture-specific reasons, timestamps, endpoint status and starting XIs.
- Partial API failures preserve successful endpoints; empty injury responses remain unknown, not proof of a healthy squad.
- Snapshots captured at/after kickoff, live fixtures, missing timestamps and optional player-statistics payloads cannot be marked pre-match safe.
- The UI's optional context gate is on by default. Missing, stale or incomplete context makes the selective signal abstain, without changing baseline probabilities. This new gated subset has **not** been backtested; older accuracy/coverage figures below describe the old policy only.
- Player statistics use provider league/team names, stable player IDs and a recent-fixture window. Later context-only refreshes do not erase completed stats. Historical injury mentions are not current injury status.
- All ten league keys now work with the history importer and training CLI. Four new league models still require licensed historical results and forward evaluation.
- `python -m match_predictor doctor` checks local readiness without exposing the API key or making network requests.

See [LIVE_USE.md](LIVE_USE.md) for setup, import commands and explicit release blockers. Injury/lineup information is included in the dashboard and quality gates, **not in learned probability adjustments**. No new accuracy or profitability result is claimed by this release.

A reproducible pre-match football forecasting application with an English Streamlit dashboard. It produces home/draw/away probabilities, expected goals and likely scores, then adds team history, league tables and player form from timestamped context snapshots. It also emits a **double-chance pick for every fixture** by excluding the least likely 1-X-2 result; the stricter exact-result mode can still abstain when a fixture does not meet a validation-selected confidence rule.

Two modes are available:

- **Statistics only:** causal form, Elo, venue history, rest, shots and ensemble models.
- **Market assisted:** the statistical forecast plus three contemporaneous decimal 1-X-2 odds. Historical validation often assigned 90–100% of the outcome blend to normalized market consensus, so this mode should be understood as a market-informed forecast with the statistical model retained for score shape and comparison.

## Run the prepared package

Python **3.11 or 3.12** is recommended. From the repository root:

```bash
python -m venv .venv
# Windows: .venv\Scripts\activate
# macOS/Linux: source .venv/bin/activate
python -m pip install -r requirements-tested.txt
python -m streamlit run streamlit/user_interface.py
```

The dashboard has five views: **Match Forecast**, **Team Dashboard**, **Player Dashboard**, **Injuries & Lineups** and **Model Evaluation**. Six domestic league model artifacts are included in prepared packages; the other four need historical data and evaluation.

The prepared release includes trained artifacts. A fresh Git clone excludes downloaded history and binary artifacts; build them with:

```bash
python -m pip install -r requirements.txt
python -m match_predictor download --start 2018
python -m match_predictor train
python -m match_predictor report
```

CLI prediction with optional odds:

```bash
python -m match_predictor predict \
  --league premier_league --home Arsenal --away Chelsea --date 2026-09-26 \
  --home-odds 2.10 --draw-odds 3.50 --away-odds 3.40
```

Capture timestamped pre-match context from API-Football. Keep the key in the environment, never in source control:

```bash
# Windows PowerShell: $env:API_FOOTBALL_KEY="..."
export API_FOOTBALL_KEY="..."
python -m match_predictor collect --fixture 123456
python -m match_predictor collect --league premier_league --season 2026 \
  --from-date 2026-09-25 --to-date 2026-09-27 --details
```

For completed fixtures, add player-level appearances, minutes, ratings, goals and assists to the dashboard snapshot:

```bash
python -m match_predictor collect --fixture 123456 --include-player-stats
```

Raw responses and normalized context are stored under `data/api_football/`. The default collector requests fixture metadata, lineups, injuries and pre-match odds; the opt-in player-statistics flag is reserved for completed fixtures and is marked unsafe for pre-match modeling. Re-run pre-match collection at fixed offsets such as T-24h and T-1h so later model training can compare equivalent information windows.

Supply all three odds from the same source and snapshot, or supply none. Odds must be decimal and greater than 1.00. Exact provider team names are available in the UI dropdowns. Historical forecasts require a pre-date refit; the application rejects dates already included in the artifact.

## Recorded result

Six domestic leagues were pooled only for the global strong-forecast rule. The recipe and threshold used older expanding validation windows. They were then frozen before two later periods were scored.

| Mode | Full confirmation accuracy | Strong threshold | Strong confirmation | Coverage | Latest partial-season audit |
|---|---:|---:|---:|---:|---:|
| Statistics only | 51.2% | 65.0% | **72.6% (231/318)** | 15.5% | **83.9% (26/31)** |
| Market assisted | 53.4% | 62.5% | **73.3% (296/404)** | 19.6% | **74.6% (47/63)** |

For a useful output on every match, the separate double-chance task reached **77.3%** statistics-only and **79.3%** market-assisted accuracy on all 2,058 confirmation fixtures, with 100% coverage. The strongest constant double-chance baseline scored 74.4%, while the six market-assisted league results ranged from 78.4% to 80.5%. Double chance covers two of the three possible outcomes, so this number is deliberately not presented as exact 1-X-2 accuracy.

The confirmation set contains 2,058 matches from 2025/26. The later audit contains the first 256 matches available from 2026/27. Confirmation 95% intervals are 67.5–77.2% for statistics only and 68.7–77.3% for market assisted, so the point estimates exceed 70% but do not prove that the long-run rate is above 70%. The result is pooled; individual league samples are smaller and some league point estimates are below 70%.

The filter predicts only about one in five fixtures in the stronger market-assisted mode. Reporting 73.3% without its 19.6% coverage would be misleading. A green decision means the fixture matches the historical selection rule; it does not mean the individual outcome has a 73.3% chance or is guaranteed.

Full evidence is in `reports/benchmark.md`, `artifacts/selective_report.json`, and the per-match validation, confirmation and audit CSV files.

## Modeling and leakage controls

- Features are snapshotted before a match day and results from that date are observed only afterwards.
- Features include 5/10/20-match form, home/away history, exponentially weighted scoring, Elo, rest within the competition, experience, league scoring levels and lagged shots/on-target shots.
- Candidates include league/rolling Poisson references, two regularized Poisson regressions, two multinomial logistic models, Poisson gradient boosting, Random Forest and CatBoost.
- Ensemble weights, calibration temperature and market weight are selected on three expanding validation folds.
- The strong threshold is selected from pooled validation predictions with a multiple-search-adjusted one-sided Wilson lower bound. Confirmation and audit labels never change it.
- The final confirmation season stays outside model, ensemble, calibration and threshold selection. A newer partial season is reported as a secondary forward audit.
- Joint score probabilities are reconciled to 1-X-2 probabilities, keeping likely scores and outcome probabilities mathematically consistent.
- Production artifacts are refitted on all available results only after evaluation.

Run the meaningful invariance and probability checks with:

```bash
python -m unittest discover -s tests -v
```

## Data and files

Historical source: [Football-Data CSVs](https://www.football-data.co.uk/data.php). The normalized files retain completed results, laggable match statistics and a complete market-average closing price triplet when available, with fallbacks to earlier market-average or one-provider prices. Odds are converted to normalized implied probabilities, removing the overround.

The source page states that its free data is intended for private individuals and restricts commercial/data-training uses. Review its current terms before publishing, redistributing, commercializing or automating this project; use a licensed feed where required. Source files can be revised. The manifest records URL, retrieval time, source and normalized SHA-256 hashes, match count and last match date.

The trained Football-Data models currently cover Premier League (`E0`), Bundesliga (`D1`), La Liga (`SP1`), Serie A (`I1`), Ligue 1 (`F1`) and Eredivisie (`N1`). The API-Football dashboard catalog additionally includes Primeira Liga, Belgian Pro League, Süper Lig and Scottish Premiership; those four are marked as data-collection pending until their history has passed the same forward evaluation. Internal `eredivise` spelling remains for compatibility. Champions League uses only a small legacy 2024/25 sample, is excluded from the 70% system and remains experimental.

- `match_predictor/`: data, features, models, selective policy, training, prediction and CLI.
- `artifacts/<league>.joblib`: production model and feature state.
- `artifacts/*_validation_predictions.csv`: policy-development predictions.
- `artifacts/*_test_predictions.csv`: untouched full-season confirmation predictions.
- `artifacts/*_audit_predictions.csv`: latest partial-season forward audit.
- `artifacts/selective_report.json`: threshold search, coverage, pooled and league results.
- `reports/benchmark.md`: readable benchmark and limitations.
- `tests/`: leakage, temporal split, probability, input and selection checks.

Legacy notebooks and models remain as project history and are not used by v3. Their random-split scores are not comparable because their feature construction contains future information.

## Active-use limits

Refresh and retrain before use. The application blocks the strong signal when the latest result is more than 14 days before the selected fixture. It also blocks teams with fewer than ten historical matches and rejects unknown teams.

The v3.3 collector now records dated lineups, injuries, odds and optional completed-match player snapshots, but the production probability model does not consume those new context fields until enough historical snapshots exist for a leakage-safe forward evaluation. API-Football is not treated as a consistent historical xG feed; the normalized schema leaves xG empty unless a licensed provider supplies it. Rest reflects this competition only. Totals and both-teams-to-score values are derived from the goal grid and have not received a separate confirmation study. No betting edge, expected value or profitability after bookmaker margin has been established.

A legacy source file contained an API credential. It has been removed from this snapshot, but an earlier public Git history may retain it. Revoke or rotate that credential with its provider. The new CSV flow does not use it.

MIT license applies to project code. External data remains subject to its provider's terms.
