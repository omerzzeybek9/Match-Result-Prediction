# Independent chronological benchmark

These are recorded held-out results, not training scores. Recipe selection used earlier validation periods only.

| League | Test matches | Accuracy | League-mean accuracy | Rolling accuracy | Log loss | Rolling log loss | Exact score |
|---|---:|---:|---:|---:|---:|---:|---:|
| Bundesliga | 306 | 54.9% | 43.8% | 51.0% | 0.974 | 1.021 | 11.8% |
| Champions League · legacy data | 16 | 37.5% | 37.5% | 37.5% | 1.173 | 1.158 | 6.2% |
| Eredivisie | 306 | 48.7% | 44.4% | 48.4% | 1.003 | 1.032 | 10.8% |
| La Liga | 380 | 53.7% | 48.9% | 52.9% | 0.974 | 0.998 | 14.2% |
| Ligue 1 | 306 | 50.7% | 46.1% | 48.0% | 1.001 | 1.038 | 7.8% |
| Premier League | 380 | 47.4% | 42.6% | 44.7% | 1.028 | 1.054 | 12.4% |
| Serie A | 380 | 52.1% | 38.9% | 48.2% | 0.995 | 1.043 | 13.2% |

Domestic leagues, match-weighted: **2,058 test matches; 51.2% accuracy; 0.996 log loss vs 1.031 rolling reference.**

Lower log loss is better. The model is selected for probability quality, so higher accuracy is not guaranteed.
The old random-split notebook scores are not a valid comparison because their feature construction contains future information.

## Evidence by league

### Bundesliga

- History: 2018-08-24 to 2026-09-13; 2,475 matches.
- Test: 2025-08-22 to 2026-05-16.
- Protocol: Untouched season 2025/2026; 3 expanding validation folds in preceding 2 seasons.
- Log-loss difference against rolling reference: -0.0472; paired day-bootstrap 95% interval [-0.0720, -0.0226]. Negative favors the selected model; intervals crossing zero do not establish improvement.
- Selected mixture: poisson_1=0.224, poisson_0.1=0.244, logistic_0.05=0.237, logistic_0.5=0.028, boosted_poisson=0.115, random_forest=0.153; temperature=1.15.

### Champions League · legacy data

- History: 2024-09-17 to 2025-02-19; 160 matches.
- Test: 2025-02-11 to 2025-02-19.
- Protocol: Small-data fallback: final 20% of dates held out; 3 expanding folds over preceding 30%.
- Log-loss difference against rolling reference: +0.0151; paired day-bootstrap 95% interval [-0.1359, +0.2901]. Negative favors the selected model; intervals crossing zero do not establish improvement.
- Selected mixture: league_mean=0.013, rolling_poisson=0.455, logistic_0.05=0.513, random_forest=0.020; temperature=0.85.

### Eredivisie

- History: 2018-08-10 to 2026-09-15; 2,428 matches.
- Test: 2025-08-08 to 2026-05-17.
- Protocol: Untouched season 2025/2026; 3 expanding validation folds in preceding 2 seasons.
- Log-loss difference against rolling reference: -0.0287; paired day-bootstrap 95% interval [-0.0601, +0.0081]. Negative favors the selected model; intervals crossing zero do not establish improvement.
- Selected mixture: poisson_1=0.103, poisson_0.1=0.219, logistic_0.05=0.255, boosted_poisson=0.202, random_forest=0.220; temperature=1.0.

### La Liga

- History: 2018-08-17 to 2026-09-17; 3,099 matches.
- Test: 2025-08-15 to 2026-05-24.
- Protocol: Untouched season 2025/2026; 3 expanding validation folds in preceding 2 seasons.
- Log-loss difference against rolling reference: -0.0232; paired day-bootstrap 95% interval [-0.0519, +0.0065]. Negative favors the selected model; intervals crossing zero do not establish improvement.
- Selected mixture: poisson_1=0.271, poisson_0.1=0.175, logistic_0.05=0.123, boosted_poisson=0.105, random_forest=0.326; temperature=0.85.

### Ligue 1

- History: 2018-08-10 to 2026-09-13; 2,753 matches.
- Test: 2025-08-15 to 2026-05-17.
- Protocol: Untouched season 2025/2026; 3 expanding validation folds in preceding 2 seasons.
- Log-loss difference against rolling reference: -0.0369; paired day-bootstrap 95% interval [-0.0596, -0.0153]. Negative favors the selected model; intervals crossing zero do not establish improvement.
- Selected mixture: rolling_poisson=0.203, poisson_1=0.220, poisson_0.1=0.143, logistic_0.05=0.147, logistic_0.5=0.060, random_forest=0.227; temperature=1.0.

### Premier League

- History: 2018-08-10 to 2026-09-14; 3,080 matches.
- Test: 2025-08-15 to 2026-05-24.
- Protocol: Untouched season 2025/2026; 3 expanding validation folds in preceding 2 seasons.
- Log-loss difference against rolling reference: -0.0256; paired day-bootstrap 95% interval [-0.0516, -0.0013]. Negative favors the selected model; intervals crossing zero do not establish improvement.
- Selected mixture: poisson_1=0.256, poisson_0.1=0.354, logistic_0.05=0.058, logistic_0.5=0.156, boosted_poisson=0.043, random_forest=0.133; temperature=1.0.

### Serie A

- History: 2018-08-18 to 2026-09-14; 3,080 matches.
- Test: 2025-08-23 to 2026-05-24.
- Protocol: Untouched season 2025/2026; 3 expanding validation folds in preceding 2 seasons.
- Log-loss difference against rolling reference: -0.0483; paired day-bootstrap 95% interval [-0.0734, -0.0223]. Negative favors the selected model; intervals crossing zero do not establish improvement.
- Selected mixture: poisson_1=0.060, poisson_0.1=0.162, logistic_0.05=0.357, logistic_0.5=0.003, boosted_poisson=0.140, random_forest=0.278; temperature=1.0.

## Limits and interpretation

- Six domestic leagues have multi-season history. Champions League is a separate small-data legacy experiment.
- Model coefficients stay fixed during the test; prior match-day results update causal features.
- Production artifacts are refitted afterwards on all results; their future accuracy remains unknown.
- No lineup/injury/transfer/xG feed, bookmaker benchmark, or demonstrated betting profitability.
- Confidence intervals describe sampling uncertainty, not guarantees against future distribution shifts.
- Per-match predictions, calibration buckets, periods, versions and data hashes are available beside each model.
- Tests: future-result invariance, prefix invariance, same-day isolation, training/live parity, temporal separation, current partial-season handling, valid probability mass, score/outcome consistency, invalid-input rejection and opponent sensitivity.
