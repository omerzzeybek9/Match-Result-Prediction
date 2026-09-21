# Independent chronological benchmark

Model recipes and strong-prediction thresholds were chosen with older validation periods. The confirmation season and latest partial-season audit were evaluated afterwards.

| League | Confirmation matches | Stats accuracy | Market-assisted accuracy | Stats log loss | Market-assisted log loss |
|---|---:|---:|---:|---:|---:|
| Bundesliga | 306 | 55.2% | 56.5% | 0.973 | 0.951 |
| Champions League · legacy data | 16 | 43.8% | 43.8% | 1.199 | 1.199 |
| Eredivisie | 306 | 48.7% | 52.6% | 0.999 | 0.980 |
| La Liga | 380 | 52.6% | 54.5% | 0.969 | 0.965 |
| Ligue 1 | 306 | 50.7% | 53.9% | 1.001 | 0.975 |
| Premier League | 380 | 47.4% | 49.5% | 1.028 | 1.012 |
| Serie A | 380 | 52.9% | 53.7% | 0.995 | 0.978 |
- Domestic stats, match-weighted: **2,058 matches; 51.2% accuracy; 0.995 log loss.**
- Domestic market-assisted, match-weighted: **2,058 matches; 53.4% accuracy; 0.978 log loss.**

## Every-match forecast: double chance

For every fixture, the two highest-probability 1-X-2 outcomes are retained. This is an easier target than exact 1-X-2 and must not be compared as if they were the same task.

| Mode | Confirmation matches | Coverage | Accuracy | Best constant pick | Uplift | 95% interval |
|---|---:|---:|---:|---:|---:|---:|
| Statistics only | 2058 | 100.0% | 77.3% | 74.4% | 2.9% | 75.4%–79.1% |
| Market assisted | 2058 | 100.0% | 79.3% | 74.4% | 4.8% | 77.4%–80.9% |

The rule excludes the outcome with the lowest predicted probability. It emits a pick for every fixture and was scored on the untouched confirmation season.

## 70% target: selective exact 1-X-2 forecasts

The system may abstain. Accuracy without coverage is misleading, so both are always reported.

| Mode | Validation-selected threshold | Confirmation selected | Confirmation coverage | Confirmation accuracy | Latest audit selected | Latest audit accuracy |
|---|---:|---:|---:|---:|---:|---:|
| Statistics only | 65.0% | 318 | 15.5% | 72.6% | 31 | 83.9% |
| Market assisted | 62.5% | 404 | 19.6% | 73.3% | 63 | 74.6% |

The threshold search used expanding validation predictions only. Confirmation and audit outcomes did not change the threshold.

## Evidence by league

### Bundesliga

- History: 2018-08-24 to 2026-09-13; 2,475 matches.
- Confirmation: 2025-08-22 to 2026-05-16; 306 matches.
- Protocol: Untouched season 2025/2026; 3 expanding validation folds in preceding 2 seasons.
- Statistical log-loss difference against rolling reference: -0.0486; paired day-bootstrap 95% interval [-0.0734, -0.0242]. Negative favors the selected model.
- Statistical mixture: poisson_1=0.176, poisson_0.1=0.248, logistic_0.05=0.140, logistic_0.5=0.021, boosted_poisson=0.038, random_forest=0.049, catboost_4=0.327; temperature=1.15; market weight=1.0.

### Champions League · legacy data

- History: 2024-09-17 to 2025-02-19; 160 matches.
- Confirmation: 2025-02-11 to 2025-02-19; 16 matches.
- Protocol: Small-data fallback: final 20% of dates held out; 3 expanding folds over preceding 30%.
- Statistical log-loss difference against rolling reference: +0.0412; paired day-bootstrap 95% interval [-0.1279, +0.3169]. Negative favors the selected model.
- Statistical mixture: league_mean=0.032, rolling_poisson=0.118, logistic_0.05=0.164, logistic_0.5=0.006, catboost_4=0.679; temperature=0.85; market weight=0.0.

### Eredivisie

- History: 2018-08-10 to 2026-09-15; 2,428 matches.
- Confirmation: 2025-08-08 to 2026-05-17; 306 matches.
- Protocol: Untouched season 2025/2026; 3 expanding validation folds in preceding 2 seasons.
- Statistical log-loss difference against rolling reference: -0.0325; paired day-bootstrap 95% interval [-0.0645, +0.0037]. Negative favors the selected model.
- Statistical mixture: poisson_1=0.046, poisson_0.1=0.191, logistic_0.05=0.195, boosted_poisson=0.164, random_forest=0.162, catboost_4=0.240; temperature=1.0; market weight=0.9.

### La Liga

- History: 2018-08-17 to 2026-09-17; 3,099 matches.
- Confirmation: 2025-08-15 to 2026-05-24; 380 matches.
- Protocol: Untouched season 2025/2026; 3 expanding validation folds in preceding 2 seasons.
- Statistical log-loss difference against rolling reference: -0.0287; paired day-bootstrap 95% interval [-0.0575, +0.0027]. Negative favors the selected model.
- Statistical mixture: poisson_1=0.240, poisson_0.1=0.154, logistic_0.05=0.020, boosted_poisson=0.032, random_forest=0.233, catboost_4=0.322; temperature=0.85; market weight=1.0.

### Ligue 1

- History: 2018-08-10 to 2026-09-13; 2,753 matches.
- Confirmation: 2025-08-15 to 2026-05-17; 306 matches.
- Protocol: Untouched season 2025/2026; 3 expanding validation folds in preceding 2 seasons.
- Statistical log-loss difference against rolling reference: -0.0369; paired day-bootstrap 95% interval [-0.0596, -0.0153]. Negative favors the selected model.
- Statistical mixture: rolling_poisson=0.203, poisson_1=0.220, poisson_0.1=0.143, logistic_0.05=0.147, logistic_0.5=0.060, random_forest=0.227; temperature=1.0; market weight=1.0.

### Premier League

- History: 2018-08-10 to 2026-09-14; 3,080 matches.
- Confirmation: 2025-08-15 to 2026-05-24; 380 matches.
- Protocol: Untouched season 2025/2026; 3 expanding validation folds in preceding 2 seasons.
- Statistical log-loss difference against rolling reference: -0.0256; paired day-bootstrap 95% interval [-0.0516, -0.0013]. Negative favors the selected model.
- Statistical mixture: poisson_1=0.256, poisson_0.1=0.354, logistic_0.05=0.058, logistic_0.5=0.156, boosted_poisson=0.043, random_forest=0.133; temperature=1.0; market weight=1.0.

### Serie A

- History: 2018-08-18 to 2026-09-14; 3,080 matches.
- Confirmation: 2025-08-23 to 2026-05-24; 380 matches.
- Protocol: Untouched season 2025/2026; 3 expanding validation folds in preceding 2 seasons.
- Statistical log-loss difference against rolling reference: -0.0476; paired day-bootstrap 95% interval [-0.0754, -0.0179]. Negative favors the selected model.
- Statistical mixture: poisson_1=0.020, poisson_0.1=0.149, logistic_0.05=0.170, boosted_poisson=0.050, random_forest=0.159, catboost_4=0.451; temperature=1.0; market weight=1.0.

## Interpretation limits

- Double chance covers two of three outcomes and is therefore materially easier than exact 1-X-2. Its measured accuracy is not an exact-result accuracy.
- A 70% selective result is a historical group rate, not a guarantee for one match.
- Market-assisted mode requires three contemporaneous pre-match decimal odds. The historical evaluation uses the source's closing or available market-average snapshot.
- The system has no dated lineup, injury, transfer or xG feed yet.
- Goal totals and both-teams-to-score outputs have not been separately calibrated or confirmed.
- No betting profitability, transaction-cost edge or superiority to the market is established.
- Production artifacts are refitted on all available results after evaluation; their future performance remains unknown.
