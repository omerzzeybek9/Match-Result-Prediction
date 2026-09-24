# Research-backed roadmap for every-match forecasts

Research date: 2026-09-21

## What the evidence supports

Exact pre-match home/draw/away prediction is a difficult three-class problem. Ren and Susnjak's review and experiments report that pre-match accuracy is consistently below 60% in the surveyed literature and is almost always below 55% over multiple seasons. Their higher-confidence strategy deliberately focuses on easier fixtures rather than every match.

A six-season Premier League study combining structured statistics with journalist previews reached 63.18% exact 1-X-2 accuracy. This is a useful stretch benchmark for richer contextual inputs, but it is still below 70%.

A 2026 study reported 70.2%, but it forecast matches in-play at minute intervals using kick-off market calibration and post-shot xG. It is not evidence that 70% exact accuracy is attainable for every match before kick-off.

For decision support, probability calibration and log loss matter alongside accuracy. A calibrated 45% estimate can be useful even when its most likely class loses; an overconfident and poorly calibrated model is dangerous despite a similar headline accuracy.

Primary research:

- [Predicting Football Match Outcomes with eXplainable Machine Learning and the Kelly Index](https://arxiv.org/abs/2211.15734)
- [Combining Machine Learning and Human Experts to Predict Match Outcomes in Football](https://arxiv.org/abs/2012.04380)
- [A market-calibrated accelerated failure time model for in-play football forecasting](https://arxiv.org/abs/2605.16066)
- [Machine learning for sports betting: should model selection be based on accuracy or calibration?](https://arxiv.org/abs/2303.06021)

## What this repository now proves

The untouched 2025/26 confirmation set contains 2,058 matches from six domestic leagues.

| Forecast | Coverage | Statistics only | Market assisted |
|---|---:|---:|---:|
| Exact 1-X-2 | 100% | 51.2% | 53.4% |
| Double chance | 100% | 77.3% | 79.3% |
| Selective exact 1-X-2 | 15.5% / 19.6% | 72.6% | 73.3% |

The every-match double-chance rule excludes only the lowest-probability outcome. It was not tuned on confirmation labels. Its market-assisted 95% interval is 77.4%–80.9%; individual league results range from 78.4% to 80.5%. The strongest constant pick (`12` for every fixture) scores 74.4%, so the match-specific rule adds 4.8 percentage points. Double chance is an easier two-of-three target and must never be described as 79.3% exact-result accuracy.

## Data required for the next exact-result improvement

The current model lacks the information that changes close fixtures shortly before kick-off. The next data layer should store timestamped snapshots rather than only the final value:

1. Expected and confirmed starting elevens, minutes before kick-off, and player-level strength.
2. Injuries, suspensions, doubtful status and expected return date.
3. Team and player xG, non-penalty xG, xG conceded and post-shot xG where licensed.
4. Opening, 24-hour, 6-hour, 1-hour and closing 1-X-2 plus totals odds from several bookmakers.
5. Transfers, manager changes, days of rest across all competitions, travel and fixture congestion.
6. Weather and referee features only after the higher-value inputs above are stable.

## Provider decision

### Lowest-cost operational start: API-Football

The official pricing page lists all competitions and endpoints on paid plans. The $19/month plan currently includes 7,500 requests per day and exposes fixtures, lineups, transfers, sidelined players, injuries, pre-match/in-play odds and statistics. This is the practical first feed for live operation and timestamped data collection.

- [API-Football pricing and included endpoints](https://www.api-football.com/pricing)

### Richer single-provider option: Sportmonks

Sportmonks advertises lineups, injuries and suspensions, expected lineups, pre-match and in-play odds, predictions, and xG-family metrics. Published plans start at €29/month for five leagues; xG, odds, historical access and additional leagues have separately published starting prices. This is the stronger candidate when historical xG and expected-lineup depth justify the higher spend.

- [Sportmonks Football API, coverage and pricing](https://www.sportmonks.com/football-api/)

### Specialized odds archive: The Odds API

The Odds API offers multi-bookmaker soccer odds and historical snapshots back to 2020. Paid access currently starts at $30/month for 20,000 credits. It is useful if consistent market snapshots are not adequate in the main football feed.

- [The Odds API pricing and coverage](https://the-odds-api.com/)

StatsBomb Open Data is useful for event-model prototyping, but it contains selected competitions and seasons and is not a current production feed for all six leagues.

- [StatsBomb Open Data](https://github.com/statsbomb/open-data)

## Implementation sequence

1. Use the new double-chance output for every fixture and retain exact probabilities without claiming a 70% exact hit rate.
2. **Implemented in v3.2:** add an API-Football snapshot collector and immutable raw store. Run it at scheduled pre-match offsets so training and live inference see information from comparable times.
3. **Implemented in v3.3:** expose a ten-league API catalog and an English dashboard for forecast, team history, league tables and player snapshots. The four leagues without legacy training files remain explicitly marked as data-collection pending.
4. Resolve team and player identities across seasons, then calculate availability-weighted squad strength and expected-lineup uncertainty.
5. Train a market baseline, Dixon-Coles/Poisson score model and calibrated tree model. Blend them using only expanding historical validation windows.
6. Reforecast when the confirmed lineup arrives. Keep the earlier forecast so the improvement can be measured honestly.
7. Promote a feature only when it improves forward log loss, Brier score or calibration across several leagues. Track exact accuracy and double-chance accuracy separately.
8. Consider a separate in-play model only after event/xG latency and licensing are reliable.

The realistic near-term target is 55–60% exact 1-X-2 across every match, with 60–63% as a strong stretch goal. The current evidence supports roughly 78–82% for the separate every-match double-chance task. A 70% exact target becomes defensible only as a selective or in-play target unless future forward tests demonstrate otherwise.
