# v3.4: setup and release checklist

This is a research application, not a validated profitable betting system. No money is staked automatically. Historical accuracy, a high probability and a profitable price are different quantities.

## Start locally

```bash
python -m pip install -r requirements-tested.txt
python -m match_predictor doctor
python -m streamlit run streamlit/user_interface.py
```

Only load model artifacts from trusted sources: joblib/pickle can execute code. Fresh clones require trained artifacts; prepared packages contain the previous six domestic models plus the experimental Champions League model.

## Configure live context

Set your licensed API-Football key locally. Never paste it into a chat, notebook, tracked file or commit.

PowerShell:

```powershell
$env:API_FOOTBALL_KEY="YOUR_KEY"
```

macOS/Linux:

```bash
export API_FOOTBALL_KEY="YOUR_KEY"
```

Use a real provider fixture ID in the **Injuries & Lineups** refresh form, or:

```bash
python -m match_predictor collect --league super_lig --season 2026 --from-date 2026-09-21 --to-date 2026-09-27 --details
```

Adapt the date range. Each detailed fixture makes four requests by default (metadata, lineups, injuries, odds); player stats add another. Check your subscription's coverage and quota before season-wide detailed requests. This release does not schedule unattended jobs.

The panel retains injury/suspension reasons as reported by the provider; it does not infer suspension eligibility or recovery dates. Missing responses, failed requests and unconfirmed XIs remain visible. An empty injury list is not proof that everyone is fit. The six-hour freshness gate is an operational default, not an empirically optimized threshold.

Collect completed-match player statistics separately with `--include-player-stats --without-odds`. Completed-match data is never automatically treated as information available before that match. The recent-player table covers stored fixtures only, not necessarily the full recent schedule.

## Train the four additional leagues from licensed history

All ten catalog keys are accepted by training. For each league and historical season, discover fixtures without `--details` to avoid unnecessary per-fixture calls:

```bash
python -m match_predictor collect --league super_lig --season 2023
```

Import the **actual saved listing path printed by that command**:

```bash
python -m match_predictor import-api-history --league super_lig --input PATH_TO_SAVED_LISTING.json --output-dir data/licensed_history
```

Repeat for multiple completed seasons. Then:

```bash
python -m match_predictor train --leagues super_lig --data-dir data/licensed_history
```

Repeat for `primeira_liga`, `belgian_pro_league`, `scottish_premiership`. The importer includes only finished regulation-time results for the requested league and refuses to overwrite existing seasons. To refresh a season, rebuild into a new directory and train against that complete directory. Do not mix provider aliases with the old Football-Data files. New bundles retain the history directory for the team dashboard; retrain or restore the directory after moving machines.

Imported fixture lists contain results, not historical odds or shots. Consequently this import path cannot validate an odds blend or recreate a historic injury feed. Importing today's injuries for yesterday's game is not a valid backtest.

The old Football-Data downloader remains available but its current source terms restrict automated/AI data-training uses. No additional Football-Data download was performed in this update. Obtain appropriate permission or a licensed alternative: https://www.football-data.co.uk/data.php

## Link context to a forecast

Use the collected fixture ID in **Match Forecast**. League ID, exact home/away names and UTC match date must match the selected model. Cross-provider aliases are deliberately rejected rather than guessed; use the same provider for training/context or resolve identities before linking.

The default context gate can suppress the selective signal. It does not change 1-X-2 or double-chance probabilities. Existing benchmark percentages do not measure the new gate. API odds are displayed as stored context, not automatically injected as contemporaneous manually entered odds.

## Still required before claiming live readiness

1. Configure a valid key and verify live coverage/quota for each chosen league.
2. Import licensed history and train/evaluate the four missing league models.
3. Accumulate timestamped pre-kickoff snapshots. Test injury/lineup features against the baseline using chronological holdouts, including missing-data cases. Only then enable learned context adjustments.
4. Run forward paper forecasts with frozen timestamps and record every match, including abstentions. Assess calibration, log loss, coverage and profitability separately.
5. For public hosting, add authentication, API budget controls and an operational refresh schedule; the current local collection button is not a production multi-user backend.

No %70 exact all-match accuracy, betting edge or positive return has been established. Current tests validate code behavior, not profitable betting performance.
