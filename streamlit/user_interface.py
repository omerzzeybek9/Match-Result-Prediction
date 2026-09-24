"""Run with: python -m streamlit run streamlit/user_interface.py"""
from datetime import date, timedelta
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import joblib
import pandas as pd
import streamlit as st

from match_predictor.dashboard import league_table, player_dashboard, team_history, team_summary, load_context_records
from match_predictor.context import assess_context, availability_rows, latest_fixture_context
from match_predictor.api_football import ApiFootballClient, SnapshotStore, collect_fixture
from match_predictor.data import API_LEAGUES, LEAGUES, load_matches
from match_predictor.predict import predict_match

st.set_page_config(page_title="Match Result Intelligence", page_icon="⚽", layout="wide")
st.title("⚽ Match Result Intelligence")
st.caption("Evidence-based match probabilities, team history, league tables and player form")


@st.cache_resource
def load_bundle(path, modified):
    return joblib.load(path)


@st.cache_data(ttl=60)
def load_team_history(league, team, directory=None, limit=20):
    return team_history(league, team, limit=limit, data_dir=directory)


@st.cache_data(ttl=60)
def load_league_table(league, season, directory=None):
    return league_table(league, season=season, data_dir=directory)


@st.cache_data(ttl=60)
def load_player_dashboard(team, league_id, window):
    return player_dashboard(team, league_id=league_id, window=window)


@st.cache_data(ttl=60)
def context_records():
    return load_context_records()


def pct(value):
    return "—" if value is None else f"{100 * value:.1f}%"


def available_models():
    return [name for name in LEAGUES if (ROOT / "artifacts" / f"{name}.joblib").exists()]


available = available_models()
st.sidebar.header("Model and data")
if st.sidebar.button("Refresh local data"):
    st.cache_data.clear()
    st.rerun()
if not available:
    st.info("No trained model is available yet. Run the following commands first.")
    st.code("python -m match_predictor download\npython -m match_predictor train", language="bash")
    st.stop()

league = st.sidebar.selectbox("Model league", available, format_func=lambda key: LEAGUES[key][0])
path = ROOT / "artifacts" / f"{league}.joblib"
try:
    bundle = load_bundle(str(path), path.stat().st_mtime_ns)
except (ImportError, ValueError, OSError) as error:
    st.error(f"Model could not be loaded: {error}. Install requirements-tested.txt or retrain this model.")
    st.stop()
history_directory = bundle.get("history_directory")
if bundle.get("schema_version") != 3:
    st.error("This model uses an old schema. Run `python -m match_predictor train` again.")
    st.stop()

st.sidebar.caption(f"Latest result: {bundle['last_match_date']}")
st.sidebar.caption(f"Historical matches: {bundle['report']['data']['matches']:,}")
show_all = st.sidebar.checkbox("Include teams from older seasons")
teams = bundle["teams"] if show_all else bundle["recent_teams"]
latest = date.fromisoformat(bundle["last_match_date"])
age = (date.today() - latest).days
if age > 14:
    st.warning(f"The result feed is {age} days old. The selective exact-result filter may abstain until data is refreshed.")
if league == "cl":
    st.warning("The Champions League model uses a small legacy dataset; the selective filter is disabled.")

with st.sidebar.expander("Top-10 league coverage", expanded=False):
    catalog = []
    for key, (name, api_id) in API_LEAGUES.items():
        catalog.append({"League": name, "API-Football ID": api_id,
                        "Model": "Ready" if key in available else "Data collection pending"})
    st.dataframe(pd.DataFrame(catalog), hide_index=True, width="stretch")
    st.caption("Models become available after a leakage-safe historical dataset is collected and trained.")

forecast_tab, team_tab, player_tab, context_tab, evaluation_tab = st.tabs([
    "Match Forecast", "Team Dashboard", "Player Dashboard", "Injuries & Lineups", "Model Evaluation"
])
st.caption("Research tool · No proven betting profitability · Probabilities are not guarantees.")

with forecast_tab:
    left, right, when = st.columns([2, 2, 1])
    home = left.selectbox("Home team", teams, key="forecast_home")
    away_options = [team for team in teams if team != home]
    away = right.selectbox("Away team", away_options, key="forecast_away")
    match_date = when.date_input("Match date", value=max(date.today(), latest + timedelta(days=1)),
                                 min_value=latest + timedelta(days=1))
    use_odds = st.checkbox("Add current 1-X-2 decimal odds (recommended)")
    require_context = st.checkbox("Require fresh fixture context for the selective signal", value=True)
    fixture_id = st.number_input("API fixture ID (0 = no linked snapshot)", min_value=0, value=0, step=1)
    st.caption("A linked snapshot must match the model's exact team names, league and date. Name mismatches are rejected, not guessed.")
    odds = None
    if use_odds:
        st.caption("Enter home, draw and away prices from the same market snapshot.")
        o1, ox, o2 = st.columns(3)
        home_odds = o1.number_input(f"{home} odds", min_value=1.01, max_value=100.0, value=2.20, step=.01)
        draw_odds = ox.number_input("Draw odds", min_value=1.01, max_value=100.0, value=3.40, step=.01)
        away_odds = o2.number_input(f"{away} odds", min_value=1.01, max_value=100.0, value=3.20, step=.01)
        odds = [home_odds, draw_odds, away_odds]
    if st.button("Predict match", type="primary", width="stretch"):
        try:
            linked_context = latest_fixture_context(context_records(), int(fixture_id)) if fixture_id else None
            result = predict_match(bundle, home, away, match_date, odds,
                                   context=linked_context, require_context=require_context)
        except ValueError as error:
            st.error(str(error))
        else:
            for warning in result["warnings"]:
                st.warning(warning)
            st.caption("Context checks affect the selective signal only. They do not change probabilities or validate a betting edge.")
            double_chance = result["double_chance"]
            st.info(f"Every-match estimate: **{double_chance['label']} ({double_chance['code']})** · model probability {pct(double_chance['probability'])}")
            mode_report = (bundle.get("selective_report") or {}).get("modes", {}).get(result["prediction_mode"], {})
            full_coverage = mode_report.get("all_match_double_chance")
            if full_coverage:
                league_result = mode_report.get("all_match_double_chance_by_league", {}).get(league)
                evidence = (f"In the independent 2025/26 test, it covered {full_coverage['total_matches']:,} matches "
                            f"with {pct(full_coverage['accuracy'])} accuracy at 100% coverage. "
                            f"Best fixed double chance: {pct(full_coverage['best_constant_accuracy'])}.")
                if league_result:
                    evidence += f" {LEAGUES[league][0]}: {pct(league_result['accuracy'])} ({league_result['correct']}/{league_result['total_matches']})."
                st.caption(evidence + " Double chance covers two outcomes; it is a different target from exact 1-X-2.")
            verdict = result["selection"]
            if verdict["selected"]:
                st.success(f"Selective exact 1-X-2: **{result['predicted_outcome']}** · confidence policy passed")
            else:
                st.info("Selective exact 1-X-2: **ABSTAIN**")
                for reason in verdict["reasons"]:
                    st.caption(f"• {reason}")
            st.caption(verdict["note"])
            st.subheader(f"{home} — {away}")
            probabilities = result["probabilities"]
            columns = st.columns(3)
            for column, label, value in zip(columns, [f"{home} win", "Draw", f"{away} win"], probabilities.values()):
                column.metric(label, f"{100 * value:.1f}%")
            st.bar_chart(pd.DataFrame({"Probability (%)": [100 * probabilities["home"], 100 * probabilities["draw"], 100 * probabilities["away"]]},
                                      index=[home, "Draw", away]), color="#16a085", horizontal=True)
            if result["market_probabilities"]:
                with st.expander("Statistical model and market consensus"):
                    labels = [home, "Draw", away]
                    statistical = result["statistical_probabilities"]
                    market = result["market_probabilities"]
                    st.dataframe(pd.DataFrame({"Outcome": labels,
                        "Statistical model (%)": [100 * v for v in statistical.values()],
                        "Market implied (%)": [100 * v for v in market.values()],
                        "Blended prediction (%)": [100 * v for v in probabilities.values()]}).round(1), hide_index=True, width="stretch")
                    st.caption("The bookmaker margin is removed before probabilities are blended.")
            c1, c2 = st.columns(2)
            with c1:
                st.markdown("**Most likely scorelines**")
                st.dataframe(pd.DataFrame([{"Score": f"{score['home']}–{score['away']}", "Probability": f"{100 * score['probability']:.1f}%"}
                                           for score in result["top_scores"]]), hide_index=True, width="stretch")
                st.caption("The most likely scoreline can still have a low absolute probability.")
            with c2:
                expected = result["expected_goals"]
                st.metric("Expected goals", f"{expected['home']:.2f} — {expected['away']:.2f}")
                st.caption("These are model goal means, not event-level xG from shot locations.")
                st.metric("Over 2.5 goals", f"{100 * result['over_2_5']:.1f}%")
                st.metric("Both teams to score", f"{100 * result['both_teams_score']:.1f}%")
            with st.expander("Team strength and form summary"):
                form = result["form"]
                st.dataframe(pd.DataFrame({"Team": [home, away], "Elo": [round(form["home_elo"]), round(form["away_elo"])],
                    "Adjusted points per game (last 5)": [round(form["home_points_last5"], 2), round(form["away_points_last5"], 2)]}), hide_index=True)
            st.caption("Probabilities are estimates, not a guarantee. Lineup and injury snapshots are collected separately and enter the model only after forward validation.")

with team_tab:
    st.subheader("Team history and league table")
    selected_team = st.selectbox("Team", teams, key="team_dashboard_team")
    try:
        history = load_team_history(league, selected_team, history_directory)
    except (FileNotFoundError, ValueError):
        history = pd.DataFrame()
    summary = team_summary(history)
    if history.empty:
        st.info("No completed fixtures were found for this team in the local history.")
    else:
        metrics = st.columns(6)
        metrics[0].metric("Matches", summary["matches"])
        metrics[1].metric("Record", summary["record"])
        metrics[2].metric("Points", summary["points"])
        metrics[3].metric("Points/game", f"{summary['points_per_game']:.2f}")
        metrics[4].metric("Goals", f"{summary['goals_for']}–{summary['goals_against']}")
        metrics[5].metric("Clean sheets", pct(summary["clean_sheet_rate"]))
        st.caption(f"Recent form (oldest → newest): {summary['form']}")
        display = history.rename(columns={"date": "Date", "opponent": "Opponent", "venue": "Venue", "score": "Score",
                                          "result": "Result", "points": "Points", "goals_for": "GF", "goals_against": "GA",
                                          "goal_difference": "GD", "shots_for": "Shots for", "shots_against": "Shots against",
                                          "sot_for": "Shots on target", "sot_against": "Opp. shots on target"})
        st.dataframe(display, hide_index=True, width="stretch")
    try:
        seasons = sorted(load_matches(league, history_directory).season.unique(), reverse=True)
        season = st.selectbox("Table season", seasons, key="table_season")
        table = load_league_table(league, int(season), history_directory)
        st.subheader(f"{LEAGUES[league][0]} table · {season}")
        st.caption("Computed from stored results: no point deductions, official head-to-head rules or playoff adjustments. Not official standings.")
        st.dataframe(table.rename(columns={"position": "#", "team": "Team", "played": "P", "wins": "W", "draws": "D",
                                           "losses": "L", "goals_for": "GF", "goals_against": "GA", "goal_difference": "GD",
                                           "points": "Pts"}), hide_index=True, width="stretch")
    except (FileNotFoundError, ValueError) as error:
        st.info(f"League table is not available: {error}")

with player_tab:
    st.subheader("Player form and availability")
    player_league = st.selectbox("Player data league", list(API_LEAGUES), format_func=lambda key: API_LEAGUES[key][0])
    player_league_id = API_LEAGUES[player_league][1]
    api_teams = sorted({row.get(f"{side}_team") for row in context_records()
                        if row.get("league_id") == player_league_id for side in ("home", "away")
                        if row.get(f"{side}_team")})
    player_team = st.selectbox("Provider team", api_teams or ["No snapshots collected"], key="player_dashboard_team")
    window = st.selectbox("Recent fixture window", [5, 10, 20], index=1)
    players, context = load_player_dashboard(player_team, player_league_id, window)
    st.caption("Aggregates cover the selected number of stored fixtures, not a complete season. Injury mentions are historical, not current medical status.")
    if players.empty:
        st.info("No API-Football snapshots were found for this team yet.")
        st.code('export API_FOOTBALL_KEY="..."\npython -m match_predictor collect --fixture FIXTURE_ID --include-player-stats', language="bash")
    else:
        st.caption(f"Snapshots: {context['snapshots']} · completed player statistics: {'available' if context['stats_available'] else 'not collected'}")
        if not context["stats_available"]:
            st.warning("Only lineup snapshots are available. Collect completed fixtures with `--include-player-stats` to populate minutes, ratings and goals.")
        columns = ["player", "player_id", "position", "snapshots", "lineup_starts", "starts", "appearances", "minutes", "goals", "assists",
                   "shots", "shots_on", "key_passes", "average_rating", "injury_mentions"]
        if not context["stats_available"]:
            columns = ["player", "player_id", "position", "snapshots", "lineup_starts", "injury_mentions"]
        shown = players[[column for column in columns if column in players]].rename(columns={
            "player": "Player", "player_id": "ID", "position": "Position", "snapshots": "Snapshots", "starts": "Completed starts", "lineup_starts": "Lineup-only starts",
            "appearances": "Apps", "minutes": "Minutes", "goals": "Goals", "assists": "Assists", "shots": "Shots",
            "shots_on": "Shots on target", "key_passes": "Key passes", "average_rating": "Avg rating",
            "injury_mentions": "Injury mentions"})
        st.dataframe(shown.round(2), hide_index=True, width="stretch")

with context_tab:
    st.subheader("Fixture availability and confirmed lineups")
    st.caption("Coverage depends on your provider plan. Empty and failed responses remain visibly unknown.")
    collect_id = st.number_input("Fixture ID to refresh", min_value=1, value=1, step=1)
    if st.button("Fetch fixture context from API-Football"):
        try:
            client = ApiFootballClient.from_environment()
            store = SnapshotStore(ROOT / "data/api_football")
            _, collected = collect_fixture(client, int(collect_id), store)
            store.save_normalized(collected)
        except (ValueError, RuntimeError, OSError) as error:
            st.error(f"Collection failed: {error}")
        else:
            st.cache_data.clear()
            st.success("Snapshot saved. Endpoint gaps are shown below.")
    context_league = st.selectbox("Context league", list(API_LEAGUES), format_func=lambda key: API_LEAGUES[key][0])
    records = [row for row in context_records() if row.get("league_id") == API_LEAGUES[context_league][1]]
    fixtures = {}
    for row in records:
        fixtures[row["fixture_id"]] = f"{row.get('kickoff_utc')} · {row.get('home_team')} — {row.get('away_team')} · #{row['fixture_id']}"
    if not fixtures:
        st.info("No collected fixtures in this league. Fetch a fixture above or use the collect CLI command.")
    else:
        selected_fixture = st.selectbox("Collected fixture", list(fixtures), format_func=lambda value: fixtures[value])
        record = latest_fixture_context(records, selected_fixture)
        quality = assess_context(record)
        if record is None:
            st.warning("No snapshot is available as of the current time.")
        else:
            st.caption(f"Captured: {record.get('captured_at')} · Status: {record.get('fixture_status') or 'unknown'}")
            for issue in quality["issues"]:
                st.warning(issue)
            st.dataframe(pd.DataFrame([{"Endpoint": key, "Status": value} for key, value in record.get("endpoint_status", {}).items()]), hide_index=True)
            availability = availability_rows(record)
            if availability:
                st.dataframe(pd.DataFrame(availability), hide_index=True, width="stretch")
            else:
                st.info("No availability records returned. This does not confirm that all players are fit or eligible.")
            for column, side in zip(st.columns(2), ("home", "away")):
                lineup = record.get("lineups", {}).get(side, {})
                column.subheader(record.get(f"{side}_team") or side.title())
                column.caption(f"Formation: {lineup.get('formation') or 'Unknown'} · XI complete: {bool(lineup.get('confirmed'))}")
                if lineup.get("players"):
                    column.dataframe(pd.DataFrame(lineup["players"]), hide_index=True)
            st.info("Availability is context only. No injury multiplier or unvalidated probability uplift is applied.")

with evaluation_tab:
    st.warning("These are historical baseline-policy metrics. The new injury/lineup freshness gate has not been backtested and changes selection coverage.")
    report = bundle["report"]
    selected = report["selected"]
    period = report["test_period"]
    st.subheader("Held-out full-season evaluation")
    st.caption(f"{period['from']} — {period['to']} · {period['matches']} matches")
    comparison = [{"Model": "Statistical model", "Accuracy (%)": 100 * selected["accuracy"], "Log loss": selected["log_loss"]},
                  {"Model": "Market-assisted", "Accuracy (%)": 100 * report["market_assisted"]["accuracy"], "Log loss": report["market_assisted"]["log_loss"]}]
    for name, baseline in report["baselines"].items():
        comparison.append({"Model": {"league_mean": "League mean", "rolling_poisson": "Historical goal averages"}[name],
                           "Accuracy (%)": 100 * baseline["accuracy"], "Log loss": baseline["log_loss"]})
    st.dataframe(pd.DataFrame(comparison).round(3), hide_index=True, width="stretch")
    st.caption("Lower log loss is better. The market-assisted row only uses matches with a complete historical odds snapshot.")
    selective = bundle.get("selective_report")
    if selective:
        st.subheader("Double chance on every match")
        rows = []
        for mode, label in [("stats", "Statistical only"), ("assisted", "Market-assisted")]:
            details = selective["modes"][mode]
            result = details.get("all_match_double_chance")
            if result:
                interval = result["accuracy_95_interval"]
                rows.append({"Mode": label, "Test matches": result["total_matches"], "Coverage": pct(result["coverage"]),
                             "Accuracy": pct(result["accuracy"]), "Best fixed pick": pct(result["best_constant_accuracy"]),
                             "Improvement": pct(result["improvement_vs_best_constant"]), "95% interval": f"{pct(interval[0])}–{pct(interval[1])}"})
        if rows:
            st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch")
            st.caption("The lowest-probability outcome is removed and the other two are reported. This is a separate, easier target from exact 1-X-2.")
        st.subheader("Selective exact 1-X-2 policy")
        rows = []
        for mode, label in [("stats", "Statistical only"), ("assisted", "Market-assisted")]:
            details = selective["modes"][mode]
            policy = details["policy"]
            confirmation = details["confirmation"]
            audit = details.get("latest_partial_season_audit")
            rows.append({"Mode": label, "Selection threshold": pct(policy.get("threshold")), "Selected test matches": confirmation["selected_matches"],
                         "Test coverage": pct(confirmation["coverage"]), "Test accuracy": pct(confirmation["accuracy"]),
                         "Latest-season accuracy": pct(audit["accuracy"]) if audit else "—",
                         "Latest-season selected": audit["selected_matches"] if audit else "—"})
        st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch")
        st.caption("The threshold was selected only on older validation matches. Coverage is the share of matches where the model does not abstain.")
    st.markdown("**Calibration by confidence bucket**")
    st.dataframe(pd.DataFrame([{"Confidence": bucket["range"], "Matches": bucket["matches"],
                               "Mean confidence (%)": round(100 * bucket["mean_probability"], 1),
                               "Observed accuracy (%)": round(100 * bucket["accuracy"], 1)}
                              for bucket in selected["confidence_buckets"]]), hide_index=True)
    st.info("Evaluation metrics come from a frozen historical test period. The live bundle is retrained on all available results.")
    with st.expander("Experiment details"):
        st.write(report["protocol"])
        st.json({"weights": report["weights"], "temperature": report["temperature"],
                 "market_weight": report["market_weight"], "validation_log_loss": report["validation_log_loss"]})
