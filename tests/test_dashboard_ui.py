"""Offline UI smoke checks; no provider requests or bets are made."""
import importlib.util
from pathlib import Path
import unittest
from unittest.mock import patch
from datetime import datetime, timedelta, timezone

ROOT = Path(__file__).resolve().parents[1]
HAS_STREAMLIT = importlib.util.find_spec("streamlit") is not None


@unittest.skipUnless(HAS_STREAMLIT and (ROOT / "artifacts/premier_league.joblib").exists(),
                     "Requires Streamlit and prepared artifacts")
class DashboardSmokeTests(unittest.TestCase):
    def test_initial_view_and_prediction(self):
        from streamlit.testing.v1 import AppTest
        app = AppTest.from_file(str(ROOT / "streamlit/user_interface.py"), default_timeout=60).run()
        self.assertFalse(app.exception)
        self.assertEqual(len(app.tabs), 5)
        next(button for button in app.button if button.label == "Predict match").click()
        app.run()
        self.assertFalse(app.exception)
        self.assertTrue(any("ABSTAIN" in item.value for item in app.info))

    def test_context_view_with_injury_record(self):
        import streamlit as st
        from streamlit.testing.v1 import AppTest
        now = datetime.now(timezone.utc)
        row = {"fixture_id": 99, "league_id": 39, "home_team": "Home", "away_team": "Away",
               "captured_at": (now - timedelta(minutes=10)).isoformat(),
               "kickoff_utc": (now + timedelta(hours=1)).isoformat(), "fixture_status": "NS",
               "prematch_safe": True, "endpoint_status": {"injuries": "available", "lineups": "empty"},
               "injuries": {"players": [{"team": "home", "player_id": 7, "player_name": "Player", "reason": "Suspended"}]},
               "lineups": {}, "player_stats": {}}
        with patch("match_predictor.dashboard.load_context_records", return_value=[row]):
            st.cache_data.clear()
            app = AppTest.from_file(str(ROOT / "streamlit/user_interface.py"), default_timeout=60).run()
            self.assertFalse(app.exception)
            self.assertTrue(any("Both starting XIs" in warning.value for warning in app.warning))
        st.cache_data.clear()
