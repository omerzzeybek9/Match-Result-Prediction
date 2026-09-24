"""Read-only readiness report; never prints credentials or makes API calls."""
import os
from pathlib import Path

import joblib
import pandas as pd

from .data import API_LEAGUES, ROOT
from .dashboard import load_context_records


def readiness_report(model_dir=None, context_dir=None):
    rows = []
    for key, (name, _) in API_LEAGUES.items():
        path = Path(model_dir or ROOT / "artifacts") / f"{key}.joblib"
        row = {"league": key, "name": name, "status": "missing_model"}
        if path.exists():
            try:
                bundle = joblib.load(path)
                latest = pd.Timestamp(bundle["last_match_date"], tz="UTC")
                age = (pd.Timestamp.now(tz="UTC").normalize() - latest).days
                row.update({"last_result": str(latest.date()), "age_days": age,
                            "status": "current" if bundle.get("schema_version") == 3 and 0 <= age <= 14 else "needs_refresh"})
            except Exception:
                row["status"] = "unreadable_model"
        rows.append(row)
    return {"api_key_configured": bool(os.environ.get("API_FOOTBALL_KEY")),
            "context_snapshots": len(load_context_records(context_dir)), "leagues": rows,
            "injury_adjusted_model_validated": False, "profitability_established": False,
            "note": "Current files do not establish live API coverage, calibration or betting profitability."}
