"""Auditable context checks. These are data-quality gates, not probability boosts."""
from __future__ import annotations

from datetime import datetime, timezone


def utc_time(value):
    try:
        result = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
    except (ValueError, TypeError):
        return None
    return result.astimezone(timezone.utc) if result.tzinfo is not None else None


def assess_context(record, as_of=None, max_age_hours=6):
    """Fail closed on unknown timestamps, unavailable endpoints and live results."""
    now = utc_time(as_of) if as_of is not None else datetime.now(timezone.utc)
    if now is None:
        raise ValueError("as_of must include a timezone")
    issues = []
    if not record:
        return {"ready": False, "issues": ["No fixture context collected."], "age_hours": None,
                "model_uses_context": False}
    captured, kickoff = utc_time(record.get("captured_at")), utc_time(record.get("kickoff_utc"))
    age = (now - captured).total_seconds() / 3600 if captured else None
    if captured is None or kickoff is None:
        issues.append("Missing or invalid timezone-aware timestamps.")
    elif captured >= kickoff or now >= kickoff:
        issues.append("Fixture has started or this snapshot was captured after kickoff.")
    if age is not None and (age < 0 or age > max_age_hours):
        issues.append("Context timestamp is in the future or older than the freshness limit.")
    if record.get("prematch_safe") is not True:
        issues.append("Snapshot is not pre-match safe.")
    if record.get("fixture_status") != "NS":
        issues.append("Fixture is not scheduled for pre-match use.")
    endpoints = record.get("endpoint_status", {})
    if endpoints.get("injuries") != "available":
        issues.append("Injury coverage is unverified; an empty response does not prove a healthy squad.")
    if not all(record.get("lineups", {}).get(side, {}).get("confirmed") for side in ("home", "away")):
        issues.append("Both starting XIs are not confirmed.")
    return {"ready": not issues, "issues": issues, "age_hours": age,
            "model_uses_context": False, "captured_at": record.get("captured_at"),
            "fixture_id": record.get("fixture_id")}


def latest_fixture_context(records, fixture_id, as_of=None):
    """Select the latest snapshot available by the requested information cutoff."""
    cutoff = utc_time(as_of) if as_of is not None else datetime.now(timezone.utc)
    if cutoff is None:
        raise ValueError("as_of must include a timezone")
    candidates = [row for row in records if row.get("fixture_id") == fixture_id
                  and utc_time(row.get("captured_at")) is not None
                  and utc_time(row["captured_at"]) <= cutoff]
    return max(candidates, key=lambda row: utc_time(row["captured_at"])) if candidates else None


def availability_rows(record):
    """Retain fixture, team and reason instead of treating historical mentions as current injuries."""
    rows = []
    for player in (record or {}).get("injuries", {}).get("players", []):
        side = player.get("team")
        rows.append({"Team": record.get(f"{side}_team"), "Player": player.get("player_name"),
                     "Player ID": player.get("player_id"), "Reported type": player.get("type"),
                     "Reason": player.get("reason"), "Captured at": record.get("captured_at")})
    return rows
