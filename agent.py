"""
Autonomous agent layer.

Takes a structured TriageResult and, without further human input, decides:
  - whether a referral note is needed, and drafts it
  - a follow-up interval, and creates a schedulable event

This is what elevates the project from "a model that outputs text" to
"a system that takes action" -- matching the hackathon's own "generates
calendar invites" example.
"""

import json
import os
from datetime import datetime, timedelta

FOLLOW_UP_DAYS_BY_SEVERITY = {
    "low": 7,
    "moderate": 3,
    "high": 1,
    "emergency": None,  # immediate referral, no scheduled follow-up
}

CALENDAR_PATH = os.path.join(os.path.dirname(__file__), "data", "follow_ups.json")


def _facility_for_severity(severity: str) -> str:
    return {
        "low": "Monitor at community clinic",
        "moderate": "Refer to nearest primary health center",
        "high": "Refer to district hospital within 24 hours",
        "emergency": "Immediate referral to nearest hospital / emergency transport",
    }.get(severity, "Refer to primary health center")


def build_referral_note(triage: dict, patient_ref: str = "Patient") -> dict:
    severity = triage["severity"]
    follow_up_days = FOLLOW_UP_DAYS_BY_SEVERITY.get(severity, 3)

    note = {
        "patient_summary": (
            f"{patient_ref}: severity assessed as {severity}. "
            f"Likely conditions: {', '.join(triage['likely_conditions']) or 'unclear'}. "
            f"Red flags: {', '.join(triage['red_flags']) or 'none noted'}. "
            f"Reasoning: {triage['reasoning']}"
        ),
        "urgency": severity,
        "facility_recommendation": _facility_for_severity(severity),
        "follow_up_in_days": follow_up_days,
        "generated_at": datetime.utcnow().isoformat() + "Z",
    }
    return note


def schedule_follow_up(note: dict, patient_ref: str = "Patient") -> dict:
    """
    Appends a follow-up event to a local JSON 'calendar' file. In a fuller
    build this would push to a real calendar API; for the demo, a local
    schedule that the UI can render as an agenda is enough to prove the
    concept and works fully offline/cloud-agnostic.
    """
    os.makedirs(os.path.dirname(CALENDAR_PATH), exist_ok=True)

    event = None
    if note["follow_up_in_days"] is not None:
        due_date = (datetime.utcnow() + timedelta(days=note["follow_up_in_days"])).date().isoformat()
        event = {
            "patient": patient_ref,
            "due_date": due_date,
            "urgency": note["urgency"],
            "facility": note["facility_recommendation"],
            "created_at": note["generated_at"],
        }

        events = []
        if os.path.exists(CALENDAR_PATH):
            with open(CALENDAR_PATH, "r") as f:
                try:
                    events = json.load(f)
                except json.JSONDecodeError:
                    events = []
        events.append(event)
        with open(CALENDAR_PATH, "w") as f:
            json.dump(events, f, indent=2)

    return event
