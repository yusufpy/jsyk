"""
Structured data models for Ilera.

We force Gemma to return JSON matching TriageResult so the rest of the
pipeline (translation, referral generation, UI rendering) can rely on a
fixed shape instead of parsing free text.
"""

from typing import List, Optional
from pydantic import BaseModel, Field


class TriageResult(BaseModel):
    severity: str = Field(..., description="One of: low, moderate, high, emergency")
    likely_conditions: List[str] = Field(
        default_factory=list, description="Ranked list of plausible conditions, most likely first"
    )
    red_flags: List[str] = Field(
        default_factory=list, description="Specific danger signs observed, empty if none"
    )
    recommended_action: str = Field(..., description="Plain-language next step for the health worker")
    referral_needed: bool = Field(..., description="Whether this case should be referred to a clinic/hospital")
    reasoning: str = Field(..., description="Short clinical reasoning, 1-3 sentences")


class TranslatedTriageResult(BaseModel):
    original: TriageResult
    language: str  # "ha" or "yo"
    translated_summary: str
    translated_recommended_action: str
    back_translation_check: str
    translation_confidence: str  # "high", "medium", "low"


class ReferralNote(BaseModel):
    patient_summary: str
    urgency: str
    facility_recommendation: str
    follow_up_in_days: Optional[int]
    generated_at: str


class TriageRequest(BaseModel):
    symptom_text: str
    language: str = "en"  # "en", "ha" (Hausa), "yo" (Yoruba)
