"""
Wrapper around Google's Generative Language API for Gemma models.

Uses the REST endpoint directly (no SDK dependency) so it's easy to swap
to a local Ollama endpoint later if you want to add the offline story back
in after the hackathon.

Docs: https://ai.google.dev/api/generate-content
"""

import base64
import json
import os
import requests

GEMMA_API_KEY = "AQ.Ab8RN6KKlbrXlgsHn6ZxUMHXGjgzqpqHbongXco2ALMuk5J_Jw"#os.environ.get("GEMMA_API_KEY", "")
# IMPORTANT: run list_models.py first to see which exact model IDs your key
# can access, then set GEMMA_MODEL to that exact string in .env.
# Gemma 4 sizes are named by capability (E2B, E4B, 12B, 26B MoE, 31B dense),
# not by parameter count like Gemma 3 was -- don't guess this string.
MODEL = "gemma-4-12B-it"#os.environ.get("GEMMA_MODEL", "gemma-3-27b-it")
BASE_URL = f"https://generativelanguage.googleapis.com/v1beta/models/{MODEL}:generateContent"
REQUEST_TIMEOUT = int(os.environ.get("GEMMA_TIMEOUT_SECONDS", "120"))

TRIAGE_SCHEMA = {
    "type": "object",
    "properties": {
        "severity": {"type": "string", "enum": ["low", "moderate", "high", "emergency"]},
        "likely_conditions": {"type": "array", "items": {"type": "string"}},
        "red_flags": {"type": "array", "items": {"type": "string"}},
        "recommended_action": {"type": "string"},
        "referral_needed": {"type": "boolean"},
        "reasoning": {"type": "string"},
    },
    "required": [
        "severity",
        "likely_conditions",
        "red_flags",
        "recommended_action",
        "referral_needed",
        "reasoning",
    ],
}

SYSTEM_INSTRUCTION = (
    "You are a clinical triage assistant supporting a community health worker "
    "in a low-resource rural setting who is NOT a doctor. You are not making a "
    "diagnosis; you are helping them decide urgency and next steps. "
    "Be conservative: if in doubt, escalate severity and recommend referral. "
    "Always mention any visible red flags (e.g. spreading redness, fever signs, "
    "necrosis, heavy bleeding, signs of infection) explicitly. "
    "Respond ONLY with JSON matching the provided schema."
)


class GemmaClientError(Exception):
    pass


def _call_gemini(parts: list, response_schema: dict) -> dict:
    if not GEMMA_API_KEY:
        raise GemmaClientError(
            "GEMMA_API_KEY is not set. Export it or put it in a .env file before running."
        )

    payload = {
        "contents": [{"role": "user", "parts": parts}],
        "systemInstruction": {"parts": [{"text": SYSTEM_INSTRUCTION}]},
        "generationConfig": {
            "responseMimeType": "application/json",
            "responseSchema": response_schema,
            "temperature": 0.2,
        },
    }

    resp = requests.post(
        f"{BASE_URL}?key={GEMMA_API_KEY}",
        headers={"Content-Type": "application/json"},
        data=json.dumps(payload),
        timeout=REQUEST_TIMEOUT,
    )

    if resp.status_code != 200:
        raise GemmaClientError(f"Gemma API error {resp.status_code}: {resp.text}")

    data = resp.json()
    try:
        text = data["candidates"][0]["content"]["parts"][0]["text"]
        return json.loads(text)
    except (KeyError, IndexError, json.JSONDecodeError) as e:
        raise GemmaClientError(f"Unexpected Gemma response shape: {e}\nRaw: {data}")


def run_triage(symptom_text: str, image_bytes: bytes | None = None, image_mime: str = "image/jpeg") -> dict:
    """
    Core multimodal call: symptom description (+ optional photo) -> structured
    TriageResult dict (see schema.TriageResult).
    """
    parts = [{"text": f"Community health worker's notes: {symptom_text}"}]

    if image_bytes:
        parts.append(
            {
                "inline_data": {
                    "mime_type": image_mime,
                    "data": base64.b64encode(image_bytes).decode("utf-8"),
                }
            }
        )
        parts.append(
            {"text": "A photo of the affected area is attached. Factor it into your assessment."}
        )

    return _call_gemini(parts, TRIAGE_SCHEMA)


def translate_text(text: str, target_language_name: str) -> str:
    """
    Plain-text translation helper (no schema needed, just a string back).
    target_language_name: e.g. "Hausa" or "Yoruba"
    """
    parts = [
        {
            "text": (
                f"Translate the following clinical guidance into {target_language_name}. "
                "Keep it simple, direct, and appropriate for a community health worker "
                "to read aloud to a patient. Return ONLY the translation, nothing else.\n\n"
                f"Text: {text}"
            )
        }
    ]
    payload = {
        "contents": [{"role": "user", "parts": parts}],
        "generationConfig": {"temperature": 0.1},
    }
    resp = requests.post(
        f"{BASE_URL}?key={GEMMA_API_KEY}",
        headers={"Content-Type": "application/json"},
        data=json.dumps(payload),
        timeout=REQUEST_TIMEOUT,
    )
    if resp.status_code != 200:
        raise GemmaClientError(f"Gemma API error {resp.status_code}: {resp.text}")
    data = resp.json()
    return data["candidates"][0]["content"]["parts"][0]["text"].strip()
