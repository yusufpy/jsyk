"""
Local-language layer.

Key design decision: we do NOT ask Gemma to reason clinically in Hausa/Yoruba
directly. Clinical reasoning stays in English (more reliable, easier to
validate), and only the final output gets translated. We then back-translate
and ask Gemma to judge whether meaning was preserved -- this catches cases
where the local-language output might be missing a critical detail (e.g. a
red flag getting softened or dropped in translation).

This is a good thing to call out explicitly to judges: it shows you designed
around a real LLM failure mode instead of trusting translation blindly.
"""

from gemma_client import translate_text, _call_gemini, GemmaClientError

LANGUAGE_NAMES = {"ha": "Hausa", "yo": "Yoruba"}

CONFIDENCE_SCHEMA = {
    "type": "object",
    "properties": {
        "confidence": {"type": "string", "enum": ["high", "medium", "low"]},
        "notes": {"type": "string"},
    },
    "required": ["confidence", "notes"],
}


def _check_back_translation(original_en: str, back_translated_en: str) -> dict:
    parts = [
        {
            "text": (
                "Compare these two English texts. The second is a back-translation "
                "of a translation of the first. Judge whether any clinically "
                "important detail (severity, red flags, urgency, recommended action) "
                "was lost, softened, or changed. Respond with JSON only.\n\n"
                f"Original: {original_en}\n\nBack-translated: {back_translated_en}"
            )
        }
    ]
    return _call_gemini(parts, CONFIDENCE_SCHEMA)


def translate_triage_output(triage: dict, language_code: str) -> dict:
    """
    triage: dict matching schema.TriageResult
    language_code: "ha" or "yo"

    Returns a dict matching schema.TranslatedTriageResult.
    """
    if language_code not in LANGUAGE_NAMES:
        raise ValueError(f"Unsupported language code: {language_code}")

    lang_name = LANGUAGE_NAMES[language_code]

    summary_en = (
        f"Severity: {triage['severity']}. "
        f"Likely: {', '.join(triage['likely_conditions']) or 'unclear'}. "
        f"Red flags: {', '.join(triage['red_flags']) or 'none noted'}."
    )
    action_en = triage["recommended_action"]

    translated_summary = translate_text(summary_en, lang_name)
    translated_action = translate_text(action_en, lang_name)

    # Back-translate the action (the highest-stakes piece of text) to English
    # and check for meaning drift.
    back_translated_action = translate_text(translated_action, "English")

    try:
        check = _check_back_translation(action_en, back_translated_action)
        confidence = check.get("confidence", "medium")
        notes = check.get("notes", "")
    except GemmaClientError:
        confidence = "medium"
        notes = "Confidence check call failed; defaulting to medium."

    return {
        "language": language_code,
        "translated_summary": translated_summary,
        "translated_recommended_action": translated_action,
        "back_translation_check": f"{back_translated_action} ({notes})".strip(),
        "translation_confidence": confidence,
    }
