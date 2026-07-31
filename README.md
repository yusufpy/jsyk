# Ilera — Field Triage Assistant

**Multimodal, multilingual, agentic triage support for community health workers — built on Gemma 4.**

> Built for the *Build with Gemma: AI for Africa* Hackathon (Minna, 2026)

🔗 **Live demo: [ilera.onrender.com](https://ilera.onrender.com)**

A community health worker describes symptoms — optionally attaching a photo
— and Gemma 4 returns a structured clinical triage assessment: severity,
likely conditions, red flags, and a recommended action. The output can be
translated into Hausa or Yoruba, with an automatic back-translation check to
catch meaning drift. If a referral is warranted, the system autonomously
drafts a referral note and schedules a follow-up date, with no further human
input required.


## Why this isn't "just a chat prompt"

- **Multimodal reasoning, not captioning.** Gemma 4 reasons jointly over a
  photo and free-text notes to produce a differential and flag risks a
  non-specialist might miss (e.g. the elevated danger of facial infections
  spreading via the "danger triangle").
- **Structured output that drives real logic.** Gemma's response is forced
  into a strict JSON schema via the API's `responseSchema` feature. That
  object — not free text — is what the referral and follow-up scheduling
  logic actually runs on.
- **Local language, done carefully.** Clinical reasoning stays in English
  internally (more reliable to validate). Only the final output is
  translated, and it's back-translated and confidence-checked before being
  shown — Gemma auditing Gemma's own translation. See `translate.py`.
- **Agentic follow-through.** The triage result triggers real downstream
  action: a referral note is drafted and a follow-up is scheduled
  automatically based on urgency. See `agent.py`.

## Setup

To run it locally:

```bash
cd ilera
python3 -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
pip install -r requirements.txt

cp .env.example .env
# edit .env and paste in your Gemma API key (from Google AI Studio)
```

### Find your exact model name first — don't guess it

Gemma 4 uses capability-tier naming (e.g. `gemma-4-31b-it`,
`gemma-4-26b-a4b-it`), not the parameter-count naming Gemma 3 used. Run this
before anything else:

```bash
python list_models.py
```

This queries Google's `ListModels` endpoint directly and prints every model
your key can access, flagged with whether it supports `generateContent`.
Copy the exact string into `GEMMA_MODEL` in your `.env`. This project was
built and tested against `gemma-4-31b-it`.

## Run

```bash
python -m uvicorn app:app --reload --port 8000
```

Open **http://localhost:8000** in your browser.

## Demo flow

1. Type a symptom description (or attach a photo of a skin condition, wound, etc.).
2. Pick an output language: English, Hausa, or Yoruba.
3. Click **Run triage**.
4. Review: severity, likely conditions, red flags, recommended action, the
   local-language translation with a confidence tag, an auto-generated
   referral note, and (if applicable) a scheduled follow-up date.

## Project structure

```
ilera/
├── app.py             # FastAPI routes tying the pipeline together
├── gemma_client.py     # Calls to Gemma 4 via Google's Generative Language API
├── schema.py           # Structured output shapes (Pydantic)
├── translate.py        # Local-language layer + back-translation confidence check
├── agent.py            # Referral note drafting + follow-up scheduling
├── list_models.py      # Utility: lists exact model IDs available to your key
├── static/index.html   # Demo frontend
├── requirements.txt
└── .env.example
```

## Engineering notes (things that broke, and why the fix was correct)

Built in a one-week sprint, so this section is deliberately honest about
what went wrong along the way - we think it's more useful to judges than
pretending it was smooth.

- **Model naming**: Gemma 4's naming convention is not backward-compatible
  with Gemma 3 guesses. `list_models.py` turns this into a diagnostic
  instead of trial-and-error against the API.
- **Gemma 4's "thinking" mode leaking into output**: Gemma 4 returns internal
  reasoning as separate response parts marked `"thought": true`, ahead of
  the real answer. We initially grabbed the first part returned, which was
  sometimes the reasoning scratchpad, not the answer — visible as garbled
  draft-translation text in early demo runs. Attempting to disable thinking
  via `generationConfig.thinkingConfig` proved unreliable (accepted in some
  call shapes, rejected with a 400 in others), so the fix lives entirely on
  our side: `gemma_client._extract_answer_text()` explicitly filters out any
  part marked as a thought, regardless of what the request config says.
  This is more robust than depending on an API flag we don't fully control.
- **Silent failures**: an early version caught translation errors but
  stored them in an unused response field, so failed requests still
  returned `200 OK` with no explanation. Fixed with explicit error
  surfacing at each pipeline stage rather than a silent English fallback.

## Notes for judges

- The "calendar" is a local JSON file (`data/follow_ups.json`) for demo
  purposes — in production this would push to a real calendar/SMS API, but
  the core claim (the model autonomously deciding *and acting* on follow-up
  timing) is fully demonstrated as-is.
- The translation confidence check is a small feature but the detail we'd
  point to first: it shows the system isn't blindly trusting a single LLM
  call in a safety-relevant context.
- This build uses the cloud Gemma API for speed of iteration during the
  sprint. `gemma_client.py` is structured so a local/offline endpoint (e.g.
  Ollama) could be swapped in with minimal changes if a fully offline
  version is built out later.

## Troubleshooting

- **`GEMMA_API_KEY is not set`**: make sure `.env` exists and has your key,
  and that you're running `uvicorn` from inside the `ilera/` directory.
- **502 error from `/api/triage`**: check the terminal output — the full
  Gemma API error message is included in the raised exception.
- **504 / timeout error**: almost always means `GEMMA_MODEL` in `.env`
  doesn't match a real model ID your key can access. Run
  `python list_models.py` to confirm the exact valid string. If the model
  name is confirmed correct and it still times out, it's likely a
  network/firewall issue blocking `generativelanguage.googleapis.com`.
- **400 error mentioning `thinkingConfig`**: this field isn't accepted by
  all Gemma 4 call shapes — it's safe to remove from `generationConfig`
  entirely, since thought-filtering already happens client-side in
  `_extract_answer_text()`.
- **Photo not being used in reasoning**: confirm your `GEMMA_MODEL`
  supports multimodal input and `generateContent` (check with
  `list_models.py`) — some smaller/older Gemma variants are text-only.

## License

Built for the Build with Gemma: AI for Africa Hackathon (Minna, 2026).
