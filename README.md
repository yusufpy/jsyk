# Ilera — Field Triage Assistant

Multimodal + multilingual + agentic triage support for community health
workers, built on Gemma.

A health worker describes symptoms (and optionally attaches a photo).
Gemma returns a structured triage assessment — severity, likely conditions,
red flags, recommended action — which can be output in English, Hausa, or
Yoruba. The system then autonomously drafts a referral note and schedules
a follow-up based on urgency.

## Why this isn't "just a chat prompt"

- **Multimodal**: the model reasons jointly over a photo and free-text notes,
  not just captioning the image.
- **Local language, done carefully**: clinical reasoning stays in English
  internally (more reliable), and only the final patient-facing output is
  translated. A back-translation + confidence check catches cases where
  translation may have dropped or softened a critical detail — see
  `translate.py`.
- **Agentic**: the model's output triggers real downstream actions — a
  referral note is drafted and a follow-up is scheduled automatically,
  without further human input (see `agent.py`).

## Setup

```bash
cd ilera
python3 -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
pip install -r requirements.txt

cp .env.example .env
# edit .env and paste in your Gemma API key (from Google AI Studio)
```

### Find your exact model name first — don't guess it

Gemma 4 (Google's current generation as of mid-2026) doesn't use the old
`gemma-3-27b-it` naming pattern. Run this before anything else:

```bash
python list_models.py
```

This prints every model your API key can actually access, flagged with
whether it supports `generateContent` (required for this project). Copy the
exact string into `GEMMA_MODEL` in your `.env`.

## Run

```bash
uvicorn app:app --reload --port 8000
```

Open **http://localhost:8000** in your browser.

## Demo flow

1. Type a symptom description (or attach a photo of a skin condition, wound, etc).
2. Pick output language: English, Hausa, or Yoruba.
3. Click **Run triage**.
4. You'll see: severity, likely conditions, red flags, recommended action,
   the local-language translation with a confidence tag, an auto-generated
   referral note, and (if applicable) a scheduled follow-up date.

## Project structure

```
ilera/
├── app.py            # FastAPI routes
├── gemma_client.py    # Calls to Gemma via Google's Generative Language API
├── schema.py          # Structured output shapes (Pydantic)
├── translate.py        # Local-language layer + back-translation check
├── agent.py             # Referral note + follow-up scheduling
├── static/index.html   # Frontend
├── requirements.txt
└── .env.example
```

## Notes for judges / pitch

- We deliberately keep the "calendar" a local JSON file (`data/follow_ups.json`)
  for the demo — in a production build this would push to a real calendar
  API, but the point (the model autonomously deciding *and acting* on
  follow-up timing) is fully demonstrated as-is.
- The confidence check on translations is a small thing, but it's the detail
  that shows we're not blindly trusting the model's output in a
  safety-relevant context — call this out explicitly during judging.
- This build uses the cloud Gemma API for speed of iteration. The `gemma_client.py`
  module is written so a local Ollama endpoint could be swapped in with
  ~10 lines of change if an offline demo is wanted later.

## Troubleshooting

- **`GEMMA_API_KEY is not set`**: make sure `.env` exists and has your key,
  and that you're running `uvicorn` from inside the `ilera/` directory.
- **502 error from `/api/triage`**: check the terminal output — the full
  Gemma API error message is included in the exception and printed there.
- **504 / timeout error**: this almost always means `GEMMA_MODEL` in `.env`
  doesn't match a real model ID your key can access. Run
  `python list_models.py` to get the exact valid string. If the model name
  is confirmed correct and it still times out, it's likely a network/firewall
  issue on your machine blocking `generativelanguage.googleapis.com` — run
  `python list_models.py` alone first, since it isolates the network call
  from the rest of the app.
- **Photo not being used in reasoning**: confirm the model you set in
  `GEMMA_MODEL` supports multimodal input and `generateContent` (check with
  `list_models.py`) — some smaller/older Gemma variants are text-only.
