"""
Run this FIRST, before anything else, to see the exact model ID strings
your API key can access:

    python list_models.py

Gemma 4 model IDs don't follow the old "gemma-3-27b-it" pattern -- they're
named by capability tier (e.g. something like "gemma-4-27b-it" or
"gemma-4-12b-it" depending on how Google exposes it via this API). Copy the
exact string that supports generateContent into GEMMA_MODEL in your .env.
"""

import os
import requests
from dotenv import load_dotenv

load_dotenv()

API_KEY = os.environ.get("GEMMA_API_KEY", "")

if not API_KEY:
    print("GEMMA_API_KEY not set. Put it in .env first.")
    raise SystemExit(1)

resp = requests.get(
    f"https://generativelanguage.googleapis.com/v1beta/models?key={API_KEY}",
    timeout=30,
)

if resp.status_code != 200:
    print(f"Error {resp.status_code}: {resp.text}")
    raise SystemExit(1)

models = resp.json().get("models", [])
print(f"Found {len(models)} models available to this key:\n")

for m in models:
    name = m.get("name", "").replace("models/", "")
    methods = m.get("supportedGenerationMethods", [])
    if "gemma" in name.lower():
        marker = " <-- Gemma" + ("  [supports generateContent]" if "generateContent" in methods else "  [NO generateContent support]")
        print(f"  {name}{marker}")

print("\nFull list (including non-Gemma models) also returned; if nothing")
print("above says 'Gemma', your key/tier may not have Gemma 4 access yet --")
print("check https://ai.google.dev for current availability.")
