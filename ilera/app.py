"""
Ilera backend.

Run with:
    uvicorn app:app --reload --port 8000

Then open http://localhost:8000
"""

import os
import requests
from fastapi import FastAPI, UploadFile, Form, File, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from dotenv import load_dotenv

import gemma_client
import translate
import agent

load_dotenv()

app = FastAPI(title="Ilera Triage API")


@app.get("/")
def root():
    return FileResponse(os.path.join("static", "index.html"))


app.mount("/static", StaticFiles(directory="static"), name="static")


@app.post("/api/triage")
async def triage(
    symptom_text: str = Form(...),
    language: str = Form("en"),  # "en", "ha", "yo"
    patient_ref: str = Form("Patient"),
    image: UploadFile | None = File(None),
):
    image_bytes = None
    image_mime = "image/jpeg"
    if image is not None:
        image_bytes = await image.read()
        image_mime = image.content_type or "image/jpeg"

    try:
        result = gemma_client.run_triage(symptom_text, image_bytes, image_mime)
    except gemma_client.GemmaClientError as e:
        raise HTTPException(status_code=502, detail=str(e))
    except requests.exceptions.Timeout:
        raise HTTPException(
            status_code=504,
            detail=(
                f"Request to Gemma timed out after {gemma_client.REQUEST_TIMEOUT}s. "
                "This usually means the model name is wrong/inaccessible for your key "
                "(run list_models.py to check), or a network/firewall issue is blocking "
                "the connection to generativelanguage.googleapis.com."
            ),
        )
    except requests.exceptions.RequestException as e:
        raise HTTPException(status_code=502, detail=f"Network error calling Gemma: {e}")

    response = {"triage": result}

    if language in ("ha", "yo"):
        try:
            translated = translate.translate_triage_output(result, language)
            response["translation"] = translated
        except gemma_client.GemmaClientError as e:
            response["translation_error"] = str(e)

    note = agent.build_referral_note(result, patient_ref)
    event = agent.schedule_follow_up(note, patient_ref)
    response["referral"] = note
    response["follow_up_event"] = event

    return response


@app.get("/api/follow-ups")
def list_follow_ups():
    if not os.path.exists(agent.CALENDAR_PATH):
        return []
    import json

    with open(agent.CALENDAR_PATH, "r") as f:
        return json.load(f)
