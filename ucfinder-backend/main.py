import os
import json
import time
import difflib
import tempfile
import traceback
from pathlib import Path
from typing import List, Optional

from fastapi import FastAPI, Query, UploadFile, File
from pydantic import BaseModel
from google import genai as genai_client
from google.genai.errors import (
    APIError,
    ClientError,
    ServerError,
)

app = FastAPI()

BASE_DIR = Path(__file__).resolve().parent
kb_path = BASE_DIR / "kb.json"

if kb_path.exists():
    with open(kb_path, "r", encoding="utf-8") as f:
        KB = json.load(f)
else:
    KB = []

# Initialize client using environment variable safely
client = genai_client.Client(api_key=os.environ.get("GEMINI_API_KEY"))


PRIMARY_MODEL = "gemini-3.5-flash-lite"
FALLBACK_MODEL = "gemini-3.1-flash-lite"


# --- Models ---

class AskRequest(BaseModel):
    question: str


class ChatActionModel(BaseModel):
    type: str
    target: Optional[str] = None
    label: Optional[str] = None


class RoomInfo(BaseModel):
    room_code: str
    room_name: str
    building: str
    floor: str
    nav_target: str


class AskResponse(BaseModel):
    answer: str
    action: ChatActionModel
    found: bool
    room: Optional[RoomInfo] = None


class BuildingListResponse(BaseModel):
    buildings: List[str]


class FloorListResponse(BaseModel):
    floors: List[str]


class RoomListResponse(BaseModel):
    rooms: List[RoomInfo]


class TranscribeResponse(BaseModel):
    text: str


# --- Robust Gemini Call Handler ---

def generate_with_retry(contents, primary_model=PRIMARY_MODEL, fallback_model=FALLBACK_MODEL, max_retries=3):
    """
    Executes request using gemini-3.6-flash first. If 429 rate limit or 5xx server
    errors occur, retries with exponential backoff before falling back to gemini-2.0-flash.
    """
    models_to_try = [primary_model, fallback_model]

    for model in models_to_try:
        delay = 2  # Start backoff at 2 seconds
        for attempt in range(max_retries):
            try:
                response = client.models.generate_content(
                    model=model,
                    contents=contents
                )
                if response and response.text:
                    return response.text.strip()
                return ""

            except ClientError as e:
                err_code = getattr(e, "code", None)
                print(f"[ClientError on {model}] Code {err_code}: {e}")

                # If 404 Model Not Found, switch to fallback model immediately
                if err_code == 404:
                    print(f"[404 NotFound] Model '{model}' not found or deprecated. Switching to fallback...")
                    break

                # If 401/403 Auth errors, abort
                if err_code in (401, 403):
                    print(f"[{err_code} Auth Error] API Key missing or invalid.")
                    return None

                # If 429 Rate Limit Exceeded, wait & retry; if retries exhausted, move to fallback model
                if err_code == 429:
                    if attempt < max_retries - 1:
                        print(f"[429 Rate Limit] Retrying {model} in {delay}s... (Attempt {attempt + 1}/{max_retries})")
                        time.sleep(delay)
                        delay *= 2
                        continue
                    else:
                        print(f"[429 Rate Limit] Max retries exhausted for {model}. Switching to fallback model...")
                        break

                break

            except ServerError as e:
                err_code = getattr(e, "code", None)
                print(f"[ServerError on {model}] Code {err_code}: {e}")

                if attempt < max_retries - 1:
                    print(f"[Server Error] Retrying {model} in {delay}s... (Attempt {attempt + 1}/{max_retries})")
                    time.sleep(delay)
                    delay *= 2
                    continue
                break

            except APIError as e:
                print(f"[Generic APIError on {model}]: {e}")
                if attempt < max_retries - 1:
                    time.sleep(delay)
                    delay *= 2
                    continue
                break

            except Exception as e:
                print(f"[Unexpected Exception on {model}]: {e}")
                break

    return None


# --- Helper Functions ---

def to_room_info(entry: dict) -> RoomInfo:
    return RoomInfo(
        room_code=entry.get("room_code", entry.get("name", "")),
        room_name=entry.get("room_name", entry.get("name", "")),
        building=entry.get("building", ""),
        floor=str(entry.get("floor", "")),
        nav_target=entry.get("nav_target", "")
    )


def find_room_direct(query: str):
    q = query.strip().lower()
    for entry in KB:
        code = entry.get("room_code", entry.get("name", "")).lower()
        if code and (code == q or code in q):
            return entry
        for alias in entry.get("aliases", []):
            if alias.lower() in q:
                return entry
    return None


# --- Routes ---

@app.get("/ping")
def ping():
    return {"status": "ok"}


@app.get("/buildings", response_model=BuildingListResponse)
def get_buildings():
    buildings = sorted(set(r["building"] for r in KB if "building" in r))
    return BuildingListResponse(buildings=buildings)


@app.get("/floors", response_model=FloorListResponse)
def get_floors(building: Optional[str] = Query(None)):
    filtered = KB
    if building:
        filtered = [r for r in filtered if r.get("building", "").lower() == building.lower()]
    floors = sorted(set(str(r["floor"]) for r in filtered if "floor" in r))
    return FloorListResponse(floors=floors)


@app.get("/rooms", response_model=RoomListResponse)
def get_rooms(building: Optional[str] = Query(None), floor: Optional[str] = Query(None)):
    filtered = KB
    if building:
        filtered = [r for r in filtered if r.get("building", "").lower() == building.lower()]
    if floor:
        filtered = [r for r in filtered if str(r.get("floor", "")).lower() == floor.lower()]

    return RoomListResponse(rooms=[to_room_info(r) for r in filtered])


@app.post("/ask", response_model=AskResponse)
def ask(req: AskRequest):
    question = req.question.strip()

    room = find_room_direct(question)
    if room:
        info = to_room_info(room)
        return AskResponse(
            answer=f"Found Room {info.room_code} ({info.room_name}) - {info.floor}, {info.building}.",
            action=ChatActionModel(
                type="navigate",
                target=info.nav_target,
                label=f"Navigate to {info.room_code}"
            ),
            found=True,
            room=info
        )

    try:
        prompt = f"""You are WAV AI, the in-app assistant for UCFinder — a 3D campus
navigation app for the University of Cebu Lapu-Lapu and Mandaue (UCLM).

Answer the user's question in 1-2 short, friendly, plain English sentences.

Your scope: UCLM campus info (location, programs, admissions, history,
general facts), and how to use the UCFinder app (3D navigation, avatar
customization, search, this chat).

If the question is clearly unrelated to UCLM or UCFinder (homework help,
other schools, general world topics, coding help, etc.), politely decline
and redirect the user back to UCLM/UCFinder topics instead of answering it.

Do not include markdown formatting, headers, or bullet points — plain
conversational sentences only.

USER QUESTION: {question}"""

        answer_text = generate_with_retry(prompt)

        if not answer_text:
            answer_text = "Sorry, our AI system is currently busy or offline. Please try asking again in a few moments!"

        return AskResponse(
            answer=answer_text,
            action=ChatActionModel(type="none"),
            found=False,
            room=None
        )

    except Exception:
        print("[Gemini Exception]")
        traceback.print_exc()

        return AskResponse(
            answer="Sorry, I had trouble finding that information right now. Please try asking again!",
            action=ChatActionModel(type="none"),
            found=False,
            room=None
        )


@app.post("/transcribe", response_model=TranscribeResponse)
async def transcribe(audio: UploadFile = File(...)):
    temp_path = None
    audio_file = None
    try:
        audio_bytes = await audio.read()
        ext = Path(audio.filename).suffix if audio.filename else ".wav"

        with tempfile.NamedTemporaryFile(delete=False, suffix=ext) as temp_file:
            temp_file.write(audio_bytes)
            temp_path = Path(temp_file.name)

        audio_file = client.files.upload(file=str(temp_path))

        system_prompt = (
            "You are a fast speech-to-text engine for the UCLM UCFinder campus navigation app. "
            "Transcribe the audio accurately into plain English text. "
            "Common terms include room numbers (e.g., A35, CBE901, Room 101), building names (e.g., Annex 2, Main Building), "
            "and navigation phrases like 'Where is', 'How to go to', or 'Find'. "
            "Output ONLY the recognized phrase as plain text without extra commentary or punctuation."
        )

        transcribed_text = generate_with_retry([system_prompt, audio_file]) or ""
        return TranscribeResponse(text=transcribed_text)

    except Exception:
        print("[Gemini Transcribe Exception]")
        traceback.print_exc()
        return TranscribeResponse(text="")

    finally:
        if audio_file:
            try:
                client.files.delete(name=audio_file.name)
            except Exception:
                pass

        if temp_path and temp_path.exists():
            try:
                os.remove(temp_path)
            except Exception:
                pass