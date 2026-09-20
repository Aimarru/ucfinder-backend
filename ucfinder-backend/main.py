from fastapi import FastAPI, Query, UploadFile, File
from pydantic import BaseModel
from typing import List, Optional
from pathlib import Path
import json
import difflib
import os
import traceback
import google.generativeai as genai
import openai

app = FastAPI()

# Safe path resolution for kb.json relative to this file
BASE_DIR = Path(__file__).resolve().parent
kb_path = BASE_DIR / "kb.json"

if kb_path.exists():
    with open(kb_path, "r", encoding="utf-8") as f:
        KB = json.load(f)
else:
    KB = []

# Configure Gemini API
genai.configure(api_key=os.environ.get("GEMINI_API_KEY", ""))
model = genai.GenerativeModel("gemini-2.0-flash")

# Configure OpenAI (Whisper for STT)
openai_client = openai.OpenAI(api_key=os.environ.get("OPENAI_API_KEY", ""))


# --- Models ---
# NOTE: field names below MUST match Unity's ChatDataModels.cs exactly
# (room_code, room_name, building, floor, nav_target) — JsonUtility matches
# by field name, so any mismatch silently leaves those fields blank in Unity.

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


# --- Helper Functions ---

def to_room_info(entry: dict) -> RoomInfo:
    """Maps a kb.json entry to the RoomInfo shape Unity expects."""
    return RoomInfo(
        room_code=entry.get("room_code", entry.get("name", "")),
        room_name=entry.get("room_name", entry.get("name", "")),
        building=entry.get("building", ""),
        floor=str(entry.get("floor", "")),
        nav_target=entry.get("nav_target", "")
    )


def find_room_direct(query: str):
    """Exact/substring match on room_code or aliases — fast path, no AI needed."""
    q = query.strip().lower()
    for entry in KB:
        code = entry.get("room_code", entry.get("name", "")).lower()
        if code and (code == q or code in q):
            return entry
        for alias in entry.get("aliases", []):
            if alias.lower() in q:
                return entry
    return None


def retrieve(query: str, k: int = 4):
    """Fuzzy retrieval used to feed CONTEXT to Gemini for general questions."""
    q = query.lower()
    scored = []
    for e in KB:
        names = [e.get("room_code", e.get("name", ""))] + e.get("aliases", [])
        best = max((difflib.SequenceMatcher(None, q, n.lower()).ratio() for n in names if n), default=0)
        if any(n.lower() in q for n in names if n):
            best += 0.5
        scored.append((best, e))
    scored.sort(key=lambda x: -x[0])
    return [e for s, e in scored[:k] if s > 0.35]


SYSTEM = """You are WAV AI, the in-app assistant for UCFinder — a 3D campus
navigation app for the University of Cebu Lapu-Lapu and Mandaue (UCLM).

KNOWN FACTS ABOUT UCLM (use these when relevant, never invent beyond them):
- University of Cebu Lapu-Lapu and Mandaue (UCLM) is a private university,
  part of the University of Cebu system.
- Location: A.C. Cortes Avenue, Looc, Mandaue City, Cebu, Philippines.
- Offers programs including Information Technology, Engineering, Business,
  Education, Criminology, and Hospitality Management.
# TODO: add more verified facts here (founding year, more programs, contact info)

YOUR SCOPE — you can help with:
- General questions about UCLM itself (location, programs, what the school is)
- How to use the UCFinder app (navigation, avatar, search, this chat)
- Contacting UCFinder support
- General greetings/small talk related to helping the user

If asked about anything clearly outside both UCLM and the app (homework, other
schools, unrelated general knowledge, coding help, etc.), politely decline and
redirect to what you can help with.

RULES:
- For a SPECIFIC ROOM location, you will be given CONTEXT with room data if a
  match was found. If CONTEXT is empty and the user seems to want a room,
  tell them to type the exact room code or use the Find a Room button.
- Never invent a floor, building, room, contact, or schedule not in CONTEXT.
- Keep answers to 1-2 short sentences, friendly, plain English.
- Reply with ONLY a JSON object, no markdown fences, no explanation outside it:
  {"answer": "...", "action": {"type": "none"}}
"""


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

    # 1. Direct room lookup first — exact/fast, no AI cost, no hallucination risk
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

    # 2. Otherwise, fall back to Gemini with fuzzy-matched CONTEXT
    hits = retrieve(question)
    context = json.dumps(hits) if hits else "[]"
    prompt = f"{SYSTEM}\n\nCONTEXT:\n{context}\n\nUSER MESSAGE: {question}"

    try:
        response = model.generate_content(
            prompt,
            generation_config={"response_mime_type": "application/json"}
        )
        raw = response.text.strip()

        if raw.startswith("```"):
            raw = raw.strip("`").replace("json", "", 1).strip()

        parsed = json.loads(raw)

    except Exception:
        # Log the REAL error to Render's logs instead of silently swallowing it
        print("[Ask Exception]")
        traceback.print_exc()
        return AskResponse(
            answer="Sorry, I couldn't process that. Try entering a room code directly, or check the Help section.",
            action=ChatActionModel(type="none"),
            found=False,
            room=None
        )

    valid_targets = {e.get("nav_target") for e in hits if "nav_target" in e}
    act = parsed.get("action", {}) or {}
    if act.get("type") == "navigate" and act.get("target") not in valid_targets:
        act = {"type": "none"}

    return AskResponse(
        answer=parsed.get("answer", "Sorry, I didn't understand that."),
        action=ChatActionModel(**act) if act.get("type") else ChatActionModel(type="none"),
        found=False,
        room=None
    )


@app.post("/transcribe", response_model=TranscribeResponse)
async def transcribe(audio: UploadFile = File(...)):
    try:
        audio_bytes = await audio.read()
        temp_path = BASE_DIR / "temp_audio.wav"
        with open(temp_path, "wb") as f:
            f.write(audio_bytes)

        with open(temp_path, "rb") as f:
            result = openai_client.audio.transcriptions.create(
                model="whisper-1",
                file=f
            )

        return TranscribeResponse(text=result.text)

    except Exception:
        print("[Transcribe Exception]")
        traceback.print_exc()
        return TranscribeResponse(text="")