from fastapi import FastAPI, Query, UploadFile, File
from pydantic import BaseModel
from typing import List, Optional
from pathlib import Path
import json
import difflib
import os
import traceback
import tempfile
from google import genai as genai_client

app = FastAPI()

BASE_DIR = Path(__file__).resolve().parent
kb_path = BASE_DIR / "kb.json"

if kb_path.exists():
    with open(kb_path, "r", encoding="utf-8") as f:
        KB = json.load(f)
else:
    KB = []

client = genai_client.Client(api_key=os.environ.get("GEMINI_API_KEY", ""))
GEMINI_MODEL = "gemini-3.6-flash"


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

def retrieve(query: str, k: int = 5):
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

        response = client.models.generate_content(
            model=GEMINI_MODEL,
            contents=prompt
        )

        answer_text = response.text.strip() if response and response.text else \
            "Sorry, I don't have an answer for that right now."

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

        response = client.models.generate_content(
            model=GEMINI_MODEL,
            contents=[
                "Transcribe the spoken words in this audio exactly into plain text. Output ONLY the transcribed text, nothing else.",
                audio_file
            ]
        )

        transcribed_text = response.text.strip() if response and response.text else ""
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