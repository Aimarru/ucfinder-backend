from fastapi import FastAPI, Query, UploadFile, File
from pydantic import BaseModel
from typing import List, Optional
from pathlib import Path
import json
import difflib
import os
import traceback
import tempfile
import google.generativeai as genai

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

def retrieve(query: str, k: int = 4):
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

KNOWN FACTS ABOUT UCLM:
- University of Cebu Lapu-Lapu and Mandaue (UCLM) is a private university, part of the University of Cebu system.
- Location: A.C. Cortes Avenue, Looc, Mandaue City, Cebu, Philippines.
- Offers programs including Information Technology, Engineering, Business, Education, Criminology, and Hospitality Management.

YOUR SCOPE:
- General questions about UCLM campus locations, offices, and facilities
- App features (navigation, avatar customization, search)
- Contacting support or general campus greetings

RULES:
- Answer ONLY using CONTEXT when finding rooms.
- Keep answers to 1-2 short sentences, friendly, plain English.
- Reply with ONLY a JSON object: {"answer": "...", "action": {"type": "none"}}
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
        print("[Ask Exception]")
        traceback.print_exc()
        return AskResponse(
            answer="Sorry, I couldn't process that. Try entering a room code directly.",
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
    temp_path = None
    audio_file = None
    try:
        audio_bytes = await audio.read()
        ext = Path(audio.filename).suffix if audio.filename else ".wav"

        # 1. Create a safe temporary file on disk
        with tempfile.NamedTemporaryFile(delete=False, suffix=ext) as temp_file:
            temp_file.write(audio_bytes)
            temp_path = Path(temp_file.name)

        # 2. Upload file to Google Generative AI temporary storage
        audio_file = genai.upload_file(path=str(temp_path))

        # 3. Prompt Gemini 2.0 Flash to transcribe spoken audio directly
        response = model.generate_content([
            "Transcribe the spoken words in this audio exactly into plain text. Output ONLY the transcribed text, nothing else.",
            audio_file
        ])

        transcribed_text = response.text.strip() if response.text else ""
        return TranscribeResponse(text=transcribed_text)

    except Exception:
        print("[Gemini Transcribe Exception]")
        traceback.print_exc()
        return TranscribeResponse(text="")

    finally:
        # 4. Clean up Google Cloud temporary storage reference
        if audio_file:
            try:
                genai.delete_file(audio_file.name)
            except Exception:
                pass

        # 5. Clean up local Render server temporary file
        if temp_path and temp_path.exists():
            try:
                os.remove(temp_path)
            except Exception:
                pass