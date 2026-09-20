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
    """Maps a kb.json entry to the RoomInfo shape expected by Unity."""
    return RoomInfo(
        room_code=entry.get("room_code", entry.get("name", "")),
        room_name=entry.get("room_name", entry.get("name", "")),
        building=entry.get("building", ""),
        floor=str(entry.get("floor", "")),
        nav_target=entry.get("nav_target", "")
    )

def find_room_direct(query: str):
    """Exact/substring match on room_code or aliases — fast path, zero AI latency."""
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

SYSTEM = """You are WAV AI, the helpful in-app virtual assistant for UCFinder — a 3D campus navigation app for the University of Cebu Lapu-Lapu and Mandaue (UCLM).

YOUR SCOPE & ROLE:
1. Assist users with campus room searches and physical navigation at UCLM.
2. Answer general questions about UCLM (location, campus offices, programs offered, history, admissions, contact info, announcements). Use Google Search grounding when online verification is needed.
3. Assist with UCFinder app usage (avatar customization, search features, 3D path display, support).

OUT-OF-SCOPE REDIRECTION:
If asked about topics completely unrelated to UCLM or UCFinder (e.g. general homework coding, other schools, politics, general world news), politely decline and redirect the user back to UCLM campus navigation or app support.

STRICT JSON OUTPUT FORMAT REQUIREMENT:
You MUST respond with strictly a single raw JSON object. Do not add markdown backticks, markdown code blocks, or explanatory prose outside the JSON.
Follow this schema:
{
  "answer": "1-2 short, friendly, plain English sentences answering the query.",
  "action": {
    "type": "navigate" or "none",
    "target": "EXACT nav_target from CONTEXT if navigating, otherwise empty string",
    "label": "Button text or empty string"
  }
}"""

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

    # 1. Direct Room Lookup (Fast Path)
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

    # 2. General Queries with Gemini 2.0 Flash + Online Search Grounding
    hits = retrieve(question)
    context = json.dumps(hits) if hits else "[]"

    prompt = f"""{SYSTEM}

DATABASE ROOM CONTEXT:
{context}

USER QUESTION: {question}"""

    try:
        # Note: response_mime_type is intentionally omitted to avoid internal search tool execution conflicts
        response = model.generate_content(
            prompt,
            tools=[{"google_search": {}}]
        )

        raw = response.text.strip() if response and response.text else ""

        # Strip any markdown fences if present
        if "```" in raw:
            if "```json" in raw:
                raw = raw.split("```json")[-1].split("```")[0].strip()
            else:
                raw = raw.replace("```", "").strip()

        parsed = json.loads(raw)

        # Ensure action payload integrity
        act_data = parsed.get("action", {})
        if not isinstance(act_data, dict) or "type" not in act_data:
            act_data = {"type": "none"}

        valid_targets = {e.get("nav_target") for e in hits if "nav_target" in e}
        if act_data.get("type") == "navigate" and act_data.get("target") not in valid_targets:
            act_data = {"type": "none"}

        return AskResponse(
            answer=parsed.get("answer", "I am WAV AI, your UCLM campus guide. How can I assist you?"),
            action=ChatActionModel(**act_data),
            found=False,
            room=None
        )

    except Exception as e:
        print(f"[Gemini Ask Exception]: {e}")
        traceback.print_exc()

        # Context-aware fallback logic
        q_lower = question.lower()
        if any(k in q_lower for k in ["where", "location", "address", "located"]):
            ans = "University of Cebu Lapu-Lapu and Mandaue (UCLM) is located along A.C. Cortes Avenue, Looc, Mandaue City, Cebu, Philippines."
        elif any(k in q_lower for k in ["when", "create", "found", "start", "built", "established"]):
            ans = "University of Cebu Lapu-Lapu and Mandaue (UCLM) was established in May 1995."
        elif any(k in q_lower for k in ["course", "program", "major", "offer"]):
            ans = "UCLM offers programs in IT, Engineering, Business, Criminology, Education, Nursing, Customs Admin, and Maritime Studies."
        else:
            ans = "I couldn't look that up right now. Try entering a room code (e.g., 'A35') or asking about UCFinder features."

        return AskResponse(
            answer=ans,
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

        # Safe temporary file on disk
        with tempfile.NamedTemporaryFile(delete=False, suffix=ext) as temp_file:
            temp_file.write(audio_bytes)
            temp_path = Path(temp_file.name)

        # Upload audio directly to Gemini File API
        audio_file = genai.upload_file(path=str(temp_path))

        response = model.generate_content([
            "Transcribe the spoken words in this audio exactly into plain text. Output ONLY the transcribed text, nothing else.",
            audio_file
        ])

        transcribed_text = response.text.strip() if response and response.text else ""
        return TranscribeResponse(text=transcribed_text)

    except Exception:
        print("[Gemini Transcribe Exception]")
        traceback.print_exc()
        return TranscribeResponse(text="")

    finally:
        if audio_file:
            try:
                genai.delete_file(audio_file.name)
            except Exception:
                pass

        if temp_path and temp_path.exists():
            try:
                os.remove(temp_path)
            except Exception:
                pass