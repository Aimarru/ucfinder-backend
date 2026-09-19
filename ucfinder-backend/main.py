from fastapi import FastAPI, Query
from pydantic import BaseModel
from typing import List, Optional
from pathlib import Path
import json
import difflib
import os
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
model = genai.GenerativeModel("gemini-2.5-flash")

# --- Models ---
class AskRequest(BaseModel):
    question: str

class RoomInfo(BaseModel):
    name: str
    building: str
    floor: str
    nav_target: str

class BuildingListResponse(BaseModel):
    buildings: List[str]

class FloorListResponse(BaseModel):
    floors: List[str]

class RoomListResponse(BaseModel):
    rooms: List[RoomInfo]

# --- Helper Functions ---
def retrieve(query: str, k: int = 4):
    q = query.lower()
    scored = []
    for e in KB:
        names = [e["name"]] + e.get("aliases", [])
        best = max(difflib.SequenceMatcher(None, q, n.lower()).ratio() for n in names)
        if any(n.lower() in q for n in names):
            best += 0.5
        scored.append((best, e))
    scored.sort(key=lambda x: -x[0])
    return [e for s, e in scored[:k] if s > 0.35]

SYSTEM = """You are WAV AI, the in-app assistant for UCFinder — a 3D campus
navigation app for the University of Cebu Lapu-Lapu and Mandaue (UCLM).

STRICT SCOPE — you must ONLY discuss:
- Locations, rooms, offices, and facilities on the UCLM campus
- How to use the UCFinder app (navigation, avatar, search, this chat)
- Contacting UCFinder support or campus offices
- General greetings/small talk directly related to helping the user with the app

If asked about ANYTHING else — homework help, general knowledge, coding,
other schools, personal advice, current events, opinions, etc. — politely
decline and redirect: say you can only help with UCFinder and campus
navigation, and ask what they need help finding.

RULES FOR ANSWERS WITHIN SCOPE:
- Answer ONLY using the CONTEXT provided below. Never invent a floor,
  building, room, contact, or schedule not in CONTEXT.
- If CONTEXT doesn't contain the answer, say you don't have that yet
  and suggest checking the Help section. Set action type to "none".
- Keep answers to 1-2 short sentences, friendly, plain English.
- Reply with ONLY a JSON object, no markdown fences, no explanation outside it:
  {"answer": "...", "action": {"type": "navigate|none", "target": "...", "label": "..."}}
- "target" for navigate must be copied EXACTLY from a nav_target value in CONTEXT.
"""

# --- Routes ---

@app.get("/ping")
def ping():
    return {"status": "ok"}

@app.get("/buildings", response_model=BuildingListResponse)
def get_buildings():
    buildings = list(set([r["building"] for r in KB if "building" in r]))
    return BuildingListResponse(buildings=buildings)

@app.get("/floors", response_model=FloorListResponse)
def get_floors(building: Optional[str] = Query(None)):
    filtered = KB
    if building:
        filtered = [r for r in filtered if r.get("building", "").lower() == building.lower()]
    floors = list(set([r["floor"] for r in filtered if "floor" in r]))
    return FloorListResponse(floors=floors)

@app.get("/rooms", response_model=RoomListResponse)
def get_rooms(building: Optional[str] = Query(None), floor: Optional[str] = Query(None)):
    filtered = KB
    if building:
        filtered = [r for r in filtered if r.get("building", "").lower() == building.lower()]
    if floor:
        filtered = [r for r in filtered if str(r.get("floor", "")).lower() == floor.lower()]
    
    room_objects = [
        RoomInfo(
            name=r.get("name", ""),
            building=r.get("building", ""),
            floor=str(r.get("floor", "")),
            nav_target=r.get("nav_target", "")
        )
        for r in filtered
    ]
    return RoomListResponse(rooms=room_objects)

@app.post("/ask")
def ask(req: AskRequest):
    hits = retrieve(req.question)
    context = json.dumps(hits) if hits else "[]"
    prompt = f"{SYSTEM}\n\nCONTEXT:\n{context}\n\nUSER MESSAGE: {req.question}"

    try:
        response = model.generate_content(prompt)
        raw = response.text.strip()
        if raw.startswith("```"):
            raw = raw.strip("`").replace("json", "", 1).strip()
        parsed = json.loads(raw)
    except Exception:
        return {"answer": "Sorry, I couldn't process that. Try rephrasing.", "action": {"type": "none"}}

    valid = {e.get("nav_target") for e in hits if "nav_target" in e}
    act = parsed.get("action", {})
    if act.get("type") == "navigate" and act.get("target") not in valid:
        parsed["action"] = {"type": "none"}

    return parsed