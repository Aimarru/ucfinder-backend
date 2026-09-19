import os
import re
from typing import Optional, List
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from dotenv import load_dotenv
from google import genai

# Load environment variables from .env file
load_dotenv()

# Read the key from the environment variable named GEMINI_API_KEY
API_KEY = os.getenv("GEMINI_API_KEY")
client = genai.Client(api_key=API_KEY) if API_KEY else None

app = FastAPI(title="UC Finder Backend")

# Enable CORS for Unity and Web clients
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ------------------------------------------------------------------
# Data Models matching C# Unity Client
# ------------------------------------------------------------------

class AskRequestBody(BaseModel):
    question: str

class ChatAction(BaseModel):
    type: str  # "navigate" | "none"
    target: str
    label: str

class RoomInfo(BaseModel):
    room_code: str
    room_name: str
    building: str
    floor: str
    nav_target: str

class AskResponse(BaseModel):
    answer: str
    action: ChatAction
    found: bool
    room: Optional[RoomInfo] = None

class BuildingListResponse(BaseModel):
    buildings: List[str]

class FloorListResponse(BaseModel):
    floors: List[str]

class RoomListResponse(BaseModel):
    rooms: List[RoomInfo]

# ------------------------------------------------------------------
# Hardcoded Mock Room Data
# ------------------------------------------------------------------

ROOMS = [
    {
        "room_code": "A35",
        "room_name": "Accounting Office",
        "building": "Main Building",
        "floor": "3rd Floor",
        "nav_target": "node_main_fl3_a35"
    },
    {
        "room_code": "CS101",
        "room_name": "Computer Science Lab 1",
        "building": "Science Wing",
        "floor": "1st Floor",
        "nav_target": "node_sci_fl1_cs101"
    },
    {
        "room_code": "LIB1",
        "room_name": "Main Library",
        "building": "Student Center",
        "floor": "2nd Floor",
        "nav_target": "node_sc_fl2_lib1"
    }
]

SYSTEM_PROMPT = """
You are WAV AI, a helpful indoor navigation assistant for a university campus app called UC Finder.
Your job is ONLY to answer questions about campus locations, how to navigate buildings, or how to use features in the UC Finder app (like avatars, map navigation, floor selectors).

STRICT SCOPE RULE:
- If the user asks for homework help, math equations (e.g., "solve x^2+2x=0"), general coding, or non-campus topics, politely decline and remind them you are only a campus wayfinding assistant.
- Keep answers short, friendly, and easy to read in a mobile chat bubble (under 3 sentences).
"""

# ------------------------------------------------------------------
# Endpoints
# ------------------------------------------------------------------

@app.get("/")
def read_root():
    return {"status": "UC Finder Backend is running"}

@app.post("/ask", response_model=AskResponse)
def ask_question(body: AskRequestBody):
    user_query = body.question.strip()
    
    # 1. Direct Room Lookup matching (Case-insensitive)
    for room in ROOMS:
        if room["room_code"].lower() in user_query.lower() or room["room_name"].lower() in user_query.lower():
            return AskResponse(
                answer=f"Found Room {room['room_code']} ({room['room_name']}) - {room['floor']}, {room['building']}.",
                action=ChatAction(
                    type="navigate",
                    target=room["nav_target"],
                    label=f"Navigate to {room['room_code']}"
                ),
                found=True,
                room=RoomInfo(**room)
            )

    # 2. Scope Guard for Math/Equations
    if re.search(r'[\d+\-*/^=]', user_query) and any(kw in user_query.lower() for kw in ["solve", "calculate", "math", "="]):
        return AskResponse(
            answer="I can only assist with campus navigation and UC Finder app features.",
            action=ChatAction(type="none", target="", label=""),
            found=False,
            room=None
        )

    # 3. Call Gemini using modern SDK
    try:
        if not client:
            raise ValueError("GEMINI_API_KEY environment variable is missing or invalid.")

        response = client.models.generate_content(
        model='gemini-3.6-flash',
        contents=f"{SYSTEM_PROMPT}\n\nUser Question: {user_query}",
        )
        
        return AskResponse(
            answer=response.text.strip(),
            action=ChatAction(type="none", target="", label=""),
            found=False,
            room=None
        )

    except Exception as e:
        print(f"Gemini API Error: {e}")
        return AskResponse(
            answer="Sorry, I couldn't process that right now. Try entering a room code directly.",
            action=ChatAction(type="none", target="", label=""),
            found=False,
            room=None
        )

@app.get("/buildings", response_model=BuildingListResponse)
def get_buildings():
    buildings = list(set([r["building"] for r in ROOMS]))
    return BuildingListResponse(buildings=buildings)

@app.get("/floors", response_model=FloorListResponse)
def get_floors():
    floors = list(set([r["floor"] for r in ROOMS]))
    return FloorListResponse(floors=floors)

@app.get("/rooms", response_model=RoomListResponse)
def get_rooms():
    room_objects = [RoomInfo(**r) for r in ROOMS]
    return RoomListResponse(rooms=room_objects)