# main.py
from fastapi import FastAPI
from pydantic import BaseModel
import json, difflib, os
import google.generativeai as genai

app = FastAPI()
KB = json.load(open("kb.json"))

genai.configure(api_key=os.environ["GEMINI_API_KEY"])
model = genai.GenerativeModel("gemini-2.0-flash")  # cheap, fast, good enough for this

class AskRequest(BaseModel):
    question: str

def retrieve(query, k=4):
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
    except Exception as e:
        return {"answer": "Sorry, I couldn't process that. Try rephrasing.", "action": {"type": "none"}}

    valid = {e["nav_target"] for e in hits}
    act = parsed.get("action", {})
    if act.get("type") == "navigate" and act.get("target") not in valid:
        parsed["action"] = {"type": "none"}

    return parsed
