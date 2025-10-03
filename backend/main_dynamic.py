# backend_refactored.py
# Alive5 Voice Agent Backend — Orchestrator First with Greeting Trigger + Frontend sync endpoints

import os, json, time, uuid, logging
from pathlib import Path
from typing import Dict, Any, Optional, List

import httpx, uvicorn
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from livekit import api

from backend.conversational_orchestrator import (
    ConversationalOrchestrator,
    OrchestratorAction,
    OrchestratorDecision,
    create_orchestrator_from_template,
)

# --------------------------------------------------------------------
# Setup
# --------------------------------------------------------------------
current_dir = Path(__file__).parent
load_dotenv(current_dir / "../../.env")

app = FastAPI(title="Alive5 Voice Agent — Orchestrator", version="3.0")

logging.basicConfig(level=logging.INFO, format="%(message)s", force=True)
logger = logging.getLogger("orchestrator-backend")


app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "https://voice-agent-livekit.vercel.app",  # your frontend
        "http://localhost:3000",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

LIVEKIT_API_KEY = os.getenv("LIVEKIT_API_KEY")
LIVEKIT_API_SECRET = os.getenv("LIVEKIT_API_SECRET")
LIVEKIT_URL = os.getenv("LIVEKIT_URL")

A5_BASE_URL = os.getenv("A5_BASE_URL")
A5_API_KEY = os.getenv("A5_API_KEY")
A5_TEMPLATE_URL = os.getenv("A5_TEMPLATE_URL", "/1.0/org-botchain/generate-template")
A5_FAQ_URL = os.getenv("A5_FAQ_URL", "/public/1.0/get-faq-bot-response-by-bot-id")
FAQ_BOT_ID = os.getenv("FAQ_BOT_ID", "default-bot-id")

# --------------------------------------------------------------------
# State
# --------------------------------------------------------------------
PERSISTENCE_DIR = current_dir / "persistence"
FLOW_STATES_DIR = PERSISTENCE_DIR / "flow_states"
for d in [PERSISTENCE_DIR, FLOW_STATES_DIR]:
    d.mkdir(exist_ok=True)

class FlowState(BaseModel):
    current_flow: Optional[str] = None
    current_step: Optional[str] = None
    flow_data: Optional[Dict[str, Any]] = None
    conversation_history: List[Dict[str, str]] = Field(default_factory=list)

flow_states: Dict[str, FlowState] = {}
bot_template: Optional[Dict[str, Any]] = None
conversational_orchestrator: Optional[ConversationalOrchestrator] = None

def save_flow_state(room: str, state: FlowState):
    with open(FLOW_STATES_DIR / f"{room}.json", "w") as f:
        json.dump(state.dict(), f, indent=2)

def load_flow_state(room: str) -> Optional[FlowState]:
    path = FLOW_STATES_DIR / f"{room}.json"
    return FlowState(**json.load(open(path))) if path.exists() else None

# --------------------------------------------------------------------
# Models
# --------------------------------------------------------------------
class ProcessFlowMessageRequest(BaseModel):
    room_name: str
    user_message: str
    conversation_history: List[Dict[str, str]] = Field(default_factory=list)

# --------------------------------------------------------------------
# Core Flow Processing
# --------------------------------------------------------------------
@app.post("/api/process_flow_message")
async def process_flow_message(req: ProcessFlowMessageRequest):
    global conversational_orchestrator, bot_template
    if conversational_orchestrator is None:
        return {"status": "error", "message": "No orchestrator loaded"}

    room, msg = req.room_name, (req.user_message or "").strip()

    # Greeting trigger
    if msg == "__start__":
        if bot_template and "data" in bot_template:
            for flow_key, flow_data in bot_template["data"].items():
                if flow_data.get("type") == "greeting":
                    greeting = flow_data.get("text", "Hello! How can I help?")
                    state = flow_states.get(room) or FlowState()
                    state.current_flow, state.current_step, state.flow_data = flow_key, flow_data.get("name"), flow_data
                    flow_states[room] = state; save_flow_state(room, state)
                    return {"status": "processed", "flow_result": {
                        "type": "flow_started","flow_name": "greeting",
                        "response": greeting,"flow_state": state.dict()}}
        return {"status": "processed","flow_result": {
            "type":"message","response":"Hello! Thanks for calling Alive5. How can I help today?","flow_state":{}}}

    state = flow_states.get(room) or load_flow_state(room) or FlowState()
    state.conversation_history.append({"role": "user", "content": msg})
    state.conversation_history = state.conversation_history[-10:]

    try:
        decision: OrchestratorDecision = await conversational_orchestrator.process_message(
            user_message=msg, room_name=room,
            conversation_history=state.conversation_history,
            current_flow_state=state.dict(), current_step_data=state.flow_data)
    except Exception as e:
        logger.error(f"Orchestrator error: {e}", exc_info=True)
        return {"status":"error","flow_result":{"type":"error","response":"System error"}}

    response_text = decision.response or ""
    if decision.action == OrchestratorAction.EXECUTE_FLOW and decision.flow_to_execute:
        state.current_flow = decision.flow_to_execute
        state.current_step = decision.flow_to_execute
        state.flow_data = conversational_orchestrator.available_flows.get(decision.flow_to_execute)
    elif decision.action == OrchestratorAction.USE_FAQ:
        faq = await get_faq_response(msg); response_text = faq["response"]
    elif decision.action == OrchestratorAction.HANDLE_REFUSAL:
        response_text = response_text or "Got it, we’ll skip that."
    elif decision.action == OrchestratorAction.HANDLE_UNCERTAINTY:
        response_text = response_text or "No worries, let’s continue."
    elif decision.action == OrchestratorAction.SPEAK_WITH_PERSON:
        response_text = response_text or "Connecting you to an agent now."
        return {"status":"processed","flow_result":{"type":"agent_handoff","response":response_text}}
    else:
        if not response_text:
            faq = await get_faq_response(msg); response_text = faq["response"]

    flow_states[room] = state; save_flow_state(room, state)
    return {"status":"processed","flow_result":{
        "type":decision.action.value,"response":response_text,"flow_state":state.dict()}}

# --------------------------------------------------------------------
# FAQ Wrapper
# --------------------------------------------------------------------
async def get_faq_response(user_message: str) -> Dict[str, Any]:
    async with httpx.AsyncClient(timeout=10.0) as client:
        resp = await client.post(
            f"{A5_BASE_URL}{A5_FAQ_URL}",
            headers={"X-A5-APIKEY": A5_API_KEY,"Content-Type":"application/json"},
            json={"bot_id":FAQ_BOT_ID,"faq_question":user_message})
        if resp.status_code==200:
            data=resp.json()
            return {"response":data.get("data",{}).get("answer","I'm not sure.")}
        return {"response":"Sorry, I couldn’t fetch that."}

# --------------------------------------------------------------------
# Template Refresh
# --------------------------------------------------------------------
@app.post("/api/refresh_template")
@app.post("/api/validate_and_load_template")   # alias for frontend
async def refresh_template():
    global bot_template, conversational_orchestrator
    async with httpx.AsyncClient(timeout=15.0) as client:
        resp = await client.post(
            f"{A5_BASE_URL}{A5_TEMPLATE_URL}",
            headers={"X-A5-APIKEY": A5_API_KEY,"Content-Type":"application/json"},
            json={"botchain_name":os.getenv("A5_BOTCHAIN_NAME"),"org_name":os.getenv("A5_ORG_NAME")})
        resp.raise_for_status()
        bot_template=resp.json()
        conversational_orchestrator=create_orchestrator_from_template(bot_template)
        return {"status":"success","flows":list(conversational_orchestrator.available_flows.keys())}

# --------------------------------------------------------------------
# Frontend-Sync Endpoints
# --------------------------------------------------------------------
class ConnectionDetailsRequest(BaseModel):
    participant_name: str
    user_data: dict

@app.post("/api/connection_details")
async def get_connection_details(req: ConnectionDetailsRequest):
    try:
        room_name=f"room_{req.participant_name}_{os.urandom(4).hex()}"
        grant=api.VideoGrant(room=room_name,room_join=True)
        token=api.AccessToken(LIVEKIT_API_KEY,LIVEKIT_API_SECRET)
        token.identity=req.participant_name; token.metadata=str(req.user_data); token.add_grant(grant)
        participant_token=token.to_jwt()
        return {"serverUrl":LIVEKIT_URL,"participantToken":participant_token,
                "roomName":room_name,"selectedVoice":req.user_data.get("selected_voice")}
    except Exception as e:
        logger.error(f"❌ Connection details error: {e}")
        raise HTTPException(status_code=500,detail="Could not issue connection details")

class SessionUpdateRequest(BaseModel):
    room_name: str; intent: str; user_data: dict; status: str

@app.post("/api/sessions/update")
async def update_session(req: SessionUpdateRequest):
    logger.info(f"🔄 Session update {req.room_name}: intent={req.intent}, status={req.status}")
    return {"status":"success"}

@app.delete("/api/rooms/{room_name}")
async def delete_room(room_name: str):
    logger.info(f"🧹 Room cleanup: {room_name}")
    return {"status":"success"}

@app.get("/api/available_voices")
async def available_voices():
    """
    Load available voices from cached_voices.json if present,
    otherwise fall back to a minimal built-in list.
    """
    voices_path = current_dir / "cached_voices.json"
    if voices_path.exists():
        try:
            with open(voices_path, "r") as f:
                voices = json.load(f)
            return {"status": "success", "voices": voices}
        except Exception as e:
            logger.error(f"❌ Failed to load cached_voices.json: {e}")

    # fallback minimal voices
    fallback = {
        "f114a467-c40a-4db8-964d-aaba89cd08fa": "Miles - Yogi (Default)",
        "7f423809-0011-4658-ba48-a411f5e516ba": "Ashwin - Warm Narrator",
        "a167e0f3-df7e-4d52-a9c3-f949145efdab": "Blake - Helpful Agent",
        "e07c00bc-4134-4eae-9ea4-1a55fb45746b": "Brooke - Big Sister",
    }
    return {"status": "success", "voices": fallback}

class VoiceChangeRequest(BaseModel):
    room_name: str; voice_id: str

@app.post("/api/change_voice")
async def change_voice(req: VoiceChangeRequest):
    logger.info(f"🎤 Voice change request: room={req.room_name}, voice={req.voice_id}")
    return {"status":"success","voice_name":req.voice_id}

# --------------------------------------------------------------------
# Health
# --------------------------------------------------------------------
@app.get("/health")
async def health_check():
    return {"status":"ok","active_sessions":len(flow_states),"timestamp":time.time()}

# --------------------------------------------------------------------
# Run
# --------------------------------------------------------------------
if __name__=="__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
