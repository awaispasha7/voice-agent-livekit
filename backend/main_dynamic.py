# backend_refactored.py
# Alive5 Voice Agent Backend — Orchestrator First with Greeting Trigger + Frontend sync endpoints

import os, json, time, uuid, logging
from pathlib import Path
from typing import Dict, Any, Optional, List
from datetime import timedelta

import httpx, uvicorn
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from livekit import api, rtc
from livekit.api import room_service
from livekit.rtc import DataPacketKind

from backend.conversational_orchestrator import (
    ConversationalOrchestrator,
    OrchestratorAction,
    OrchestratorDecision,
    create_orchestrator_from_template,
)

# --------------------------------------------------------------------
# Setup & Env
# --------------------------------------------------------------------
current_dir = Path(__file__).parent
env_paths = [
    current_dir / "../../.env",                     # relative to backend file
    current_dir / "../../../.env",                  # project root fallback
    Path("/home/ubuntu/alive5-voice-agent/.env"),   # production path
    Path(".env"),                                   # CWD
]
env_loaded = False
for env_path in env_paths:
    if env_path.exists():
        load_dotenv(dotenv_path=str(env_path), override=True)
        print(f"✅ Loaded .env from: {env_path}")
        env_loaded = True
        break
if not env_loaded:
    load_dotenv()  # fallback to process env

app = FastAPI(title="Alive5 Voice Agent — Orchestrator", version="3.0")

logging.basicConfig(level=logging.INFO, format="%(message)s", force=True)
logger = logging.getLogger("orchestrator-backend")

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "https://voice-agent-livekit.vercel.app",  # frontend
        "https://18.210.238.67.nip.io",            # backend domain
        "http://localhost:3000",                   # local dev (optional)
        "http://localhost:5173",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# LiveKit
LIVEKIT_API_KEY = os.getenv("LIVEKIT_API_KEY")
LIVEKIT_API_SECRET = os.getenv("LIVEKIT_API_SECRET")
LIVEKIT_URL = os.getenv("LIVEKIT_URL")

# Alive5 / Template API
A5_BASE_URL   = os.getenv("A5_BASE_URL")
A5_API_KEY    = os.getenv("A5_API_KEY")
A5_TEMPLATE_URL = os.getenv("A5_TEMPLATE_URL", "/1.0/org-botchain/generate-template")
A5_FAQ_URL    = os.getenv("A5_FAQ_URL", "/public/1.0/get-faq-bot-response-by-bot-id")
FAQ_BOT_ID    = os.getenv("FAQ_BOT_ID", "default-bot-id")

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

# Live session tracking (frontend + worker use this)
DEFAULT_VOICE_ID = "f114a467-c40a-4db8-964d-aaba89cd08fa" # Miles - Yogi
active_sessions: Dict[str, Dict[str, Any]] = {}

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

class ConnectionDetailsRequest(BaseModel):
    participant_name: str
    user_data: Dict[str, Any] = {}

class SessionUpdateRequest(BaseModel):
    room_name: str
    intent: Optional[str] = None
    status: Optional[str] = None
    user_data: Optional[Dict[str, Any]] = None

class VoiceChangeRequest(BaseModel):
    room_name: str
    voice_id: str

# --------------------------------------------------------------------
# Core Flow Processing
# --------------------------------------------------------------------
@app.post("/api/process_flow_message")
async def process_flow_message(req: ProcessFlowMessageRequest):
    global conversational_orchestrator, bot_template
    if conversational_orchestrator is None:
        return {"status": "error", "message": "No orchestrator loaded"}

    room, msg = req.room_name, (req.user_message or "").strip()

    # Greeting trigger from worker
    if msg == "__start__":
        if bot_template and "data" in bot_template:
            for flow_key, flow_data in bot_template["data"].items():
                if flow_data.get("type") == "greeting":
                    greeting = flow_data.get("text", "Hello! How can I help?")
                    state = flow_states.get(room) or FlowState()
                    state.current_flow = flow_key
                    state.current_step = flow_data.get("name")
                    state.flow_data   = flow_data
                    flow_states[room] = state
                    save_flow_state(room, state)
                    return {
                        "status": "processed",
                        "flow_result": {
                            "type": "flow_started",
                            "flow_name": "greeting",
                            "response": greeting,
                            "flow_state": state.dict()
                        }
                    }
        return {
            "status": "processed",
            "flow_result": {
                "type": "message",
                "response": "Hello! Thanks for calling Alive5. How can I help today?",
                "flow_state": {}
            }
        }

    state = flow_states.get(room) or load_flow_state(room) or FlowState()
    state.conversation_history.append({"role": "user", "content": msg})
    state.conversation_history = state.conversation_history[-10:]

    try:
        decision: OrchestratorDecision = await conversational_orchestrator.process_message(
            user_message=msg,
            room_name=room,
            conversation_history=state.conversation_history,
            current_flow_state=state.dict(),
            current_step_data=state.flow_data
        )
    except Exception as e:
        logger.error(f"Orchestrator error: {e}", exc_info=True)
        return {"status": "error", "flow_result": {"type": "error", "response": "System error"}}

    response_text = decision.response or ""
    if decision.action == OrchestratorAction.EXECUTE_FLOW and decision.flow_to_execute:
        state.current_flow = decision.flow_to_execute
        state.current_step = decision.flow_to_execute
        state.flow_data = conversational_orchestrator.available_flows.get(decision.flow_to_execute)
    elif decision.action == OrchestratorAction.USE_FAQ:
        faq = await get_faq_response(msg)
        response_text = faq["response"]
    elif decision.action == OrchestratorAction.HANDLE_REFUSAL:
        response_text = response_text or "Got it, we’ll skip that."
    elif decision.action == OrchestratorAction.HANDLE_UNCERTAINTY:
        response_text = response_text or "No worries, let’s continue."
    elif decision.action == OrchestratorAction.SPEAK_WITH_PERSON:
        response_text = response_text or "Connecting you to an agent now."
        return {"status": "processed", "flow_result": {"type": "agent_handoff", "response": response_text}}
    else:
        if not response_text:
            faq = await get_faq_response(msg)
            response_text = faq["response"]

    flow_states[room] = state
    save_flow_state(room, state)
    return {
        "status": "processed",
        "flow_result": {
            "type": decision.action.value,
            "response": response_text,
            "flow_state": state.dict()
        }
    }

# --------------------------------------------------------------------
# FAQ Wrapper
# --------------------------------------------------------------------
async def get_faq_response(user_message: str) -> Dict[str, Any]:
    if not (A5_BASE_URL and A5_FAQ_URL and A5_API_KEY):
        logger.error("❌ FAQ call missing env: A5_BASE_URL/A5_FAQ_URL/A5_API_KEY")
        return {"response": "Sorry, FAQ service is not configured."}
    async with httpx.AsyncClient(timeout=10.0) as client:
        resp = await client.post(
            f"{A5_BASE_URL}{A5_FAQ_URL}",
            headers={"X-A5-APIKEY": A5_API_KEY, "Content-Type": "application/json"},
            json={"bot_id": FAQ_BOT_ID, "faq_question": user_message},
        )
        if resp.status_code == 200:
            data = resp.json()
            return {"response": data.get("data", {}).get("answer", "I'm not sure.")}
        return {"response": "Sorry, I couldn’t fetch that."}

# --------------------------------------------------------------------
# Template Refresh (accepts frontend botchain/org)
# --------------------------------------------------------------------
@app.post("/api/refresh_template")
async def refresh_template(req: Request):
    """
    Loads/refreshes Alive5 bot template. Expects JSON optionally:
    {
      "botchain_name": "<required in FE>",
      "org_name": "<optional, defaults to 'alive5stage0'>"
    }
    Falls back to env A5_BOTCHAIN_NAME, A5_ORG_NAME when missing.
    """
    global bot_template, conversational_orchestrator

    payload = {}
    try:
        payload = await req.json()
    except Exception:
        payload = {}

    botchain_name = (payload.get("botchain_name") if isinstance(payload, dict) else None) or os.getenv("A5_BOTCHAIN_NAME")
    org_name = (payload.get("org_name") if isinstance(payload, dict) else None) or os.getenv("A5_ORG_NAME") or "alive5stage0"

    if not (A5_BASE_URL and A5_TEMPLATE_URL):
        raise HTTPException(status_code=500, detail="Template API not configured (A5_BASE_URL/A5_TEMPLATE_URL)")

    if not A5_API_KEY:
        raise HTTPException(status_code=500, detail="Missing A5_API_KEY in backend config")

    if not botchain_name:
        raise HTTPException(status_code=400, detail="botchain_name is required (pass from frontend)")

    async with httpx.AsyncClient(timeout=15.0) as client:
        resp = await client.post(
            f"{A5_BASE_URL}{A5_TEMPLATE_URL}",
            headers={"X-A5-APIKEY": A5_API_KEY, "Content-Type": "application/json"},
            json={"botchain_name": botchain_name, "org_name": org_name},
        )
        if resp.status_code != 200:
            logger.error(f"Template API error: {resp.status_code} - {resp.text}")
            raise HTTPException(status_code=resp.status_code, detail="Template refresh failed")
        bot_template = resp.json()
        conversational_orchestrator = create_orchestrator_from_template(bot_template)
        return {
            "status": "success",
            "flows": list(conversational_orchestrator.available_flows.keys())
        }

# --------------------------------------------------------------------
# Connection Details (LiveKit token issuance) — GET + POST
# --------------------------------------------------------------------
def _generate_unique_room_name(participant_name: str, intent: Optional[str] = None) -> str:
    base = intent or "room"
    return f"{base}_{participant_name}_{str(uuid.uuid4())[:8]}"

@app.get("/api/connection_details")
def get_connection_details():
    """Legacy GET endpoint - generates random user and unique room"""
    if not all([LIVEKIT_API_KEY, LIVEKIT_API_SECRET, LIVEKIT_URL]):
        raise HTTPException(
            status_code=500,
            detail="Missing LiveKit credentials"
        )
    
    try:
        # Generate participant details with truly unique room
        participant_name = f"user_{str(uuid.uuid4())[:8]}"
        room_name = _generate_unique_room_name(participant_name)
        
        logger.info(f"Generating token for {participant_name} in room {room_name}")
        
        # Create token with appropriate permissions (NOTE: VideoGrants plural)
        token = api.AccessToken(LIVEKIT_API_KEY, LIVEKIT_API_SECRET)
        token.with_identity(participant_name)
        token.with_grants(api.VideoGrants(
            room_join=True,
            room=room_name,
            can_publish=True,
            can_subscribe=True,
            can_publish_data=True,
        ))
        # Extended TTL for longer conversations
        token.with_ttl(timedelta(minutes=45))
        
        jwt_token = token.to_jwt()
        
        # Track session
        session_data = {
            "participant_name": participant_name,
            "room_name": room_name,
            "created_at": time.time(),
            "intent": None,
            "status": "created",
            "user_data": {},
            "voice_id": DEFAULT_VOICE_ID,
            "selected_voice": DEFAULT_VOICE_ID,
        }
        active_sessions[room_name] = session_data
        
        return {
            "serverUrl": LIVEKIT_URL,
            "roomName": room_name,
            "participantToken": jwt_token,
            "participantName": participant_name,
            "sessionId": room_name,
            "features": ["dynamic_intent", "session_tracking"]
        }
        
    except Exception as e:
        logger.error(f"Error generating connection details: {e}")
        raise HTTPException(status_code=500, detail=str(e))

class ConnectionRequest(BaseModel):
    participant_name: str
    intent: Optional[str] = None
    user_data: Optional[Dict[str, Any]] = None

@app.post("/api/connection_details")
def create_connection_with_custom_room(request: ConnectionRequest):
    """Enhanced POST endpoint with intent support"""
    if not all([LIVEKIT_API_KEY, LIVEKIT_API_SECRET, LIVEKIT_URL]):
        raise HTTPException(
            status_code=500,
            detail="Missing LiveKit credentials"
        )
    
    try:
        participant_name = request.participant_name
        intent = request.intent
        user_data = request.user_data or {}
        
        # Generate room name with intent context
        room_name = _generate_unique_room_name(participant_name, intent)
        
        logger.info(f"Generating token for {participant_name} in room {room_name} with intent: {intent}")
        
        # Create token with extended permissions (NOTE: VideoGrants plural)
        token = api.AccessToken(LIVEKIT_API_KEY, LIVEKIT_API_SECRET)
        token.with_identity(participant_name)
        token.with_grants(api.VideoGrants(
            room_join=True,
            room=room_name,
            can_publish=True,
            can_subscribe=True,
            can_publish_data=True,  # Allow data publishing for intent updates
        ))
        token.with_ttl(timedelta(minutes=45))
        
        jwt_token = token.to_jwt()
        
        # Track session with intent and user data
        session_data = {
            "participant_name": participant_name,
            "room_name": room_name,
            "created_at": time.time(),
            "intent": intent,
            "status": "created",
            "selected_voice": user_data.get("selected_voice", DEFAULT_VOICE_ID),
            "voice_id": user_data.get("selected_voice", DEFAULT_VOICE_ID),
            "user_data": user_data
        }
        active_sessions[room_name] = session_data
        
        return {
            "serverUrl": LIVEKIT_URL,
            "roomName": room_name,
            "participantToken": jwt_token,
            "participantName": participant_name,
            "sessionId": room_name,
            "initialIntent": intent,
            "selectedVoice": session_data["selected_voice"],
            "features": ["dynamic_intent", "session_tracking", "user_data_collection"]
        }
        
    except Exception as e:
        logger.error(f"Error creating enhanced connection: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# --------------------------------------------------------------------
# Sessions: update / info / list / cleanup
# --------------------------------------------------------------------
@app.post("/api/sessions/update")
def update_session(req: SessionUpdateRequest):
    room_name = req.room_name
    if room_name not in active_sessions:
        logger.warning(f"Session {room_name} not found, creating new session")
        active_sessions[room_name] = {
            "room_name": room_name,
            "participant_name": "Unknown",
            "created_at": time.time(),
            "last_updated": time.time(),
            "user_data": {},
            "status": "active",
            "intent": None,
            "voice_id": DEFAULT_VOICE_ID,
            "selected_voice": DEFAULT_VOICE_ID,
        }

    session = active_sessions[room_name]

    if req.intent:
        session["intent"] = req.intent
        session["intent_detected_at"] = time.time()
        logger.info(f"Session {room_name}: Intent updated → {req.intent}")

    if req.user_data:
        session["user_data"].update(req.user_data)
        if "selected_voice" in req.user_data:
            session["selected_voice"] = req.user_data["selected_voice"]
            session["voice_id"] = req.user_data["selected_voice"]
            logger.info(f"Session {room_name}: Voice updated → {req.user_data['selected_voice']}")

    if req.status:
        session["status"] = req.status
        session["status_updated_at"] = time.time()

    session["last_updated"] = time.time()

    return {
        "message": "Session updated successfully",
        "session_id": room_name,
        "current_intent": session.get("intent"),
        "status": session.get("status"),
    }

@app.get("/api/sessions/{room_name}")
def get_session_info(room_name: str):
    if room_name not in active_sessions:
        raise HTTPException(status_code=404, detail="Session not found")

    session = active_sessions[room_name]
    duration = time.time() - session["created_at"]

    return {
        "session_id": room_name,
        "participant_name": session.get("participant_name"),
        "intent": session.get("intent"),
        "status": session.get("status"),
        "duration_seconds": int(duration),
        "user_data": session.get("user_data", {}),
        "created_at": session["created_at"],
        "last_updated": session.get("last_updated"),
        "voice_id": session.get("voice_id"),
        "selected_voice": session.get("selected_voice"),
    }

@app.get("/api/sessions")
def list_active_sessions():
    sessions = []
    for room_name, session in active_sessions.items():
        duration = time.time() - session["created_at"]
        sessions.append({
            "session_id": room_name,
            "participant_name": session.get("participant_name"),
            "intent": session.get("intent"),
            "status": session.get("status"),
            "duration_seconds": int(duration),
            "has_user_data": bool(session.get("user_data")),
        })
    return {"total_sessions": len(sessions), "sessions": sessions}

@app.delete("/api/rooms/{room_name}")
def cleanup_room(room_name: str):
    logger.info(f"Room cleanup requested for: {room_name}")
    session = active_sessions.get(room_name)
    if session:
        duration = time.time() - session["created_at"]
        final_summary = {
            "session_id": room_name,
            "participant_name": session.get("participant_name"),
            "intent": session.get("intent"),
            "duration_seconds": int(duration),
            "user_data": session.get("user_data", {}),
            "voice_id": session.get("voice_id"),
            "selected_voice": session.get("selected_voice"),
            "status": "completed",
            "completed_at": time.time(),
        }
        logger.info(f"Session completed: {json.dumps(final_summary, indent=2)}")
        del active_sessions[room_name]
        return {"message": f"Room {room_name} cleaned up successfully", "session_summary": final_summary}
    return {"message": f"Room {room_name} cleanup requested (no session data found)"}

# --------------------------------------------------------------------
# Voices
# --------------------------------------------------------------------
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
        "7f423809-0011-4658-ba48-a411f5e516ba": "Ashwin - Warm Narrator",
        "a167e0f3-df7e-4d52-a9c3-f949145efdab": "Blake - Helpful Agent",
        "e07c00bc-4134-4eae-9ea4-1a55fb45746b": "Brooke - Big Sister",
    }
    return {"status": "success", "voices": fallback}

@app.post("/api/change_voice")
async def change_voice(req: VoiceChangeRequest):
    logger.info(f"🎤 Voice change request: room={req.room_name}, voice={req.voice_id}")
    # Update session record so UI reflects change immediately
    if req.room_name in active_sessions:
        active_sessions[req.room_name]["selected_voice"] = req.voice_id
        active_sessions[req.room_name]["voice_id"] = req.voice_id
    return {"status": "success", "voice_name": req.voice_id}

# --------------------------------------------------------------------
# Health
# --------------------------------------------------------------------
@app.get("/health")
async def health_check():
    return {"status": "ok", "active_sessions": len(active_sessions), "timestamp": time.time()}

# --------------------------------------------------------------------
# Run
# --------------------------------------------------------------------
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000,log_level="warning",access_log=False)
