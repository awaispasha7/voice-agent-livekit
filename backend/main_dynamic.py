# backend_refactored.py
# Alive5 Voice Agent Backend — Orchestrator First with Greeting Trigger

import os
import json
import time
import uuid
import logging
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, Any, Optional, List

import httpx
import uvicorn
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from livekit import api
from pydantic import BaseModel, Field

# ==== Orchestrator ====
from backend.conversational_orchestrator import (
    ConversationalOrchestrator,
    OrchestratorAction,
    OrchestratorDecision,
    create_orchestrator_from_template,
)

# -----------------------------------------------------------------------------
# Config
# -----------------------------------------------------------------------------
current_dir = Path(__file__).parent
load_dotenv(current_dir / "../../.env")

app = FastAPI(title="Alive5 Voice Agent — Orchestrator", version="3.0")

logging.basicConfig(level=logging.INFO, format="%(message)s", force=True)
logger = logging.getLogger("orchestrator-backend")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], allow_credentials=True, allow_methods=["*"], allow_headers=["*"],
)

LIVEKIT_API_KEY = os.getenv("LIVEKIT_API_KEY")
LIVEKIT_API_SECRET = os.getenv("LIVEKIT_API_SECRET")
LIVEKIT_URL = os.getenv("LIVEKIT_URL")

A5_BASE_URL = os.getenv("A5_BASE_URL")
A5_API_KEY = os.getenv("A5_API_KEY")
A5_TEMPLATE_URL = os.getenv("A5_TEMPLATE_URL", "/1.0/org-botchain/generate-template")
A5_FAQ_URL = os.getenv("A5_FAQ_URL", "/public/1.0/get-faq-bot-response-by-bot-id")
FAQ_BOT_ID = os.getenv("FAQ_BOT_ID", "default-bot-id")

# -----------------------------------------------------------------------------
# State & Persistence
# -----------------------------------------------------------------------------
PERSISTENCE_DIR = current_dir / "persistence"
FLOW_STATES_DIR = PERSISTENCE_DIR / "flow_states"
USER_PROFILES_DIR = PERSISTENCE_DIR / "user_profiles"

for d in [PERSISTENCE_DIR, FLOW_STATES_DIR, USER_PROFILES_DIR]:
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
    path = FLOW_STATES_DIR / f"{room}.json"
    with open(path, "w") as f:
        json.dump(state.dict(), f, indent=2)

def load_flow_state(room: str) -> Optional[FlowState]:
    path = FLOW_STATES_DIR / f"{room}.json"
    if path.exists():
        with open(path, "r") as f:
            return FlowState(**json.load(f))
    return None

# -----------------------------------------------------------------------------
# Models
# -----------------------------------------------------------------------------
class ProcessFlowMessageRequest(BaseModel):
    room_name: str
    user_message: str
    conversation_history: List[Dict[str, str]] = Field(default_factory=list)

# -----------------------------------------------------------------------------
# Core: Process Message via Orchestrator
# -----------------------------------------------------------------------------
@app.post("/api/process_flow_message")
async def process_flow_message(req: ProcessFlowMessageRequest):
    global conversational_orchestrator, bot_template

    if conversational_orchestrator is None:
        return {"status": "error", "message": "No orchestrator loaded"}

    room = req.room_name
    msg = req.user_message.strip() if req.user_message else ""

    # 👋 Greeting trigger from worker
    if msg == "__start__":
        if bot_template and "data" in bot_template:
            for flow_key, flow_data in bot_template["data"].items():
                if flow_data.get("type") == "greeting":
                    greeting_text = flow_data.get("text", "Hello! How can I help you today?")
                    state = flow_states.get(room) or FlowState()
                    state.current_flow = flow_key
                    state.current_step = flow_data.get("name")
                    state.flow_data = flow_data
                    flow_states[room] = state
                    save_flow_state(room, state)
                    return {
                        "status": "processed",
                        "flow_result": {
                            "type": "flow_started",
                            "flow_name": "greeting",
                            "response": greeting_text,
                            "flow_state": state.dict()
                        }
                    }
        # fallback if no greeting flow found
        return {
            "status": "processed",
            "flow_result": {
                "type": "message",
                "response": "Hello! Thanks for calling Alive5. How can I help today?",
                "flow_state": {}
            }
        }

    # Normal orchestrator pipeline
    state = flow_states.get(room) or load_flow_state(room) or FlowState()
    state.conversation_history.append({"role": "user", "content": msg})
    state.conversation_history = state.conversation_history[-10:]

    try:
        decision: OrchestratorDecision = await conversational_orchestrator.process_message(
            user_message=msg,
            room_name=room,
            conversation_history=state.conversation_history,
            current_flow_state=state.dict(),
            current_step_data=state.flow_data,
        )
    except Exception as e:
        logger.error(f"❌ Orchestrator error: {e}", exc_info=True)
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
            "flow_state": state.dict(),
        },
    }

# -----------------------------------------------------------------------------
# FAQ Wrapper
# -----------------------------------------------------------------------------
async def get_faq_response(user_message: str) -> Dict[str, Any]:
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

# -----------------------------------------------------------------------------
# Template Load
# -----------------------------------------------------------------------------
@app.post("/api/refresh_template")
async def refresh_template():
    global bot_template, conversational_orchestrator
    async with httpx.AsyncClient(timeout=15.0) as client:
        resp = await client.post(
            f"{A5_BASE_URL}{A5_TEMPLATE_URL}",
            headers={"X-A5-APIKEY": A5_API_KEY, "Content-Type": "application/json"},
            json={"botchain_name": os.getenv("A5_BOTCHAIN_NAME"), "org_name": os.getenv("A5_ORG_NAME")},
        )
        resp.raise_for_status()
        bot_template = resp.json()
        conversational_orchestrator = create_orchestrator_from_template(bot_template)
        return {"status": "success", "flows": list(conversational_orchestrator.available_flows.keys())}

# -----------------------------------------------------------------------------
# Health
# -----------------------------------------------------------------------------
@app.get("/health")
async def health_check():
    return {"status": "ok", "active_sessions": len(flow_states), "timestamp": time.time()}

# -----------------------------------------------------------------------------
# Run
# -----------------------------------------------------------------------------
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
