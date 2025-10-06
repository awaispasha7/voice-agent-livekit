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
        "http://localhost:8080",                   # additional local dev
    ],
    allow_credentials=True,
    allow_methods=["GET", "POST", "PUT", "DELETE", "OPTIONS"],
    allow_headers=["*"],
    expose_headers=["*"],
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
A5_FAQ_CONCISE_URL = os.getenv("A5_FAQ_CONCISE_URL", "/public/1.0/get-faq-bot-response-by-bot-id")  # TODO: Replace with actual concise endpoint
FAQ_BOT_ID    = os.getenv("FAQ_BOT_ID", "default-bot-id")

# --------------------------------------------------------------------
# State
# --------------------------------------------------------------------
PERSISTENCE_DIR = current_dir / "persistence"
FLOW_STATES_DIR = PERSISTENCE_DIR / "flow_states"
USER_PROFILES_DIR = PERSISTENCE_DIR / "user_profiles"
DEBUG_LOGS_DIR = PERSISTENCE_DIR / "debug_logs"

for d in [PERSISTENCE_DIR, FLOW_STATES_DIR, USER_PROFILES_DIR, DEBUG_LOGS_DIR]:
    d.mkdir(exist_ok=True)

class FlowState(BaseModel):
    current_flow: Optional[str] = None
    current_step: Optional[str] = None
    flow_data: Optional[Dict[str, Any]] = None
    conversation_history: List[Dict[str, str]] = Field(default_factory=list)
    user_responses: Optional[Dict[str, Any]] = None
    objectives: List[str] = Field(default_factory=list)
    refused_fields: List[str] = Field(default_factory=list)
    flow_contexts: Dict[str, Dict[str, Any]] = Field(default_factory=dict)  # Store context for each flow

flow_states: Dict[str, FlowState] = {}
bot_template: Optional[Dict[str, Any]] = None
conversational_orchestrator: Optional[ConversationalOrchestrator] = None

# --------------------------------------------------------------------
# Persistence Cleanup
# --------------------------------------------------------------------
def cleanup_old_persistence_files():
    """
    Clean up persistence files older than 7 days.
    This runs automatically to prevent storage bloat.
    """
    try:
        import time
        from datetime import datetime, timedelta
        
        cutoff_time = time.time() - (7 * 24 * 60 * 60)  # 7 days ago
        cleaned_count = 0
        
        # Clean up flow states
        if FLOW_STATES_DIR.exists():
            for file_path in FLOW_STATES_DIR.glob("*.json"):
                if file_path.stat().st_mtime < cutoff_time:
                    file_path.unlink()
                    cleaned_count += 1
                    logger.info(f"🗑️ Cleaned up old flow state: {file_path.name}")
        
        # Clean up user profiles
        if USER_PROFILES_DIR.exists():
            for file_path in USER_PROFILES_DIR.glob("*.json"):
                if file_path.stat().st_mtime < cutoff_time:
                    file_path.unlink()
                    cleaned_count += 1
                    logger.info(f"🗑️ Cleaned up old user profile: {file_path.name}")
        
        # Clean up debug logs
        if DEBUG_LOGS_DIR.exists():
            for file_path in DEBUG_LOGS_DIR.glob("*.json"):
                if file_path.stat().st_mtime < cutoff_time:
                    file_path.unlink()
                    cleaned_count += 1
                    logger.info(f"🗑️ Cleaned up old debug log: {file_path.name}")
        
        if cleaned_count > 0:
            logger.info(f"🧹 Weekly cleanup completed: {cleaned_count} old files removed")
        else:
            logger.info("🧹 Weekly cleanup completed: No old files found")
            
    except Exception as e:
        logger.error(f"❌ Error during persistence cleanup: {e}")

def schedule_weekly_cleanup():
    """
    Schedule automatic weekly cleanup of persistence files.
    """
    import asyncio
    import threading
    from datetime import datetime, timedelta
    
    def run_cleanup():
        """Run cleanup in a separate thread to avoid blocking the main application."""
        try:
            cleanup_old_persistence_files()
        except Exception as e:
            logger.error(f"❌ Error in scheduled cleanup: {e}")
    
    def schedule_next_cleanup():
        """Calculate next cleanup time and schedule it."""
        now = datetime.now()
        # Schedule for next Monday at 2 AM
        days_until_monday = (7 - now.weekday()) % 7
        if days_until_monday == 0:  # If it's Monday, schedule for next Monday
            days_until_monday = 7
        
        next_cleanup = now.replace(hour=2, minute=0, second=0, microsecond=0) + timedelta(days=days_until_monday)
        
        # Calculate seconds until next cleanup
        seconds_until_cleanup = (next_cleanup - now).total_seconds()
        
        logger.info(f"📅 Next persistence cleanup scheduled for: {next_cleanup.strftime('%Y-%m-%d %H:%M:%S')}")
        
        # Schedule the cleanup
        timer = threading.Timer(seconds_until_cleanup, run_cleanup)
        timer.daemon = True
        timer.start()
        
        # After cleanup, schedule the next one
        def reschedule():
            schedule_next_cleanup()
        
        # Schedule the next cleanup 7 days after this one
        next_timer = threading.Timer(seconds_until_cleanup + (7 * 24 * 60 * 60), reschedule)
        next_timer.daemon = True
        next_timer.start()
    
    # Start the scheduling
    schedule_next_cleanup()
    logger.info("🔄 Weekly persistence cleanup scheduler started")

# Start the weekly cleanup scheduler
schedule_weekly_cleanup()

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

# ===== helpers for node execution / progression =====

async def execute_action_bot(action_data: Dict[str, Any], flow_state: FlowState) -> Dict[str, Any]:
    """
    Execute an Alive5 Action Bot node.
    Supported minimal actions:
      - action_type == "webhook": POST to a URL with simple payload
      - action_type == "email"  : mock success (hook real email service here)
      - action_type == "url"    : return "opened" info
    Anything else → echo the node's text.
    Returns: {"response": <text>, "action_data": {...}}
    """
    try:
        action_type = (action_data.get("action_type") or "").lower()
        action_text = action_data.get("text", "Action completed.")

        if action_type == "webhook":
            webhook_url = action_data.get("webhook_url")
            payload = {
                "user_message": action_text,
                "flow_state": flow_state.dict() if flow_state else None,
            }
            if webhook_url:
                async with httpx.AsyncClient(timeout=httpx.Timeout(10.0)) as client:
                    r = await client.post(webhook_url, json=payload)
                    r.raise_for_status()
                    result = r.json() if r.headers.get("content-type","").startswith("application/json") else {"status": r.status_code}
                return {"response": result.get("message", "Webhook executed successfully."), "action_data": result}

        elif action_type == "email":
            email_data = action_data.get("email_data", {})
            # integrate real email provider here
            return {"response": "Email sent successfully.", "action_data": {"email_sent": True, "recipient": email_data.get("to")}}

        elif action_type == "url":
            url = action_data.get("url")
            if url:
                return {"response": f"Opening URL: {url}", "action_data": {"url_opened": url}}

        # default
        return {"response": action_text, "action_data": {"action_type": action_type or "unknown"}}
            
    except Exception as e:
        logger.error(f"execute_action_bot error: {e}")
        return {"response": "I'm sorry, there was an error executing that action. Please try again.", "action_data": {"error": str(e)}}


async def evaluate_condition_bot(condition_data: Dict[str, Any], flow_state: FlowState, user_message: str) -> Dict[str, Any]:
    """
    Execute an Alive5 Condition Bot node.
    Two minimal modes:
      - condition_type == "variable": compare a stored variable with expected_value (string equality)
      - condition_type == "user_input": substring match on user input
    The node can contain 'true_flow' / 'false_flow' and corresponding responses.
    Returns: {"response": <text>, "condition_result": {...}}
    """
    try:
        condition_type = (condition_data.get("condition_type") or "variable").lower()
        response_text  = condition_data.get("text", "")

        if condition_type == "variable":
            var_name = condition_data.get("variable_name")
            expected = str(condition_data.get("expected_value", "")).lower()

            actual = ""
            if flow_state and flow_state.user_responses and var_name:
                actual = str(flow_state.user_responses.get(var_name, "")).lower()

            met = (actual == expected)
            next_flow = condition_data.get("true_flow") if met else condition_data.get("false_flow")
            resp      = condition_data.get("true_response") if met else condition_data.get("false_response")
            if next_flow:
                flow_state.current_step = next_flow.get("name")
                flow_state.flow_data    = next_flow
            return {
                "response": resp or response_text or ("Condition met." if met else "Condition not met."),
                "condition_result": {
                    "condition_met": met, "variable_name": var_name,
                    "expected_value": expected, "actual_value": actual
                }
            }

        if condition_type == "user_input":
            pattern = str(condition_data.get("condition_pattern","")).lower()
            met = pattern in (user_message or "").lower() if pattern else False
            next_flow = condition_data.get("true_flow") if met else condition_data.get("false_flow")
            resp      = condition_data.get("true_response") if met else condition_data.get("false_response")
            if next_flow:
                flow_state.current_step = next_flow.get("name")
                flow_state.flow_data    = next_flow
            return {
                "response": resp or response_text or ("Matched." if met else "Not matched."),
                "condition_result": {"condition_met": met, "pattern": pattern, "user_input": user_message}
            }

        # default passthrough
        return {"response": response_text or "Okay.", "condition_result": {"condition_type": condition_type}}
                
    except Exception as e:
        logger.error(f"evaluate_condition_bot error: {e}")
        return {"response": "I'm sorry, there was an error evaluating that condition. Please try again.", "condition_result": {"error": str(e)}}


async def _emit_or_advance_and_emit(state: FlowState) -> str:
    """
    Emit current node's text; if node has next_flow and doesn't require input, advance first.
    Returns the text to speak.
    """
    node = state.flow_data or {}
    node_type = (node.get("type") or "").lower()
    
    logger.info(f"🔍 _emit_or_advance_and_emit: node_type='{node_type}', node={node}")

    # nodes that do not require user input -> auto-advance once before speaking (message -> next question, etc.)
    auto_advance_types = {"message", "intent_bot"}
    if node_type in auto_advance_types and node.get("next_flow"):
        nxt = node["next_flow"]
        state.current_step = nxt.get("name")
        state.flow_data = nxt
        node = state.flow_data
        node_type = (node.get("type") or "").lower()
        logger.info(f"🔍 Auto-advanced to: node_type='{node_type}', node={node}")

    # emit appropriate content
    if node_type in {"greeting", "message", "question", "faq"}:
        text = node.get("text", "") or "Okay."
        logger.info(f"🔍 Returning text for {node_type}: '{text}'")
        return text
    if node_type == "intent_bot":
        # This should have been auto-advanced, but if not, return the text
        text = node.get("text", "") or "Okay."
        logger.info(f"🔍 Returning intent_bot text (should have auto-advanced): '{text}'")
        return text
    if node_type == "agent":
        text = node.get("text", "I'm connecting you with a human agent. Please hold on.")
        logger.info(f"🔍 Returning agent text: '{text}'")
        return text
    if node_type in {"action", "condition"}:
        # let the main dispatcher handle these during progression
        text = node.get("text", "") or "Okay."
        logger.info(f"🔍 Returning action/condition text: '{text}'")
        return text
    # sms opt-in (some tenants label it differently)
    if node_type in {"sms_opt_in", "sms_optin", "smsoptin"}:
        text = node.get("text", "Would you like to opt in to SMS?")
        logger.info(f"🔍 Returning SMS opt-in text: '{text}'")
        return text

    # Default fallback
    text = node.get("text", "") or "Okay."
    logger.info(f"🔍 Returning default text: '{text}'")
    return text


async def _progress_flow_with_user_input(state: FlowState, user_message: str, faq_verbose_mode: bool = True) -> Dict[str, Any]:
    """
    Interpret the current node and user input; progress the flow; return dict:
    { "type": <node_type>, "response": <text>, "advanced": bool }
    """
    node = state.flow_data or {}
    node_type = (node.get("type") or "").lower()

    # 1) QUESTION nodes
    if node_type == "question":
        answers = node.get("answers") or {}
        if answers:
            # multiple-choice → match user to an answer key
            from backend.llm_utils import match_answer_with_llm
            qtext = node.get("text", "")
            choice_key = match_answer_with_llm(qtext, user_message, answers)
            if choice_key and choice_key in answers:
                branch = answers[choice_key]
                # go to the branch node
                state.current_step = branch.get("name")
                state.flow_data    = branch
                # optionally auto-jump to next question after a message node
                speak = branch.get("text", "")
                nxt = branch.get("next_flow")
                if speak and nxt and (nxt.get("type") or "").lower() == "question":
                    # chain message + next question
                    state.current_step = nxt.get("name")
                    state.flow_data    = nxt
                    speak = (speak + " " + (nxt.get("text",""))).strip()
                return {"type": branch.get("type","message"), "response": speak or "Okay.", "advanced": True}
            # no match → ask to rephrase
            return {"type": "message", "response": "I didn't quite catch that. Could you pick one of the options?", "advanced": False}
        else:
            # free text answer – extract, validate confidence, persist, and advance
            from backend.llm_utils import extract_answer_with_llm
            qtext = node.get("text", "")
            extracted = extract_answer_with_llm(qtext, user_message)  # {"status","kind","value","confidence"}

            status     = extracted.get("status")
            confidence = float(extracted.get("confidence", 0.0) or 0.0)
            value      = extracted.get("value")
            kind       = (extracted.get("kind") or "").lower()

            # If unclear or very low confidence, re-ask instead of blindly advancing
            if status != "extracted" or confidence < 0.6:
                return {
                    "type": "message",
                    "response": "Sorry, I didn't quite get that. Could you say it again?",
                    "advanced": False
                }

            # Persist for downstream (e.g., Condition bots or summaries)
            state.user_responses = (state.user_responses or {})
            # store under current step name (you could normalize the key if you prefer)
            state.user_responses[state.current_step] = value

            # Advance if next node exists, otherwise end with confirmation and clear flow state
            nxt = node.get("next_flow")
            if nxt:
                state.current_step = nxt.get("name")
                state.flow_data    = nxt
                
                # Generate natural acknowledgment with next step context
                logger.info(f"🔍 Generating acknowledgment with next step context: '{user_message}'")
                from backend.llm_utils import generate_conversational_response
                next_step_text = nxt.get("text", "")
                conversational_context = {
                    "conversation_history": state.conversation_history,
                    "current_flow": state.current_flow,
                    "current_step": state.current_step,
                    "next_step_text": next_step_text,  # Provide context about what's coming next
                    "profile": {
                        "collected_info": state.user_responses,
                        "objectives": state.objectives,
                        "refused_fields": state.refused_fields
                    },
                    "refusal_context": False,
                    "uncertainty_context": False
                }
                ack = await generate_conversational_response(f"I answered: {value}", conversational_context)
                if not ack or len(ack.strip()) < 3:  # Fallback if LLM response is too short
                    ack = "Thanks."
                
                # Return just the acknowledgment, let the flow handle the next question separately
                return {"type": "acknowledgment", "response": ack, "advanced": True}

            # No next node; clear flow state and just acknowledge
            state.current_flow = None
            state.current_step = None
            state.flow_data = None
            return {"type": "message", "response": f"{ack}", "advanced": False}


    # 2) MESSAGE → just go to next
    if node_type == "message":
        nxt = node.get("next_flow")
        if nxt:
            state.current_step = nxt.get("name")
            state.flow_data    = nxt
            return {"type": nxt.get("type","message"), "response": nxt.get("text","") or "Okay.", "advanced": True}
        # No next node; clear flow state
        state.current_flow = None
        state.current_step = None
        state.flow_data = None
        return {"type": "message", "response": node.get("text","") or "Okay.", "advanced": False}

    # 3) FAQ
    if node_type == "faq":
        # use FAQ for the user's question; stay on faq or advance if template specifies
        faq = await get_faq_response(user_message, faq_verbose_mode)
        nxt = node.get("next_flow")
        if nxt:
            state.current_step = nxt.get("name")
            state.flow_data    = nxt
            speak = (faq["response"] + " " + nxt.get("text","")).strip()
            return {"type": "faq", "response": speak, "advanced": True}
        # No next node; clear flow state
        state.current_flow = None
        state.current_step = None
        state.flow_data = None
        return {"type": "faq", "response": faq["response"], "advanced": False}

    # 4) AGENT
    if node_type == "agent":
        # Agent handoff should clear the flow state since the conversation ends
        state.current_flow = None
        state.current_step = None
        state.flow_data = None
        return {"type": "agent_handoff", "response": node.get("text","I'm connecting you with a human agent. Please hold on."), "advanced": False}

    # 5) ACTION
    if node_type == "action":
        result = await execute_action_bot(node, state)
        nxt = node.get("next_flow")
        if nxt:
            state.current_step = nxt.get("name")
            state.flow_data    = nxt
            speak = (result["response"] + " " + nxt.get("text","")).strip()
            return {"type": "action_completed", "response": speak, "advanced": True}
        # No next node; clear flow state
        state.current_flow = None
        state.current_step = None
        state.flow_data = None
        return {"type": "action_completed", "response": result["response"], "advanced": False}

    # 6) CONDITION
    if node_type == "condition":
        result = await evaluate_condition_bot(node, state, user_message)
        nxt = state.flow_data  # evaluate_condition_bot may have advanced the flow_data
        speak = result["response"]
        # Condition evaluation may or may not have advanced the flow, but we should check if there's a next step
        if nxt and nxt.get("next_flow"):
            return {"type": "condition_evaluated", "response": speak, "advanced": True}
        # No valid next step; clear flow state
        state.current_flow = None
        state.current_step = None
        state.flow_data = None
        return {"type": "condition_evaluated", "response": speak, "advanced": False}

    # 7) SMS OPT-IN (treat like message/action hybrid)
    if node_type in {"sms_opt_in", "sms_optin", "smsoptin"}:
        nxt = node.get("next_flow")
        if nxt:
            state.current_step = nxt.get("name")
            state.flow_data    = nxt
            speak = (node.get("text","") + " " + nxt.get("text","")).strip() or "Okay."
            return {"type": "sms_opt_in", "response": speak, "advanced": True}
        # No next node; clear flow state
        state.current_flow = None
        state.current_step = None
        state.flow_data = None
        return {"type": "sms_opt_in", "response": node.get("text","") or "Okay.", "advanced": False}

    # default: just emit text and clear flow state for unhandled node types
    state.current_flow = None
    state.current_step = None
    state.flow_data = None
    return {"type": node_type or "message", "response": node.get("text","") or "Okay.", "advanced": False}



@app.post("/api/process_flow_message")
async def process_flow_message(req: ProcessFlowMessageRequest):
    global conversational_orchestrator, bot_template
    if conversational_orchestrator is None:
        return {"status": "error", "message": "No orchestrator loaded"}

    room, msg = req.room_name, (req.user_message or "").strip()
    
    # Get FAQ verbose mode from session data
    session_data = active_sessions.get(room, {})
    user_data = session_data.get("user_data", {})
    faq_verbose_mode = user_data.get("faq_verbose_mode", True)  # Default to verbose

    # Greeting trigger from worker
    if msg == "__start__":
        if bot_template and "data" in bot_template:
            for flow_key, flow_data in bot_template["data"].items():
                if (flow_data.get("type") or "").lower() == "greeting":
                    greeting = flow_data.get("text", "Hello! How can I help?")
                    # Greeting flows are one-time messages, not persistent states
                    # Clear any existing flow state and just return the greeting
                    state = FlowState()  # Fresh state, no flow persistence
                    flow_states[room]  = state
                    save_flow_state(room, state)
                    return {"status":"processed","flow_result":{
                        "type":"greeting","flow_name":"greeting",
                        "response":greeting,"flow_state":state.dict()}}
        return {"status":"processed","flow_result":{
            "type":"message","response":"Hello! Thanks for calling Alive5. How can I help today?","flow_state":{}}}

    state = flow_states.get(room) or load_flow_state(room) or FlowState()
    state.conversation_history.append({"role":"user","content":msg})
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
        return {"status":"error","flow_result":{"type":"error","response":"System error"}}

    response_text = decision.response or ""
    logger.info(f"🔍 Orchestrator decision: action={decision.action}, response='{response_text}', flow_to_execute='{decision.flow_to_execute}'")

    if decision.action == OrchestratorAction.EXECUTE_FLOW and decision.flow_to_execute:
        # Find the flow by text (since orchestrator uses flow text as key)
        flow_container = None
        for flow_key, flow_data in conversational_orchestrator.available_flows.items():
            if flow_data.get("name") == decision.flow_to_execute:
                flow_container = flow_data
                break
        
        if not flow_container:
            logger.warning(f"Flow '{decision.flow_to_execute}' not found in available flows")
            faq = await get_faq_response(msg, faq_verbose_mode); response_text = faq["response"]
        else:
            # Check if we're already in this flow - if so, progress it instead of restarting
            if (state.current_flow == flow_container.get("key") and 
                state.flow_data and 
                state.flow_data.get("type") != "intent_bot"):
                logger.info(f"🔍 Already in flow '{decision.flow_to_execute}', progressing current flow instead of restarting")
                # We're already in this flow, so progress it with user input
                progressed = await _progress_flow_with_user_input(state, msg, faq_verbose_mode)
                
                # If the flow couldn't be progressed, exit the flow state
                if not progressed.get("advanced", False):
                    logger.info(f"🔍 Flow could not be progressed, exiting flow state")
                    state.current_flow = None
                    state.current_step = None
                    state.flow_data = None
                
                flow_states[room] = state; save_flow_state(room, state)
                return {"status":"processed","flow_result":{
                    "type": progressed["type"],
                    "response": progressed["response"],
                    "flow_state": state.dict()
                }}
            else:
                # Check if we have collected information for this flow that we can resume from
                flow_key = flow_container.get("key")
                logger.info(f"🔍 Starting flow '{decision.flow_to_execute}' - checking for existing context")
                
                # Check if we have any collected information that might be relevant to this flow
                has_relevant_context = False
                if state.user_responses:
                    # Check if we have information that could be relevant to this flow
                    for key, value in state.user_responses.items():
                        if value and str(value).strip():
                            has_relevant_context = True
                            logger.info(f"🔍 Found existing context: {key}={value}")
                            break
                
                if has_relevant_context:
                    logger.info(f"🔍 Resuming flow '{decision.flow_to_execute}' with existing context")
                    # Check if we have saved context for this specific flow
                    flow_key = flow_container.get("key")
                    if flow_key in state.flow_contexts:
                        saved_context = state.flow_contexts[flow_key]
                        logger.info(f"🔍 Found saved context for flow {flow_key}: {saved_context}")
                        
                        # Restore the flow state from saved context
                        state.current_flow = flow_key
                        state.current_step = saved_context.get("current_step")
                        state.flow_data = saved_context.get("flow_data")
                        
                        # Merge any new user responses with the saved context
                        if state.user_responses and saved_context.get("user_responses"):
                            saved_context["user_responses"].update(state.user_responses)
                            state.user_responses = saved_context["user_responses"]
                        elif saved_context.get("user_responses"):
                            state.user_responses = saved_context["user_responses"]
                    else:
                        # No saved context, start fresh but preserve current responses
                        intent_node = flow_container.get("data", {})
                        state.current_flow = flow_container.get("key")
                        state.current_step = intent_node.get("name")
                        state.flow_data = intent_node
                else:
                    logger.info(f"🔍 Starting fresh flow '{decision.flow_to_execute}'")
                    intent_node = flow_container.get("data", {})
                    state.current_flow = flow_container.get("key")
                    state.current_step = intent_node.get("name")
                    state.flow_data = intent_node
                
                logger.info(f"🔍 Flow execution: flow_container={flow_container}")
                logger.info(f"🔍 Flow execution: intent_node={intent_node}")
                logger.info(f"🔍 Flow execution: state.flow_data={state.flow_data}")

                # advance once if intent node leads straight to a question/message
                response_text = await _emit_or_advance_and_emit(state)
                logger.info(f"Executing flow '{decision.flow_to_execute}' with response: '{response_text}'")

        # Save the current flow context before switching
        if state.current_flow and state.current_flow != flow_container.get("key"):
            logger.info(f"🔍 Saving context for previous flow: {state.current_flow}")
            state.flow_contexts[state.current_flow] = {
                "current_step": state.current_step,
                "flow_data": state.flow_data,
                "user_responses": state.user_responses.copy() if state.user_responses else {}
            }
        
        flow_states[room] = state; save_flow_state(room, state)
        return {"status":"processed","flow_result":{
            "type":"flow_started","flow_name":decision.flow_to_execute,
            "response":response_text,"flow_state":state.dict()}}

    elif decision.action == OrchestratorAction.USE_FAQ:
        faq = await get_faq_response(msg, faq_verbose_mode); response_text = faq["response"]

    elif decision.action == OrchestratorAction.HANDLE_REFUSAL:
        # After handling refusal, advance to the next question in the flow
        if state.flow_data and state.flow_data.get("next_flow"):
            logger.info(f"🔍 Progressing flow after refusal handling")
            logger.info(f"🔍 Current flow_data: {state.flow_data}")
            # Advance to the next flow step
            next_flow = state.flow_data.get("next_flow")
            logger.info(f"🔍 Next flow: {next_flow}")
            state.current_step = next_flow.get("name")
            state.flow_data = next_flow
            logger.info(f"🔍 Advanced to step: {state.current_step}")
            
            # Generate a natural transition response that includes the next question
            logger.info(f"🔍 Generating conversational response for refusal with next step context: '{msg}'")
            from backend.llm_utils import generate_conversational_response
            next_step_text = next_flow.get("text", "")
            logger.info(f"🔍 Next step text: '{next_step_text}'")
            conversational_context = {
                "conversation_history": state.conversation_history,
                "current_flow": state.current_flow,
                "current_step": state.current_step,
                "next_step_text": next_step_text,  # Provide context about what's coming next
                "profile": {
                    "collected_info": state.user_responses,
                    "objectives": state.objectives,
                    "refused_fields": state.refused_fields
                },
                "refusal_context": True,
                "uncertainty_context": False
            }
            response_text = await generate_conversational_response(msg, conversational_context)
            logger.info(f"🔍 Generated refusal response: '{response_text}'")
            
            # If we still don't have a good response, just use the next question
            if not response_text or response_text.strip() == "":
                next_response = await _emit_or_advance_and_emit(state)
                if next_response and next_response.strip():
                    response_text = next_response
                    logger.info(f"🔍 Using next question as response: '{response_text}'")
        else:
            # No next flow, just generate a refusal response
            if not response_text or response_text.strip() == "":
                logger.info(f"🔍 Generating conversational response for refusal: '{msg}'")
                from backend.llm_utils import generate_conversational_response
                conversational_context = {
                    "conversation_history": state.conversation_history,
                    "current_flow": state.current_flow,
                    "current_step": state.current_step,
                    "profile": {
                        "collected_info": state.user_responses,
                        "objectives": state.objectives,
                        "refused_fields": state.refused_fields
                    },
                    "refusal_context": True,
                    "uncertainty_context": False
                }
                response_text = await generate_conversational_response(msg, conversational_context)

    elif decision.action == OrchestratorAction.HANDLE_UNCERTAINTY:
        # Handle uncertainty differently based on whether we're in a flow or not
        if state.flow_data:
            # We're in a flow - skip the current question and move to next
            logger.info(f"🔍 Handling uncertainty within flow context - skipping current question")
            
            # For questions, we need to find the next question in the flow structure
            current_node = state.flow_data
            next_question = None
            
            # If current node is a question, look for the next question in the answers
            if current_node.get("type") == "question":
                answers = current_node.get("answers", {})
                logger.info(f"🔍 Current question has {len(answers)} answer options")
                
                # Find the first answer that leads to another question
                for answer_key, answer_data in answers.items():
                    if answer_data.get("type") == "message" and answer_data.get("next_flow"):
                        next_flow = answer_data.get("next_flow")
                        if next_flow.get("type") == "question":
                            next_question = next_flow
                            logger.info(f"🔍 Found next question via answer '{answer_key}': {next_question.get('text', '')}")
                            break
                
                # If no direct next question found, try to find any next question in the flow
                if not next_question:
                    for answer_key, answer_data in answers.items():
                        if answer_data.get("next_flow"):
                            next_flow = answer_data.get("next_flow")
                            # Check if this next_flow leads to a question
                            if next_flow.get("type") == "question":
                                next_question = next_flow
                                logger.info(f"🔍 Found next question via answer '{answer_key}': {next_question.get('text', '')}")
                                break
                            # Or if it's a message that leads to a question
                            elif next_flow.get("type") == "message" and next_flow.get("next_flow"):
                                next_next_flow = next_flow.get("next_flow")
                                if next_next_flow.get("type") == "question":
                                    next_question = next_next_flow
                                    logger.info(f"🔍 Found next question via answer '{answer_key}' -> message -> question: {next_question.get('text', '')}")
                                    break
            
            # Check if we found a next question or if there's a direct next_flow
            if next_question:
                logger.info(f"🔍 Progressing flow after uncertainty handling to next question")
                # Advance to the next question
                state.current_step = next_question.get("name")
                state.flow_data = next_question
                
                # Generate a natural transition response that includes the next question
                logger.info(f"🔍 Generating conversational response for uncertainty with next step context: '{msg}'")
                from backend.llm_utils import generate_conversational_response
                next_step_text = next_question.get("text", "")
                logger.info(f"🔍 Next step text: '{next_step_text}'")
                conversational_context = {
                    "conversation_history": state.conversation_history,
                    "current_flow": state.current_flow,
                    "current_step": state.current_step,
                    "next_step_text": next_step_text,  # Provide context about what's coming next
                    "profile": {
                        "collected_info": state.user_responses,
                        "objectives": state.objectives,
                        "refused_fields": state.refused_fields
                    },
                    "refusal_context": False,
                    "uncertainty_context": True
                }
                response_text = await generate_conversational_response(msg, conversational_context)
                logger.info(f"🔍 Generated uncertainty response: '{response_text}'")
                
                # If we still don't have a good response, just use the next question
                if not response_text or response_text.strip() == "":
                    next_response = await _emit_or_advance_and_emit(state)
                    if next_response and next_response.strip():
                        response_text = next_response
                        logger.info(f"🔍 Using next question as response: '{response_text}'")
            elif state.flow_data.get("next_flow"):
                logger.info(f"🔍 Progressing flow after uncertainty handling via direct next_flow")
                # Advance to the next flow step
                next_flow = state.flow_data.get("next_flow")
                state.current_step = next_flow.get("name")
                state.flow_data = next_flow
                
                # Generate a natural transition response that includes the next question
                logger.info(f"🔍 Generating conversational response for uncertainty with next step context: '{msg}'")
                from backend.llm_utils import generate_conversational_response
                next_step_text = next_flow.get("text", "")
                logger.info(f"🔍 Next step text: '{next_step_text}'")
                conversational_context = {
                    "conversation_history": state.conversation_history,
                    "current_flow": state.current_flow,
                    "current_step": state.current_step,
                    "next_step_text": next_step_text,  # Provide context about what's coming next
                    "profile": {
                        "collected_info": state.user_responses,
                        "objectives": state.objectives,
                        "refused_fields": state.refused_fields
                    },
                    "refusal_context": False,
                    "uncertainty_context": True
                }
                response_text = await generate_conversational_response(msg, conversational_context)
                logger.info(f"🔍 Generated uncertainty response: '{response_text}'")
                
                # If we still don't have a good response, just use the next question
                if not response_text or response_text.strip() == "":
                    next_response = await _emit_or_advance_and_emit(state)
                    if next_response and next_response.strip():
                        response_text = next_response
                        logger.info(f"🔍 Using next question as response: '{response_text}'")
            else:
                # No next flow, just acknowledge uncertainty and clear flow state
                response_text = "No problem, I understand you're not sure. Let me know if you need anything else."
                logger.info(f"🔍 No next flow available, clearing flow state")
                state.current_flow = None
                state.current_step = None
                state.flow_data = None
        else:
            # We're not in a flow - handle uncertainty conversationally
            logger.info(f"🔍 Handling uncertainty in conversational context: '{msg}'")
            if not response_text or response_text.strip() == "":
                from backend.llm_utils import generate_conversational_response
                conversational_context = {
                    "conversation_history": state.conversation_history,
                    "current_flow": state.current_flow,
                    "current_step": state.current_step,
                    "profile": {
                        "collected_info": state.user_responses,
                        "objectives": state.objectives,
                        "refused_fields": state.refused_fields
                    },
                    "refusal_context": False,
                    "uncertainty_context": True
                }
                response_text = await generate_conversational_response(msg, conversational_context)

    elif decision.action == OrchestratorAction.SPEAK_WITH_PERSON:
        response_text = response_text or "Connecting you to an agent now."
        return {"status":"processed","flow_result":{"type":"agent_handoff","response":response_text}}

    elif decision.action == OrchestratorAction.END_CALL:
        response_text = response_text or "Thank you for calling! Have a great day!"
        # Clear flow state and end the conversation
        state.current_flow = None
        state.current_step = None
        state.flow_data = None
        flow_states[room] = state; save_flow_state(room, state)
        return {"status":"processed","flow_result":{"type":"call_ended","response":response_text,"flow_state":state.dict()}}

    else:
        # If orchestrator chose conversational handling AND we're in a node,
        # try to progress the node per user input (question/faq/action/condition/etc.)
        # BUT skip this for greeting flows - they should just generate conversational responses
        current_flow_type = state.flow_data.get("type", "").lower() if state.flow_data else ""
        logger.info(f"🔍 Flow progression check: state.flow_data={bool(state.flow_data)}, decision.action={decision.action}, current_flow_type='{current_flow_type}'")
        if (state.flow_data and
            (decision.action == OrchestratorAction.HANDLE_CONVERSATIONALLY) and
            current_flow_type != "greeting"):
            progressed = await _progress_flow_with_user_input(state, msg, faq_verbose_mode)

            # If the flow couldn't be progressed, exit the flow state for conversational handling
            if not progressed.get("advanced", False):
                logger.info(f"🔍 Flow could not be progressed, exiting flow state for conversational handling")
                state.current_flow = None
                state.current_step = None
                state.flow_data = None
            elif progressed.get("type") == "acknowledgment":
                # For acknowledgments, the LLM has already incorporated the next step context
                # No need to emit additional questions - the response is complete
                logger.info(f"🔍 Handling acknowledgment: '{progressed['response']}'")
                logger.info(f"🔍 Acknowledgment is complete, no additional processing needed")

            flow_states[room] = state; save_flow_state(room, state)
            return {"status":"processed","flow_result":{
                "type": progressed["type"],
                "response": progressed["response"],
                "flow_state": state.dict()
            }}

        # If orchestrator chose conversational handling but no response provided, generate one
        # Also clear flow state if we're in a dead-end flow (no valid progression)
        if decision.action == OrchestratorAction.HANDLE_CONVERSATIONALLY:
            if state.flow_data:
                current_flow_type = state.flow_data.get("type", "").lower()
                next_flow = state.flow_data.get("next_flow")

                # Clear greeting flow state
                if current_flow_type == "greeting":
                    logger.info(f"🔍 Clearing greeting flow state for conversational handling")
                    state.current_flow = None
                    state.current_step = None
                    state.flow_data = None
                # Clear flow state if it has no valid next step
                elif not next_flow or next_flow.get("name") == "N/A":
                    logger.info(f"🔍 Clearing dead-end flow state for conversational handling")
                    state.current_flow = None
                    state.current_step = None
                    state.flow_data = None

        if decision.action == OrchestratorAction.HANDLE_CONVERSATIONALLY and (not response_text or response_text.strip() == ""):
            logger.info(f"🔍 Generating conversational response for: '{msg}'")
            from backend.llm_utils import generate_conversational_response
            conversational_context = {
                "user_message": msg,
                "conversation_history": state.conversation_history,
                "current_flow": state.current_flow,
                "current_step": state.current_step,
                "profile": {},  # Could add user profile here if needed
                "refusal_context": False,
                "skipped_fields": []
            }
            response_text = await generate_conversational_response(msg, conversational_context)
            logger.info(f"🔍 Generated conversational response: '{response_text}'")
        else:
            logger.info(f"🔍 Not generating conversational response: action={decision.action}, response_text='{response_text}'")

        # fallback: if nothing to say, use FAQ to avoid "..."
        if not response_text:
            faq = await get_faq_response(msg, faq_verbose_mode)
            response_text = faq["response"]

    flow_states[room] = state
    save_flow_state(room, state)
    return {"status":"processed","flow_result":{
        "type": decision.action.value,
        "response": response_text,
        "flow_state": state.dict()
    }}


# --------------------------------------------------------------------
# FAQ Wrapper
# --------------------------------------------------------------------
async def get_faq_response(user_message: str, verbose: bool = True) -> Dict[str, Any]:
    if not (A5_BASE_URL and A5_FAQ_URL and A5_API_KEY):
        logger.error("❌ FAQ call missing env: A5_BASE_URL/A5_FAQ_URL/A5_API_KEY")
        return {"response": "Sorry, FAQ service is not configured."}
    
    # Use different API endpoints based on verbose mode
    faq_url = A5_FAQ_URL if verbose else A5_FAQ_CONCISE_URL
    
    async with httpx.AsyncClient(timeout=10.0) as client:
        payload = {
            "bot_id": FAQ_BOT_ID, 
            "faq_question": user_message,
            "verbose": verbose  # Add verbose parameter to request
        }
        
        resp = await client.post(
            f"{A5_BASE_URL}{faq_url}",
            headers={"X-A5-APIKEY": A5_API_KEY, "Content-Type": "application/json"},
            json=payload,
        )
        if resp.status_code == 200:
            data = resp.json()
            return {"response": data.get("data", {}).get("answer", "I'm not sure.")}
        return {"response": "Sorry, I couldn't fetch that."}

# --------------------------------------------------------------------
# CORS OPTIONS Handlers (explicit preflight support)
# --------------------------------------------------------------------
@app.options("/api/available_voices")
async def options_available_voices():
    return {"message": "OK"}

@app.options("/api/refresh_template")
async def options_refresh_template():
    return {"message": "OK"}

@app.options("/api/connection_details")
async def options_connection_details():
    return {"message": "OK"}

@app.options("/api/process_flow_message")
async def options_process_flow_message():
    return {"message": "OK"}

@app.options("/api/change_voice")
async def options_change_voice():
    return {"message": "OK"}

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
# Security & Bot Protection
# --------------------------------------------------------------------
@app.get("/admin/{path:path}")
async def admin_security_redirect(path: str):
    """Handle common admin scanning attempts"""
    logger.warning(f"🚨 Admin scan attempt: /admin/{path} from unknown source")
    return {"error": "Not found", "message": "Admin endpoints not available"}

@app.get("/config/{path:path}")
async def config_security_redirect(path: str):
    """Handle common config scanning attempts"""
    logger.warning(f"🚨 Config scan attempt: /config/{path} from unknown source")
    return {"error": "Not found", "message": "Configuration endpoints not available"}

@app.get("/.env")
async def env_security_redirect():
    """Handle .env file scanning attempts"""
    logger.warning(f"🚨 Environment file scan attempt from unknown source")
    return {"error": "Not found", "message": "Environment files not accessible"}

# --------------------------------------------------------------------
# Health & Maintenance
# --------------------------------------------------------------------
@app.get("/health")
async def health_check():
    return {"status": "ok", "active_sessions": len(active_sessions), "timestamp": time.time()}

@app.post("/api/cleanup_persistence")
async def manual_cleanup_persistence():
    """
    Manually trigger persistence cleanup.
    Useful for administrative purposes or testing.
    """
    try:
        cleanup_old_persistence_files()
        return {
            "status": "success", 
            "message": "Persistence cleanup completed successfully",
            "timestamp": time.time()
        }
    except Exception as e:
        logger.error(f"Manual cleanup failed: {e}")
        return {
            "status": "error", 
            "message": f"Cleanup failed: {str(e)}",
            "timestamp": time.time()
        }

@app.get("/api/persistence_stats")
async def get_persistence_stats():
    """
    Get statistics about persistence files.
    """
    try:
        stats = {
            "flow_states": 0,
            "user_profiles": 0,
            "debug_logs": 0,
            "total_size_mb": 0
        }
        
        total_size = 0
        
        # Count flow states
        if FLOW_STATES_DIR.exists():
            flow_files = list(FLOW_STATES_DIR.glob("*.json"))
            stats["flow_states"] = len(flow_files)
            for file_path in flow_files:
                total_size += file_path.stat().st_size
        
        # Count user profiles
        if USER_PROFILES_DIR.exists():
            profile_files = list(USER_PROFILES_DIR.glob("*.json"))
            stats["user_profiles"] = len(profile_files)
            for file_path in profile_files:
                total_size += file_path.stat().st_size
        
        # Count debug logs
        if DEBUG_LOGS_DIR.exists():
            log_files = list(DEBUG_LOGS_DIR.glob("*.json"))
            stats["debug_logs"] = len(log_files)
            for file_path in log_files:
                total_size += file_path.stat().st_size
        
        stats["total_size_mb"] = round(total_size / (1024 * 1024), 2)
        
        return {
            "status": "success",
            "stats": stats,
            "timestamp": time.time()
        }
    except Exception as e:
        logger.error(f"Failed to get persistence stats: {e}")
        return {
            "status": "error",
            "message": f"Failed to get stats: {str(e)}",
            "timestamp": time.time()
        }

# --------------------------------------------------------------------
# Run
# --------------------------------------------------------------------
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000,log_level="warning",access_log=False)
