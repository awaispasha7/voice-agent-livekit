import os
import json
import logging
import re
import time
import uuid
import hashlib
import threading
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional, Dict, Any, List

import httpx
import openai
import uvicorn
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from livekit import api, rtc
from livekit.api import room_service
# from livekit.api import enums

# from livekit.api import room_models
from livekit.rtc import DataPacketKind
# from livekit.api.room_models import DataPacketKind as APIDataPacketKind
from pydantic import BaseModel, Field

# Import centralized LLM utilities
from backend.llm_utils import (
    analyze_transcription_quality,
    extract_answer_with_llm,
    detect_intent_with_llm,
    match_answer_with_llm,
    detect_uncertainty_with_llm
)

# Import orchestrator for intelligent conversation management
from backend.conversational_orchestrator import (
    ConversationalOrchestrator,
    OrchestratorAction,
    create_orchestrator_from_template
)

# =============================================================================
# CONFIGURATION AND SETUP
# =============================================================================

# Get the current file's directory
current_dir = Path(__file__).parent

# =============================================================================
# VOICE CACHING UTILITIES
# =============================================================================
VOICE_CACHE_FILE = current_dir / "cached_voices.json"
DEFAULT_VOICE_ID = "7f423809-0011-4658-ba48-a411f5e516ba"  # Ashwin - Warm Narrator


def fetch_cartesia_voices():
    """Fetch all available voices from Cartesia API with pagination support"""
    try:
        api_key = os.getenv("CARTESIA_API_KEY")
        if not api_key:
            logger.warning("CARTESIA_API_KEY not found, using fallback voices")
            return {}

        headers = {
            "Authorization": f"Bearer {api_key}",
            "Cartesia-Version": "2025-04-16",
        }

        all_voices = {}
        next_page = None
        page_count = 0

        while True:
            page_count += 1
            url = "https://api.cartesia.ai/voices"
            params = {}
            if next_page:
                params["starting_after"] = next_page

            response = httpx.get(
                url, headers=headers, params=params, timeout=30)
            response.raise_for_status()
            data = response.json()

            # Extract voices from this page
            voices = data.get("data", [])
            if isinstance(voices, list):
                page_voices = {}
                for voice in voices:
                    if isinstance(voice, dict) and "id" in voice:
                        page_voices[voice["id"]] = voice.get("name", "Unknown")
                all_voices.update(page_voices)
                logger.info(
                    f"Page {page_count}: Added {
                        len(page_voices)} voices (total: {
            len(all_voices)})")
            else:
                logger.warning(
                    f"Unexpected voices format on page {page_count}: {
                        type(voices)}")

            # Check if there are more pages
            has_more = data.get("has_more", False)
            next_page = data.get("next_page")

            if not has_more or not next_page:
                break

            # Safety limit to prevent infinite loops
            if page_count >= 50:  # Max 50 pages (5000 voices)
                logger.warning(
                    "Reached maximum page limit (50), stopping pagination")
                break

        logger.info(
            f"✅ Fetched {
                len(all_voices)} voices across {page_count} pages")
        return all_voices

    except Exception as e:
        logger.error(f"Failed to fetch Cartesia voices: {e}")
        return {}


def load_cached_voices():
    """Load voices from cache file or return empty dict"""
    try:
        if VOICE_CACHE_FILE.exists():
            with open(VOICE_CACHE_FILE, 'r', encoding='utf-8') as f:
                return json.load(f)
    except Exception as e:
        logger.error(f"Failed to load cached voices: {e}")
    return {}


def save_cached_voices(voices_dict):
    """Save voices to cache file"""
    try:
        with open(VOICE_CACHE_FILE, 'w', encoding='utf-8') as f:
            json.dump(voices_dict, f, indent=2, ensure_ascii=False)
        logger.info(
            f"✅ Cached {
                len(voices_dict)} voices to {VOICE_CACHE_FILE}")
    except Exception as e:
        logger.error(f"Failed to save cached voices: {e}")


def update_voice_cache():
    """Update voice cache with latest Cartesia voices"""
    logger.info("🔄 Updating voice cache from Cartesia API...")
    voices = fetch_cartesia_voices()
    if voices:
        save_cached_voices(voices)
        logger.info(f"✅ Updated voice cache with {len(voices)} voices")
        return voices
    else:
        logger.warning("⚠️ No voices fetched, keeping existing cache")
        return load_cached_voices()


def get_available_voices():
    """Get available voices (cached or fresh)"""
    cached_voices = load_cached_voices()
    if not cached_voices:
        logger.info("No cached voices found, fetching fresh voices...")
        return update_voice_cache()
    return cached_voices


# Try different possible .env locations
env_paths = [
    current_dir / "../.env",  # Relative to backend directory
    current_dir / "../../.env",  # Relative to project root
    Path("/home/ubuntu/alive5-voice-agent/.env"),  # Absolute production path
    Path(".env"),  # Current working directory
]

env_loaded = False
for env_path in env_paths:
    if env_path.exists():
        load_dotenv(dotenv_path=str(env_path))
        # print(f"✅ Loaded .env from: {env_path}")  # Commented out to reduce
        # log clutter
        env_loaded = True
        break

if not env_loaded:
    # print("⚠️ No .env file found in any expected location")  # Commented out to reduce log clutter
    # print(f"   Searched paths: {[str(p) for p in env_paths]}")
    # Fallback to default behavior
    load_dotenv()

app = FastAPI(title="Alive5 Voice Agent Server", version="2.0")

# Configure clean logging (no systemd prefixes)
logging.basicConfig(
    level=logging.INFO,
    format='%(message)s',  # Clean format without timestamps/prefixes
    force=True  # Override any existing configuration
)

# Disable verbose FastAPI request logs
logging.getLogger("uvicorn.access").setLevel(logging.WARNING)
logger = logging.getLogger("token-server")

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Get credentials from environment
LIVEKIT_API_KEY = os.getenv("LIVEKIT_API_KEY")
LIVEKIT_API_SECRET = os.getenv("LIVEKIT_API_SECRET")
LIVEKIT_URL = os.getenv("LIVEKIT_URL")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
A5_BASE_URL = os.getenv("A5_BASE_URL")
A5_API_KEY = os.getenv("A5_API_KEY")

# Alive5 API endpoints - fully configurable
A5_TEMPLATE_URL = os.getenv(
    "A5_TEMPLATE_URL",
    "/1.0/org-botchain/generate-template")
A5_FAQ_URL = os.getenv(
    "A5_FAQ_URL",
    "/public/1.0/get-faq-bot-response-by-bot-id")
A5_BOTCHAIN_NAME = os.getenv("A5_BOTCHAIN_NAME", "dustin-gpt")
A5_ORG_NAME = os.getenv("A5_ORG_NAME", "alive5stage0")
FAQ_BOT_ID = os.getenv("FAQ_BOT_ID", "default-bot-id")

# Template polling configuration
TEMPLATE_POLLING_INTERVAL = int(
    os.getenv(
        "TEMPLATE_POLLING_INTERVAL",
        "1"))  # hours
TEMPLATE_POLLING_ENABLED = os.getenv(
    "TEMPLATE_POLLING_ENABLED",
    "true").lower() == "true"

# Environment variables loaded successfully

# Create persistence directory
PERSISTENCE_DIR = "flow_states"
if not os.path.exists(PERSISTENCE_DIR):
    os.makedirs(PERSISTENCE_DIR)
# Persistence directory ready

# Intent descriptions for LLM-based detection
INTENT_DESCRIPTIONS = {
    "sales": "Questions about pricing, plans, demos, buying, or team licenses. Examples include questions about cost, plans, packages, upgrades, or setting up a meeting with sales.",
    "support": "Technical issues, troubleshooting help, or how-to questions. Examples include problems with installation, errors, bugs, or requests for setup guides.",
    "billing": "Questions about invoices, payments, account management, or subscription changes. Examples include billing issues, refund requests, or payment method updates."
}

# Session tracking for analytics
active_sessions: Dict[str, Dict[str, Any]] = {}

# Flow management
bot_template = None
flow_states: Dict[str, Any] = {}

# Conversational Orchestrator - The intelligent brain
conversational_orchestrator: Optional[ConversationalOrchestrator] = None

# Helper: find a step in the template by its exact text (case-insensitive)


def _find_step_by_text(
    template: Dict[str, Any], target_text: str) -> Optional[Dict[str, Any]]:
    if not template or not target_text:
        return None
    tt = target_text.strip().lower()
    try:
        for flow_key, flow_data in (template.get("data", {}) or {}).items():
            # traverse next_flow chain
            stack = []
            if isinstance(flow_data, dict):
                stack.append({"flow_key": flow_key, "node": flow_data})
            while stack:
                cur = stack.pop()
                node = cur["node"]
                text = (node.get("text") or "").strip().lower()
                if text and text == tt:
                    return {"flow_key": cur["flow_key"], "node": node}
                nxt = node.get("next_flow")
                if isinstance(nxt, dict):
                    stack.append({"flow_key": cur["flow_key"], "node": nxt})
    except Exception:
        return None
    return None


# =============================================================================
# PYDANTIC MODELS FOR REQUEST/RESPONSE VALIDATION
# =============================================================================

# Request models
class ConnectionRequest(BaseModel):
    participant_name: str
    room_name: Optional[str] = None
    intent: Optional[str] = None 
    user_data: Optional[Dict[str, Any]] = None


class TranscriptRequest(BaseModel):
    room_name: str
    transcript: str
    session_id: Optional[str] = None


class SessionUpdateRequest(BaseModel):
    room_name: str
    intent: Optional[str] = None
    user_data: Optional[Dict[str, Any]] = None
    status: Optional[str] = None

# Alive5 API request models


class GenerateTemplateRequest(BaseModel):
    botchain_name: str
    org_name: str


class GetFAQResponseRequest(BaseModel):
    bot_id: str
    faq_question: str


class ProcessFlowMessageRequest(BaseModel):
    room_name: str
    user_message: str
    conversation_history: List[Dict[str, str]] = Field(default_factory=list)
    botchain_name: Optional[str] = None
    org_name: Optional[str] = None

# Flow management models
class FlowState(BaseModel):
    current_flow: Optional[str] = None
    current_step: Optional[str] = None
    flow_key: Optional[str] = None
    flow_data: Optional[Dict[str, Any]] = None
    user_responses: Optional[Dict[str, str]] = None
    conversation_history: List[Dict[str, str]] = Field(default_factory=list)
    # Pending question lock for strict flow
    pending_step: Optional[str] = None
    # 'number'|'zip'|'yesno'|'text'
    pending_expected_kind: Optional[str] = None
    pending_asked_at: Optional[float] = None
    pending_reask_count: int = 0
    deferred_intent: Optional[str] = None


class FlowResponse(BaseModel):
    room_name: str
    user_message: str
    current_flow_state: Optional[FlowState] = None
    conversation_history: Optional[List[Dict[str, str]]] = None

# Local file-based persistence functions


# =============================================================================
# FLOW STATE MANAGEMENT FUNCTIONS
# =============================================================================

def save_flow_state_to_file(room_name: str, flow_state: FlowState) -> bool:
    """Save flow state to local JSON file"""
    try:
        # Sanitize room name for filename
        safe_room_name = "".join(
            c for c in room_name if c.isalnum() or c in (
                '-', '_')).rstrip()
        file_path = os.path.join(PERSISTENCE_DIR, f"{safe_room_name}.json")
        
        # Convert FlowState to dict
        data = {
            "current_flow": flow_state.current_flow,
            "current_step": flow_state.current_step,
            "flow_data": flow_state.flow_data,
            "flow_key": flow_state.flow_key,
            "conversation_history": flow_state.conversation_history,
            "user_responses": flow_state.user_responses,
            "pending_step": flow_state.pending_step,
            "pending_expected_kind": flow_state.pending_expected_kind,
            "pending_asked_at": flow_state.pending_asked_at,
            "pending_reask_count": flow_state.pending_reask_count,
            "saved_at": time.time()
        }
        
        with open(file_path, 'w') as f:
            json.dump(data, f, indent=2)
        
        logger.info(
            f"PERSISTENCE: Saved flow state for room {room_name} to {file_path}")
        return True
        
    except Exception as e:
        logger.error(f"PERSISTENCE: Error saving flow state: {e}")
        return False


def load_flow_state_from_file(room_name: str) -> Optional[FlowState]:
    """Load flow state from local JSON file"""
    try:
        # Sanitize room name for filename
        safe_room_name = "".join(
            c for c in room_name if c.isalnum() or c in (
                '-', '_')).rstrip()
        file_path = os.path.join(PERSISTENCE_DIR, f"{safe_room_name}.json")
        
        if not os.path.exists(file_path):
            logger.info(
                f"PERSISTENCE: No flow state file found for room {room_name}")
            return None
        
        with open(file_path, 'r') as f:
            data = json.load(f)
        
        # Check if file is too old (older than 24 hours)
        saved_at = data.get("saved_at", 0)
        if time.time() - saved_at > 24 * 60 * 60:  # 24 hours
            logger.info(
                f"PERSISTENCE: Flow state for room {room_name} is too old, ignoring")
            os.remove(file_path)  # Clean up old file
            return None
        
        flow_state = FlowState(
            current_flow=data.get("current_flow"),
            current_step=data.get("current_step"),
            flow_data=data.get("flow_data"),
            flow_key=data.get("flow_key"),
            conversation_history=data.get("conversation_history", []),
            user_responses=data.get("user_responses"),
            pending_step=data.get("pending_step"),
            pending_expected_kind=data.get("pending_expected_kind"),
            pending_asked_at=data.get("pending_asked_at"),
            pending_reask_count=data.get("pending_reask_count", 0)
        )
        
        logger.info(
            f"PERSISTENCE: Loaded flow state for room {room_name} from {file_path}")
        return flow_state
        
    except Exception as e:
        logger.error(f"PERSISTENCE: Error loading flow state: {e}")
        return None


def delete_flow_state_from_file(room_name: str) -> bool:
    """Delete flow state from local JSON file"""
    try:
        # Sanitize room name for filename
        safe_room_name = "".join(
            c for c in room_name if c.isalnum() or c in (
                '-', '_')).rstrip()
        file_path = os.path.join(PERSISTENCE_DIR, f"{safe_room_name}.json")
        
        if os.path.exists(file_path):
            os.remove(file_path)
            logger.info(
                f"PERSISTENCE: Deleted flow state file for room {room_name}")
        else:
            logger.info(
                f"PERSISTENCE: No flow state file found to delete for room {room_name}")
        
        return True
        
    except Exception as e:
        logger.error(f"PERSISTENCE: Error deleting flow state: {e}")
        return False


def cleanup_old_flow_states():
    """Clean up flow state files older than 24 hours"""
    try:
        if not os.path.exists(PERSISTENCE_DIR):
            return
        
        current_time = time.time()
        cleaned_count = 0
        
        for filename in os.listdir(PERSISTENCE_DIR):
            if filename.endswith('.json'):
                file_path = os.path.join(PERSISTENCE_DIR, filename)
                try:
                    with open(file_path, 'r') as f:
                        data = json.load(f)
                    
                    saved_at = data.get("saved_at", 0)
                    if current_time - saved_at > 24 * 60 * 60:  # 24 hours
                        os.remove(file_path)
                        cleaned_count += 1
                        logger.info(
                            f"PERSISTENCE: Cleaned up old flow state file: {filename}")
                        
                except Exception as e:
                    logger.warning(
                        f"PERSISTENCE: Error processing file {filename}: {e}")
        
        if cleaned_count > 0:
            logger.info(
                f"PERSISTENCE: Cleaned up {cleaned_count} old flow state files")
        
    except Exception as e:
        logger.error(f"PERSISTENCE: Error during cleanup: {e}")


# Clean up old files on startup
cleanup_old_flow_states()


async def is_ambiguous_transcription(user_text: str) -> bool:
    """Use LLM to detect if the user text appears to be a garbled/ambiguous transcription."""
    u = (user_text or "").strip()

    # Skip empty or very short inputs
    if not u or len(u) < 2:
            return True
    
    # For very obvious cases, use simple heuristics to avoid unnecessary LLM calls
    if len(u) < 3:
            return True
    
    try:
        # Use centralized LLM utility for transcription quality analysis
        result = await analyze_transcription_quality(u)
        
        is_complete = result.get("is_complete", False)
        confidence = result.get("confidence", 0.0)
        
        # Return True if it's incomplete or low confidence
        if not is_complete or confidence < 0.5:
            return True
        
        return False

    except Exception as e:
        logger.error(f"🧠 TRANSCRIPTION ANALYSIS: Error calling LLM: {e}")
        # Fallback to simple heuristics if LLM call fails
        return len(u.split()) <= 2 and not re.search(r"\b(yes|no|ok|okay|thanks|bye|hello|hi)\b", u.lower())


# Removed interpret_answer function - now using enhanced LLM-only approach


# Removed gated_llm_extract_answer - using extract_answer_with_llm directly for simplicity


# Removed llm_extract_answer wrapper - using extract_answer_with_llm directly for simplicity


# =============================================================================
# UTILITY FUNCTIONS (Smart Processor functionality now integrated into Orchestrator)
# =============================================================================


async def detect_flow_intent_with_llm(
        user_message: str) -> Optional[Dict[str, Any]]:
    """Detect flow intent using LLM - simple and direct approach"""
    try:
        print(f"🔍 INTENT_DETECTION: Starting detection for: '{user_message}'")
        if not bot_template or not bot_template.get("data"):
            logger.warning("INTENT_DETECTION: No bot template available")
            print(f"❌ INTENT_DETECTION: No bot template available")
            return None
        
        # Extract available intents from template
        intent_mapping = {}
        
        for flow_key, flow_data in bot_template["data"].items():
            if flow_data.get("type") == "intent_bot":
                intent_name = flow_data.get("text", flow_key)  # Use text field for intent name
                if intent_name:
                    intent_mapping[intent_name] = {
                        "flow_key": flow_key,
                        "flow_data": flow_data,
                        "intent": intent_name
                    }
        
        if not intent_mapping:
            logger.warning("INTENT_DETECTION: No intents found in template")
            return None
        
        logger.info(f"INTENT_DETECTION: Analyzing message '{user_message}' for intents: {list(intent_mapping.keys())}")
        print(f"🔍 INTENT_DETECTION: Available intents: {list(intent_mapping.keys())}")
        print(f"🔍 INTENT_DETECTION: User message: '{user_message}'")

        # Use centralized LLM utility for intent detection
        result = await detect_intent_with_llm(user_message, intent_mapping)
        
        if result:
            if result.get("type") == "greeting":
                logger.info(f"INTENT_DETECTION: ✅ Greeting detected")
                return result
            else:
                intent_name = result.get("intent")
                if intent_name in intent_mapping:
                    logger.info(f"INTENT_DETECTION: ✅ Intent found: '{intent_name}'")
                    print(f"✅ INTENT MATCHED: '{intent_name}'")
                    return intent_mapping[intent_name]
        
        logger.info(f"INTENT_DETECTION: ❌ No intent found, will use FAQ bot")
        return None
            
    except Exception as e:
        logger.error(f"INTENT_DETECTION: Error using LLM: {e}")
        return None




def generate_truly_unique_room_name(
        participant_name: str = None, intent: str = None) -> str:
    """Generate a truly unique room name with intent context"""
    unique_id = str(uuid.uuid4())
    timestamp = int(time.time())
    
    # Include intent in room name for better organization
    intent_prefix = f"{intent}_" if intent else ""
    
    if participant_name:
        # Sanitize participant name (remove special characters)
        clean_name = ''.join(
            c for c in participant_name if c.isalnum()).lower()[
            :8]
        return f"alive5_{intent_prefix}{clean_name}_{timestamp}_{unique_id[:8]}"
    else:
        return f"alive5_{intent_prefix}user_{timestamp}_{unique_id[:8]}"


# =============================================================================
# API ENDPOINTS
# =============================================================================

# =============================================================================
# BASIC ENDPOINTS
# =============================================================================

@app.get("/")
def read_root():
    return {
        "message": "Alive5 Dynamic Token Server is running",
        "features": ["Intent-based routing", "Session tracking", "Dynamic agent assignment", "Alive5 API integration"],
        "version": "2.0",
        "alive5_endpoints": [
            "/api/alive5/generate-template",
            "/api/alive5/get-faq-bot-response"
        ]
    }


@app.get("/health")
def health_check():
    return {
        "status": "ok",
        "active_sessions": len(active_sessions),
        "timestamp": time.time()
    }


# =============================================================================
# CONNECTION AND SESSION MANAGEMENT ENDPOINTS
# =============================================================================

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
        room_name = generate_truly_unique_room_name(participant_name)
        
        logger.info(
            f"Generating token for {participant_name} in room {room_name}")
        
        # Create token with appropriate permissions
        token = api.AccessToken(LIVEKIT_API_KEY, LIVEKIT_API_SECRET)
        token.with_identity(participant_name)
        token.with_grants(api.VideoGrants(
            room_join=True,
            room=room_name,
            can_publish=True,
            can_subscribe=True,
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
            "user_data": {}
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
        room_name = generate_truly_unique_room_name(participant_name, intent)
        
        logger.info(
            f"Generating token for {participant_name} in room {room_name} with intent: {intent}")
        
        # Create token with extended permissions for dynamic features
        token = api.AccessToken(LIVEKIT_API_KEY, LIVEKIT_API_SECRET)
        token.with_identity(participant_name)
        token.with_grants(api.VideoGrants(
            room_join=True,
            room=room_name,
            can_publish=True,
            can_subscribe=True,
            can_publish_data=True,  # Allow data publishing for intent updates
        ))
        # Extended for complex conversations
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


@app.post("/api/sessions/update")
def update_session(request: SessionUpdateRequest):
    """Update session with detected intent and user data"""
    try:
        room_name = request.room_name
        
        # Auto-create session if it doesn't exist
        if room_name not in active_sessions:
            logger.warning(
                f"Session {room_name} not found, creating new session")
            active_sessions[room_name] = {
                "room_name": room_name,
                "participant_name": "Unknown",
                "created_at": time.time(),
                "last_updated": time.time(),
                "user_data": {},
                "status": "active",
                "intent": None
            }
        
        session = active_sessions[room_name]
        
        # Update session data
        if request.intent:
            session["intent"] = request.intent
            session["intent_detected_at"] = time.time()
            logger.info(
                f"Session {room_name}: Intent updated to {
                    request.intent}")
        
        if request.user_data:
            session["user_data"].update(request.user_data)
            if "selected_voice" in request.user_data:
                session["selected_voice"] = request.user_data["selected_voice"]
                session["voice_id"] = request.user_data["selected_voice"]
                logger.info(
                    f"Session {room_name}: Selected voice updated to {
                        request.user_data['selected_voice']}")
            logger.info(f"Session {room_name}: User data updated")
        
        if request.status:
            session["status"] = request.status
            session["status_updated_at"] = time.time()
        
        session["last_updated"] = time.time()
        
        return {
            "message": "Session updated successfully",
            "session_id": room_name,
            "current_intent": session.get("intent"),
            "status": session.get("status")
        }
        
    except Exception as e:
        logger.error(f"Error updating session: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/sessions/{room_name}")
def get_session_info(room_name: str):
    """Get current session information"""
    if room_name not in active_sessions:
        raise HTTPException(status_code=404, detail="Session not found")
    
    session = active_sessions[room_name]
    
    # Calculate session duration
    duration = time.time() - session["created_at"]
    
    return {
        "session_id": room_name,
        "participant_name": session["participant_name"],
        "intent": session.get("intent"),
        "status": session.get("status"),
        "duration_seconds": int(duration),
        "user_data": session.get("user_data", {}),
        "created_at": session["created_at"],
        "last_updated": session.get("last_updated"),
        "voice_id": session.get("voice_id"),
        "selected_voice": session.get("selected_voice")
    }


@app.get("/api/sessions")
def list_active_sessions():
    """List all active sessions with summary information"""
    sessions = []
    
    for room_name, session in active_sessions.items():
        duration = time.time() - session["created_at"]
        
        sessions.append({
            "session_id": room_name,
            "participant_name": session["participant_name"],
            "intent": session.get("intent"),
            "status": session.get("status"),
            "duration_seconds": int(duration),
            "has_user_data": bool(session.get("user_data"))
        })
    
    return {
        "total_sessions": len(sessions),
        "sessions": sessions
    }


@app.delete("/api/rooms/{room_name}")
def cleanup_room(room_name: str):
    """Enhanced room cleanup with session data persistence"""
    try:
        logger.info(f"Room cleanup requested for: {room_name}")
        
        # Get session data before cleanup
        session_data = active_sessions.get(room_name)
        
        if session_data:
            # Calculate final session metrics
            duration = time.time() - session_data["created_at"]
            final_summary = {
                "session_id": room_name,
                "participant_name": session_data["participant_name"],
                "intent": session_data.get("intent"),
                "duration_seconds": int(duration),
                "user_data": session_data.get("user_data", {}),
                "status": "completed",
                "completed_at": time.time()
            }
            
            # Log session summary for analytics
            logger.info(
                f"Session completed: {
                    json.dumps(
            final_summary,
            indent=2)}")
            
            # Remove from active sessions
            del active_sessions[room_name]
            
            return {
                "message": f"Room {room_name} cleaned up successfully",
                "session_summary": final_summary
            }
        else:
            return {
                "message": f"Room {room_name} cleanup requested (no session data found)"}
            
    except Exception as e:
        logger.error(f"Cleanup error for {room_name}: {e}")
        return {"error": str(e)}


@app.post("/api/sessions/{room_name}/transfer")
def initiate_transfer(room_name: str, department: str = "sales"):
    """Initiate transfer to human agent"""
    try:
        if room_name not in active_sessions:
            raise HTTPException(status_code=404, detail="Session not found")
        
        session = active_sessions[room_name]
        session["status"] = f"transferring_to_{department}"
        session["transfer_requested_at"] = time.time()
        
        logger.info(
            f"Transfer initiated for session {room_name} to {department}")
        
        return {
            "message": f"Transfer to {department} initiated",
            "session_id": room_name,
            "transfer_status": "initiated",
            "department": department
        }
        
    except Exception as e:
        logger.error(f"Transfer error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# =============================================================================
# TRANSCRIPT PROCESSING ENDPOINTS
# =============================================================================

@app.post('/api/process_transcript')
async def process_transcript(request: TranscriptRequest):
    """Process transcript and return intent detection results"""
    try:
        room_name = request.room_name
        transcript = request.transcript
        session_id = request.session_id
        
        if not room_name or not transcript:
            raise HTTPException(status_code=400,
                                detail="Missing room_name or transcript")
        
        logger.info(
            f"TRANSCRIPT_PROCESSING: Room {room_name}, Message: '{transcript}'")
        
        # Legacy intent detection removed; user data extraction now handled in process_flow_message
        detected_intent = None
        user_data = {}  # No longer extracting here - handled in main flow processing
        
        # Update session if we have one (minimal updates only)
        if room_name in active_sessions:
            session = active_sessions[room_name]
            
            # Update intent if detected
            if detected_intent and detected_intent != session.get("intent"):
                session["intent"] = detected_intent
                session["intent_detected_at"] = time.time()
                logger.info(
                    f"INTENT_UPDATE: Session {room_name} intent updated to '{detected_intent}'")
            
            # Note: User data extraction and profile updates now handled in process_flow_message
            # to avoid duplicate processing and ensure orchestrator profile consistency
            
            session["last_updated"] = time.time()
        
        # Prepare response
        response_data = {
            'status': 'processed',
            'room_name': room_name,
            'transcript': transcript
        }
        
        # (intent omitted)
        # Add user data to response if extracted
        if user_data:
            response_data['userData'] = user_data
            
        logger.info(f"TRANSCRIPT_RESPONSE: {response_data}")
        return response_data
        
    except Exception as e:
        logger.error(f"Error processing transcript: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# Alive5 API Integration Endpoints


# =============================================================================
# ALIVE5 API INTEGRATION ENDPOINTS
# =============================================================================

@app.post("/api/alive5/generate-template")
async def generate_template(request: GenerateTemplateRequest):
    """
    Generate a template using the Alive5 API
    """
    try:
        logger.info(
            f"ALIVE5_API: Generating template for {A5_BOTCHAIN_NAME} in org {A5_ORG_NAME}")
        
        async with httpx.AsyncClient() as client:
            response = await client.post(
                f"{A5_BASE_URL}{A5_TEMPLATE_URL}",
                headers={
                    "X-A5-APIKEY": A5_API_KEY,
                    "Content-Type": "application/json"
                },
                json={
                    "botchain_name": A5_BOTCHAIN_NAME,
                    "org_name": A5_ORG_NAME
                }
            )
            response.raise_for_status()
            result = response.json()
            logger.info(f"ALIVE5_API: Template generation successful")
            return result
    except httpx.HTTPError as e:
        logger.error(
            f"ALIVE5_API: HTTP error in template generation: {
                str(e)}")
        raise HTTPException(
            status_code=400,
            detail=f"Alive5 API request failed: {
                str(e)}")
    except Exception as e:
        logger.error(f"ALIVE5_API: Error in template generation: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Internal server error: {
                str(e)}")


@app.post("/api/alive5/get-faq-bot-response")
async def get_faq_bot_response(request: GetFAQResponseRequest):
    """
    Get FAQ bot response using the Alive5 API
    """
    try:
        logger.info(
            f"ALIVE5_API: Getting FAQ response for bot {
                request.bot_id} with question: {
            request.faq_question}")
        
        async with httpx.AsyncClient() as client:
            faq_endpoint = f"{A5_BASE_URL}{A5_FAQ_URL}"
            response = await client.post(
                faq_endpoint,
                headers={
                    "X-A5-APIKEY": A5_API_KEY,
                    "Content-Type": "application/json"
                },
                json={
                    "bot_id": request.bot_id,
                    "faq_question": request.faq_question
                }
            )
            response.raise_for_status()
            result = response.json()
            logger.info(f"ALIVE5_API: FAQ response retrieved successfully")
            return result
    except httpx.HTTPError as e:
        logger.error(f"ALIVE5_API: HTTP error in FAQ response: {str(e)}")
        raise HTTPException(
            status_code=400,
            detail=f"Alive5 API request failed: {
                str(e)}")
    except Exception as e:
        logger.error(f"ALIVE5_API: Error in FAQ response: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Internal server error: {
                str(e)}")

# Flow Management Functions


async def initialize_bot_template():
    """Initialize the bot template on startup using default configuration"""
    global bot_template

    try:
        logger.info("FLOW_MANAGEMENT: Initializing bot template...")

        # Load default template using environment variables
        default_botchain = os.getenv("A5_BOTCHAIN_NAME", "voice-1")
        default_org = os.getenv("A5_ORG_NAME", "alive5stage0")

        # Use the same function as custom config
        result = await initialize_bot_template_with_config(default_botchain, default_org)

        if result:
            logger.info(
                "FLOW_MANAGEMENT: Bot template initialized successfully")
            return result
        else:
            logger.error("FLOW_MANAGEMENT: Failed to initialize bot template")
            return None

    except Exception as e:
        logger.error(f"FLOW_MANAGEMENT: Error initializing bot template: {e}")
        return None

    # Removed mock template: always fetch from Alive5 API per client
    # requirement


# =============================================================================
# TEMPLATE MANAGEMENT FUNCTIONS
# =============================================================================

async def initialize_bot_template_with_config(
        botchain_name: str, org_name: str):
    """Initialize bot template with custom configuration using direct API call"""
    global bot_template, flow_states

    logger.info(
        f"🚀 INITIALIZING BOT TEMPLATE WITH CUSTOM CONFIG: {botchain_name}/{org_name}")

    try:
        # Get API credentials from environment
        a5_base_url = os.getenv("A5_BASE_URL")
        a5_template_url = os.getenv(
            "A5_TEMPLATE_URL",
            "/1.0/org-botchain/generate-template")
        a5_api_key = os.getenv("A5_API_KEY")

        if not a5_base_url or not a5_api_key:
            logger.error(
                "❌ Missing required environment variables: A5_BASE_URL or A5_API_KEY")
            return None

        # Make direct API call
        template_endpoint = f"{a5_base_url}{a5_template_url}"

        async with httpx.AsyncClient() as client:
            response = await client.post(
                template_endpoint,
                headers={
                    "X-A5-APIKEY": a5_api_key,
                    "Content-Type": "application/json"
                },
                json={
                    "botchain_name": botchain_name,
                    "org_name": org_name
                },
                timeout=30.0
            )

            if response.status_code == 200:
                template_data = response.json()
                # Check if this is a different template before clearing flow states
                global bot_template
                is_different_template = not bot_template or (
                    bot_template.get('botchain_name') != botchain_name or
                    bot_template.get('org_name') != org_name
                )

                # Add botchain and org info to template for tracking
                template_data['botchain_name'] = botchain_name
                template_data['org_name'] = org_name
                bot_template = template_data

                # 🧹 CLEAR ALL FLOW STATES only if this is a different template
                if is_different_template:
                    logger.info("🧹 CLEARING ALL FLOW STATES - Different template loaded")
                    flow_states.clear()

                    # Also clear any persisted flow state files
                    try:
                        import glob
                        flow_state_files = glob.glob("flow_states/*.json")
                        for file_path in flow_state_files:
                            try:
                                os.remove(file_path)
                                logger.info(f"🧹 CLEARED FLOW STATE FILE: {file_path}")
                            except Exception as e:
                                logger.warning(f"⚠️ Could not remove flow state file {file_path}: {e}")
                    except Exception as e:
                        logger.warning(f"⚠️ Could not clear flow state files: {e}")
                else:
                    logger.info("🧹 SKIPPING FLOW STATE CLEAR - Same template reloaded")

                logger.info("✅ CUSTOM TEMPLATE LOADED SUCCESSFULLY")
                logger.info(f"🔧 LOADED BOTCHAIN: {botchain_name}")
                logger.info(f"🔧 LOADED ORG: {org_name}")
                if is_different_template:
                    logger.info(f"🧹 CLEARED {len(flow_states)} FLOW STATES")
                else:
                    logger.info(f"🧹 PRESERVED {len(flow_states)} FLOW STATES")
                
                # Initialize the Conversational Orchestrator with the new template
                global conversational_orchestrator
                try:
                    conversational_orchestrator = create_orchestrator_from_template(bot_template)
                    logger.info("🧠 ORCHESTRATOR: Initialized successfully with template")
                except Exception as e:
                    logger.error(f"🧠 ORCHESTRATOR: Failed to initialize: {e}")
                    conversational_orchestrator = None
            
                return bot_template
            else:
                logger.error(f"❌ API ERROR: {response.status_code} - {response.text}")
                return None
            
    except Exception as e:
        logger.error(f"❌ CUSTOM TEMPLATE INITIALIZATION ERROR: {e}")
        return None

# Removed find_matching_intent - now using LLM-based detection


def clear_all_flow_states():
    """Clear all flow states and persisted files"""
    global flow_states

    logger.info("🧹 MANUALLY CLEARING ALL FLOW STATES")
    flow_states.clear()

    # Also clear any persisted flow state files
    try:
        import glob
        flow_state_files = glob.glob("flow_states/*.json")
        for file_path in flow_state_files:
            try:
                os.remove(file_path)
                logger.info(f"🧹 CLEARED FLOW STATE FILE: {file_path}")
            except Exception as e:
                logger.warning(
                    f"⚠️ Could not remove flow state file {file_path}: {e}")
    except Exception as e:
        logger.warning(f"⚠️ Could not clear flow state files: {e}")

    logger.info(
        f"🧹 CLEARED ALL FLOW STATES - {len(flow_states)} states remaining")


# =============================================================================
# CORE FLOW PROCESSING FUNCTIONS
# =============================================================================

def get_next_flow_step(current_flow_state: FlowState,
                       user_response: str = None, room_name: str = None) -> Optional[Dict[str, Any]]:
    """Get the next step in the current flow - fully dynamic"""
    logger.info(
        f"FLOW_NAVIGATION: Getting next step for flow {
            current_flow_state.current_flow}, step {
            current_flow_state.current_step}")
    logger.info(f"FLOW_NAVIGATION: User response: '{user_response}'")
    
    if not current_flow_state.current_flow or not bot_template:
        logger.info("FLOW_NAVIGATION: ❌ No current flow or bot template")
        return None
    
    flow_data = bot_template["data"].get(current_flow_state.current_flow)
    if not flow_data:
        logger.info(
            f"FLOW_NAVIGATION: ❌ Flow data not found for {
                current_flow_state.current_flow}")
        return None
    
    logger.info(
        f"FLOW_NAVIGATION: Flow data type: {
            flow_data.get('type')}, text: '{
            flow_data.get('text')}'")
    
    # If we have a user response, store it
    if user_response and current_flow_state.current_step:
        if not current_flow_state.user_responses:
            current_flow_state.user_responses = {}
        current_flow_state.user_responses[current_flow_state.current_step] = user_response
        logger.info(
            f"FLOW_NAVIGATION: Stored user response for step {
                current_flow_state.current_step}")
        
    # Navigate through the flow
    current_step_data = flow_data
    if current_flow_state.current_step:
        # Find the current step in the flow
        current_step_data = find_step_in_flow(
            flow_data, current_flow_state.current_step)
        logger.info(
            f"FLOW_NAVIGATION: Found current step data: {current_step_data}")
        
        # Update orchestrator profile with flow response (after current_step_data is defined)
        if conversational_orchestrator:
            try:
                # Extract structured data if possible
                extracted_value = None
                if current_step_data.get("type") == "question":
                    # Try to extract structured data using LLM
                    from backend.llm_utils import extract_answer_with_llm
                    try:
                        extracted_value = extract_answer_with_llm(user_response)
                    except:
                        pass  # Fall back to raw response
                
                conversational_orchestrator.update_profile_from_flow_response(
                    room_name=room_name or "unknown_room",
                    step_name=current_flow_state.current_step,
                    user_response=user_response,
                    extracted_value=extracted_value
                )
            except Exception as e:
                logger.warning(f"🧠 ORCHESTRATOR: Failed to update profile from flow response: {e}")
    
    if not current_step_data:
        logger.info("FLOW_NAVIGATION: ❌ Current step data not found")
        return None
    
    logger.info(
        f"FLOW_NAVIGATION: Current step type: {
            current_step_data.get('type')}, text: '{
            current_step_data.get('text')}'")
    logger.info(
        f"FLOW_NAVIGATION: Has answers: {
            bool(
            current_step_data.get('answers'))}")
    logger.info(
        f"FLOW_NAVIGATION: Has next_flow: {
            bool(
            current_step_data.get('next_flow'))}")
    
    # Check if current step has answers and user provided a response
    if user_response and current_step_data.get("answers"):
        logger.info(
            f"FLOW_NAVIGATION: Checking answers: {
                list(
            current_step_data['answers'].keys())}")
        # Use centralized LLM utility for answer matching
        # This replaces hardcoded matching logic with our centralized LLM approach
        # Benefits: Better accuracy, handles natural language, consistent with rest of system
        question_text = current_step_data.get("text", "")
        matched_answer_key = match_answer_with_llm(question_text, user_response, current_step_data["answers"])
        
        if matched_answer_key:
            logger.info(f"FLOW_NAVIGATION: ✅ LLM matched answer: '{matched_answer_key}' for response: '{user_response}'")
            answer_data = current_step_data["answers"][matched_answer_key]
            if answer_data.get("next_flow"):
                logger.info(
                    f"FLOW_NAVIGATION: ✅ Next flow found for answer {matched_answer_key}")
                return {
                    "type": "next_step",
                    "step_data": answer_data["next_flow"],
                    "step_name": answer_data["next_flow"].get("name"),
                    "answer_data": answer_data  # Include the answer data for message combination
                }
            else:
                logger.info(
                    f"FLOW_NAVIGATION: ✅ Answer {matched_answer_key} has no next_flow, returning answer content")
                return {
                    "type": "answer_response",
                    "step_data": answer_data,
                    "step_name": answer_data["name"]
                }
    
    # Check for next_flow
    if current_step_data.get("next_flow"):
        next_flow_name = current_step_data['next_flow'].get('name')
        logger.info(f"FLOW_NAVIGATION: ✅ Found next_flow: {next_flow_name}")
        
        # Special handling for "N/A" next_flow - this means transition to orchestrator
        if next_flow_name == "N/A" or next_flow_name is None:
            logger.info("FLOW_NAVIGATION: Next flow is 'N/A' - transitioning to orchestrator for intent detection")
            return None  # Return None to let orchestrator handle it
        
        return {
            "type": "next_step",
            "step_data": current_step_data["next_flow"],
            "step_name": next_flow_name
        }
    
    logger.info("FLOW_NAVIGATION: ❌ No next step found")
    return None


def find_step_in_flow(
        flow_data: Dict[str, Any], step_name: str) -> Optional[Dict[str, Any]]:
    """Recursively find a step in the flow"""
    if flow_data.get("name") == step_name:
        return flow_data
    
    if flow_data.get("next_flow"):
        result = find_step_in_flow(flow_data["next_flow"], step_name)
        if result:
            return result
    
    if flow_data.get("answers"):
        for answer_data in flow_data["answers"].values():
            if answer_data.get("next_flow"):
                result = find_step_in_flow(answer_data["next_flow"], step_name)
                if result:
                    return result
    
    return None


def add_agent_response_to_history(flow_state: FlowState, response_text: str):
    """Add agent response to conversation history"""
    if flow_state.conversation_history is None:
        flow_state.conversation_history = []
    
    flow_state.conversation_history.append({
        "role": "assistant",
        "content": response_text,
        "timestamp": datetime.now().isoformat()
    })
    
    # Keep only last 10 messages to avoid token limits
    if len(flow_state.conversation_history) > 10:
        flow_state.conversation_history = flow_state.conversation_history[-10:]


def print_flow_status(room_name: str, flow_state: FlowState,
                      action: str, details: str = ""):
    """Print visual flow status to console"""
    print("\n" + "=" * 80)
    print(f"🎯 FLOW TRACKING - Room: {room_name}")
    print(f"📋 Action: {action}")
    print(f"📍 Current Flow: {flow_state.current_flow or 'None'}")
    print(f"🔢 Current Step: {flow_state.current_step or 'None'}")
    if flow_state.user_responses:
        print(f"💬 User Responses: {flow_state.user_responses}")
    if details:
        print(f"📝 Details: {details}")
    print("=" * 80 + "\n")


async def process_flow_message(room_name: str, user_message: str, frontend_conversation_history:
                               List[Dict[str, str]] = None, botchain_name: str = None, org_name: str = None) -> Dict[str, Any]:
    """Process user message through the flow system - requires template to be loaded"""
    global bot_template

    # TEMPLATE LOADING IS NOW REQUIRED - No fallback to default template
    if not botchain_name:
        logger.error(
            "FLOW_MANAGEMENT: No botchain_name provided - template loading required")
        return {
            "status": "error",
            "message": "Bot configuration not provided. Please specify botchain_name.",
            "flow_result": {
                "type": "error",
                "response": "Please provide bot configuration to start the conversation."
            }
        }

    # Check if template is already loaded for this botchain
    global bot_template
    current_botchain = getattr(
        bot_template,
        'botchain_name',
        None) if bot_template else None
    current_org = getattr(
        bot_template,
        'org_name',
        None) if bot_template else None

    if (bot_template and
        current_botchain == botchain_name and
            current_org == (org_name or "alive5stage0")):
        logger.info(
            f"🔧 FLOW_MANAGEMENT: Template already loaded for {botchain_name}/{
                org_name or 'default'} - skipping reload")
    else:
        # Load template with custom configuration
        logger.info(
            f"🔧 FLOW_MANAGEMENT: Loading template with custom config - Botchain: {botchain_name}, Org: {
                org_name or 'default'}")
        try:
            template_result = await initialize_bot_template_with_config(botchain_name, org_name or "alive5stage0")
            if not template_result:
                logger.error(
                    f"FLOW_MANAGEMENT: Failed to load template for botchain: {botchain_name}")
                return {
                    "status": "error",
                    "message": f"Failed to load bot configuration for '{botchain_name}'. Please check your botchain name and try again.",
                    "flow_result": {
                        "type": "error",
                        "response": f"I couldn't find the bot configuration '{botchain_name}'. Please verify the bot name and try again."
                    }
                }
            logger.info(
                f"FLOW_MANAGEMENT: Successfully loaded template for botchain: {botchain_name}")
        except Exception as e:
            logger.error(
                f"FLOW_MANAGEMENT: Error loading template with custom config: {e}")
            return {
                "status": "error",
                "message": f"Error loading bot configuration: {str(e)}",
                "flow_result": {
                    "type": "error",
                    "response": "I'm experiencing technical difficulties loading the bot configuration. Please try again in a moment."
                }
            }

    # Verify template is loaded
    if bot_template is None:
        logger.error(
            "FLOW_MANAGEMENT: Template loading completed but bot_template is still None")
        return {
            "status": "error",
            "message": "Template loading failed - no template available",
            "flow_result": {
                "type": "error",
                "response": "I'm experiencing technical difficulties. Please try again in a moment."
            }
        }

    logger.info(
        f"FLOW_MANAGEMENT: Bot template is loaded and ready for botchain: {botchain_name}")
    print(
        f"🔧 FLOW_MANAGEMENT: Bot template is loaded and ready for botchain: {botchain_name}")

    logger.info(
        f"FLOW_MANAGEMENT: Processing message for room {room_name}: '{user_message}'")

    # Note: Greeting bot is now handled by the worker in on_enter() method
    # This ensures the greeting is sent immediately when the user joins the
    # room
    
    # Get or create flow state for this room
    if room_name not in flow_states:
        # Try to load from local file first
        flow_state = load_flow_state_from_file(room_name)
        if flow_state:
            flow_states[room_name] = flow_state
            print_flow_status(
                room_name,
                flow_state,
                "SESSION RESTORED FROM FILE",
                f"User message: '{user_message}'")
            logger.info(
                f"FLOW_MANAGEMENT: Restored flow state from file for room {room_name}")
        else:
            flow_states[room_name] = FlowState()
            print_flow_status(
                room_name,
                flow_states[room_name],
                "NEW SESSION CREATED",
                f"User message: '{user_message}'")
            logger.info(
                f"FLOW_MANAGEMENT: Created new flow state for room {room_name}")
    else:
        logger.info(
            f"FLOW_MANAGEMENT: Using existing flow state for room {room_name}")
    
    flow_state = flow_states[room_name]
    
    # Auto-save flow state to file after any changes
    def auto_save_flow_state():
        save_flow_state_to_file(room_name, flow_state)
    
    # Define escalation phrases and helper function here
    # Note: Removed "speak with" from generic escalation to allow specific
    # intents like "Speak with Affan"
    escalate_phrases = [
        "agent", "human", "representative", "connect me", "talk to", "speak to", "speak with", "someone", "person", "escalate", "transfer", "over the phone", "over the line"
    ]

    def _matches_any(phrases: list[str], text: str) -> bool:
        return any(p in text for p in phrases)

    # Initialize conversation history
    if flow_state.conversation_history is None:
        flow_state.conversation_history = []
    
    # Use frontend conversation history if provided (more complete)
    if frontend_conversation_history and len(
            frontend_conversation_history) > 0:
        flow_state.conversation_history = frontend_conversation_history.copy()
        logger.info(
            f"CONVERSATION_HISTORY: Using frontend history with {
                len(frontend_conversation_history)} messages")
    else:
        # Fallback: add current user message to existing history
        flow_state.conversation_history.append({
            "role": "user",
            "content": user_message,
            "timestamp": datetime.now().isoformat()
        })
    
    # Keep only last 10 messages to avoid token limits
    if len(flow_state.conversation_history) > 10:
        flow_state.conversation_history = flow_state.conversation_history[-10:]
    
    # 🔄 FLOW PROGRESSION: First check if user is answering a flow question
    if flow_state.current_flow and flow_state.current_step:
        logger.info(f"🔄 FLOW PROGRESSION: User is in active flow '{flow_state.current_flow}', step '{flow_state.current_step}'")
        
        # Try to progress the flow with user's response
        next_step = get_next_flow_step(flow_state, user_message, room_name)
        if next_step:
            logger.info(f"🔄 FLOW PROGRESSION: Successfully progressed flow - {next_step.get('type', 'unknown')}")
            
            # Update flow state with the next step
            if next_step.get("next_step"):
                flow_state.current_step = next_step["next_step"].get("name", flow_state.current_step)
                flow_state.flow_data = next_step["next_step"]
                auto_save_flow_state()
            
            # Add response to conversation history
            if next_step.get("response"):
                add_agent_response_to_history(flow_state, next_step["response"])
            
            # Map the response type to what the worker expects
            response_type = next_step.get("type", "flow_response")
            if response_type == "next_step":
                response_type = "flow_response"  # Worker expects "flow_response" not "next_step"
            
            return {
                "type": response_type,
                "response": next_step.get("response", ""),
                "next_step": next_step.get("next_step"),
                "flow_state": flow_state
            }
        else:
            logger.info("🔄 FLOW PROGRESSION: No flow progression found, transitioning to orchestrator for intent detection")
            # Clear the current flow state to allow fresh intent detection
            if flow_state.current_flow == "Flow_1":  # This is the greeting flow
                logger.info("🔄 FLOW PROGRESSION: Clearing greeting flow state for fresh intent detection")
                flow_state.current_flow = None
                flow_state.current_step = None
                flow_state.flow_data = None
                auto_save_flow_state()
    else:
        logger.info("🔄 FLOW PROGRESSION: No active flow, using orchestrator for routing")

    # 🧠 ORCHESTRATOR INTEGRATION: Use intelligent routing for all conversations
    if conversational_orchestrator:
        try:
            logger.info("🧠 ORCHESTRATOR: Processing message with intelligent routing")
            
            # Create user profile from flow state
            user_profile = conversational_orchestrator.get_or_create_profile(room_name)
            
            # Extract user data from message and update orchestrator profile
            from backend.llm_utils import extract_user_data_with_llm
            extracted_user_data = extract_user_data_with_llm(user_message)
            if extracted_user_data:
                logger.info(f"🧠 ORCHESTRATOR: Extracted user data: {extracted_user_data}")
                for key, value in extracted_user_data.items():
                    user_profile.add_collected_info(key, value)
                
                # Also update session data for consistency
                if room_name in active_sessions:
                    session = active_sessions[room_name]
                    if "user_data" not in session:
                        session["user_data"] = {}
                    session["user_data"].update(extracted_user_data)
                    session["last_updated"] = time.time()
                    logger.info(f"🧠 ORCHESTRATOR: Updated session data for {room_name}: {extracted_user_data}")
            
            # Get intelligent decision from orchestrator
            decision = await conversational_orchestrator.process_message(
                user_message=user_message,
                room_name=room_name,
                conversation_history=flow_state.conversation_history[-5:],  # Last 5 messages
                current_flow_state=flow_state.dict() if flow_state else None,
                current_step_data=flow_state.flow_data if flow_state else None
            )
            
            logger.info(f"🧠 ORCHESTRATOR: Decision - {decision.action} (confidence: {decision.confidence})")
            logger.info(f"🧠 ORCHESTRATOR: Reasoning - {decision.reasoning}")
            
            # Execute the orchestrator's decision
            if decision.action == OrchestratorAction.USE_FAQ:
                logger.info("🧠 ORCHESTRATOR: Routing to FAQ bot")
                return await get_faq_response(user_message, flow_state=flow_state)
                
            elif decision.action == OrchestratorAction.EXECUTE_FLOW:
                logger.info(f"🧠 ORCHESTRATOR: Executing flow - {decision.flow_to_execute}")
                # Find and execute the specified flow
                if decision.flow_to_execute in conversational_orchestrator.available_flows:
                    flow_data = conversational_orchestrator.available_flows[decision.flow_to_execute]
                    flow_state.current_flow = flow_data["key"]
                    flow_state.current_step = flow_data["data"]["name"]
                    flow_state.flow_data = flow_data["data"]
                    auto_save_flow_state()
                    
                    # Check if flow has next_flow and auto-transition
                    next_flow = flow_data["data"].get("next_flow")
                    if next_flow:
                        flow_state.current_step = next_flow.get("name")
                        flow_state.flow_data = next_flow
                        auto_save_flow_state()
                        response_text = next_flow.get("text", "")
                    else:
                        response_text = flow_data["data"].get("text", "")
                    
                    add_agent_response_to_history(flow_state, response_text)
                    
                    return {
                        "type": "flow_started",
                        "flow_name": decision.flow_to_execute,
                        "response": response_text,
                        "next_step": next_flow.get("next_flow") if next_flow else None,
                        "flow_state": flow_state
                    }
                else:
                    logger.warning(f"🧠 ORCHESTRATOR: Flow '{decision.flow_to_execute}' not found, falling back to FAQ")
                    return await get_faq_response(user_message, flow_state=flow_state)
                    
            elif decision.action == OrchestratorAction.HANDLE_CONVERSATIONALLY:
                logger.info("🧠 ORCHESTRATOR: Handling conversationally")
                
                # Generate dynamic, natural conversational response
                from backend.llm_utils import generate_conversational_response
                conversational_context = {
                    "user_message": user_message,
                    "conversation_history": flow_state.conversation_history,
                    "current_flow": flow_state.current_flow,
                    "current_step": flow_state.current_step,
                    "profile": user_profile.dict() if user_profile else {}
                }
                
                dynamic_response = await generate_conversational_response(user_message, conversational_context)
                add_agent_response_to_history(flow_state, dynamic_response)
                
                return {
                    "type": "conversational_response",
                    "response": dynamic_response,
                    "flow_state": flow_state
                }
                    
            elif decision.action == OrchestratorAction.HANDLE_REFUSAL:
                logger.info("🧠 ORCHESTRATOR: Handling refusal gracefully")
                
                # Update user profile with refused fields
                if decision.skip_fields:
                    user_profile.add_refused_fields(decision.skip_fields)
                if decision.profile_updates:
                    user_profile.update_profile(decision.profile_updates)
                conversational_orchestrator.save_user_profile(room_name, user_profile)
                
                # Generate dynamic response for refusal
                from backend.llm_utils import generate_conversational_response
                conversational_context = {
                    "user_message": user_message,
                    "conversation_history": flow_state.conversation_history,
                    "current_flow": flow_state.current_flow,
                    "current_step": flow_state.current_step,
                    "profile": user_profile.dict() if user_profile else {},
                    "refusal_context": True,
                    "skipped_fields": decision.skip_fields or []
                }
                
                dynamic_response = await generate_conversational_response(user_message, conversational_context)
                add_agent_response_to_history(flow_state, dynamic_response)
                
                return {
                    "type": "refusal_handled",
                    "response": dynamic_response,
                    "flow_state": flow_state
                }
                    
            elif decision.action == OrchestratorAction.HANDLE_UNCERTAINTY:
                logger.info("🧠 ORCHESTRATOR: Handling uncertainty")
                
                # Generate dynamic response for uncertainty
                from backend.llm_utils import generate_conversational_response
                conversational_context = {
                    "user_message": user_message,
                    "conversation_history": flow_state.conversation_history,
                    "current_flow": flow_state.current_flow,
                    "current_step": flow_state.current_step,
                    "profile": user_profile.dict() if user_profile else {},
                    "uncertainty_context": True
                }
                
                dynamic_response = await generate_conversational_response(user_message, conversational_context)
                add_agent_response_to_history(flow_state, dynamic_response)
                
                return {
                    "type": "uncertainty_handled",
                    "response": dynamic_response,
                    "flow_state": flow_state
                }
            
            # If we get here, orchestrator didn't provide a clear action, continue with legacy logic
            logger.info("🧠 ORCHESTRATOR: No clear action, continuing with legacy flow logic")
            
        except Exception as e:
            logger.error(f"🧠 ORCHESTRATOR: Error in orchestrator processing: {e}")
            logger.info("🧠 ORCHESTRATOR: Falling back to legacy flow logic")
    else:
        logger.info("🧠 ORCHESTRATOR: Not available, using legacy flow logic")
    
    # Global farewell detection to gracefully end calls regardless of step type
    um_low = (user_message or "").lower().strip()
    farewell_markers = [
        "bye", "goodbye", "that is all", "that's all", "thats all", "thanks, bye", "thank you, bye", "end call", "hang up", "we are done", "we're done", "okay, bye", "ok bye", "okay that's all", "ok that's all", "i think that's all", "that's all goodbye"
    ]
    if any(m in um_low for m in farewell_markers):
        response_text = "Thanks for calling Alive5. Have a great day! Goodbye!"
        add_agent_response_to_history(flow_state, response_text)
        logger.info(
            "FLOW_MANAGEMENT: Global farewell detected → conversation_end")
        return {
            "type": "conversation_end",
            "response": response_text,
            "flow_state": flow_state
        }

    # If no current flow, try to find matching intent using LLM
    if not flow_state.current_flow:
        # Check if we can recover flow state from conversation history
        if flow_state.conversation_history and len(
                flow_state.conversation_history) > 0:
            # Look for recent agent messages that might indicate we were in a
            # flow
            recent_agent_messages = [
                msg for msg in flow_state.conversation_history[-5:] 
                if msg.get("role") == "assistant" and msg.get("content")
            ]
            
            # Check if any recent agent message looks like a flow question
            for msg in recent_agent_messages:
                content = msg.get("content", "").lower()
                if any(keyword in content for keyword in [
                       "how many phone lines", "how many texts", "special needs", "sso", "crm"]):
                    logger.info(
                        f"FLOW_MANAGEMENT: Detected potential flow recovery from conversation history: '{content[:50]}...'")
                    # Try to recover by starting pricing flow
                    matching_intent = await detect_flow_intent_with_llm("pricing information")
                    if matching_intent:
                        logger.info(
                            f"FLOW_MANAGEMENT: Recovered flow state for pricing intent")
                        flow_state.current_flow = matching_intent["flow_key"]
                        flow_state.current_step = matching_intent["flow_data"]["name"]
                        flow_state.flow_data = matching_intent["flow_data"]
                        flow_state.flow_key = matching_intent["flow_key"]
                        break
            
            # Also check if the current user message looks like a flow response
            user_msg_lower = (user_message or "").lower()
            if any(keyword in user_msg_lower for keyword in [
                   "phone lines", "text messages", "texts", "three hundred", "two hundred", "five hundred"]):
                logger.info(
                    f"FLOW_MANAGEMENT: User message looks like flow response, attempting recovery")
                # Try to recover by starting pricing flow
                matching_intent = await detect_flow_intent_with_llm("pricing information")
                if matching_intent:
                    logger.info(
                        f"FLOW_MANAGEMENT: Recovered flow state from user message context")
                    flow_state.current_flow = matching_intent["flow_key"]
                    flow_state.current_step = matching_intent["flow_data"]["name"]
                    flow_state.flow_data = matching_intent["flow_data"]
                    # Skip intent detection and go directly to flow processing
                    logger.info(
                        f"FLOW_MANAGEMENT: Skipping intent detection, processing as flow response")
                    # Continue to flow processing below
        
        # Only run intent detection if we still don't have a flow
        matching_intent = None
        print(
            f"🔍 FLOW CHECK: current_flow = '{
                flow_state.current_flow}', current_step = '{
            flow_state.current_step}'")
        if not flow_state.current_flow:
            print_flow_status(
                room_name,
                flow_state,
                "SEARCHING FOR INTENT",
                f"Analyzing message: '{user_message}'")
            logger.info(
                f"FLOW_MANAGEMENT: Bot template available: {
                    bot_template is not None}")
            if bot_template:
                logger.info(
                    f"FLOW_MANAGEMENT: Bot template data keys: {
                        list(
            bot_template.get(
                'data',
                {}).keys())}")
                # Debug: Show all available intents
                for flow_key, flow_data in bot_template.get(
                        'data', {}).items():
                    if flow_data.get('type') == 'intent_bot':
                        logger.info(
                            f"FLOW_MANAGEMENT: Available intent '{
                                flow_data.get(
            'text', '')}' in flow {flow_key}")
            
            print(f"🔍 CALLING INTENT DETECTION for: '{user_message}'")
            matching_intent = await detect_flow_intent_with_llm(user_message)
            logger.info(
                f"FLOW_MANAGEMENT: Intent detection result: {matching_intent}")
            print(f"🔍 INTENT DETECTION: '{user_message}' -> {matching_intent}")
        else:
            print(
                f"🔍 SKIPPING INTENT DETECTION: Already in flow '{
                    flow_state.current_flow}'")
        
        if matching_intent:
            print(f"✅ INTENT FOUND: {matching_intent}")

            # Skip greeting intent detection - greeting is handled by worker
            if matching_intent.get("type") == "greeting":
                logger.info(
                    "FLOW_MANAGEMENT: Greeting intent detected, but greeting is handled by worker - skipping")
                # Don't send another greeting response, just continue with
                # normal flow
                pass
            else:
                # Handle regular intent flows
                logger.info(
                    f"FLOW_MANAGEMENT: ✅ INTENT DETECTED - {matching_intent['intent']} -> {matching_intent['flow_key']}")
                logger.info(
                    f"FLOW_MANAGEMENT: Flow data: {
                        matching_intent['flow_data']}")
            
            flow_state.current_flow = matching_intent["flow_key"]
            flow_state.current_step = matching_intent["flow_data"]["name"]
            flow_state.flow_data = matching_intent["flow_data"]
            auto_save_flow_state()  # Save after flow state changes
            
            logger.info(
                f"FLOW_MANAGEMENT: LLM started flow {flow_state.current_flow} for intent: {matching_intent['intent']}")
            logger.info(
                f"FLOW_MANAGEMENT: Current step set to: {flow_state.current_step}")
            logger.info(
                f"FLOW_MANAGEMENT: Flow data type: {matching_intent['flow_data'].get('type')}")
            logger.info(
                f"FLOW_MANAGEMENT: Has next_flow: {bool(matching_intent['flow_data'].get('next_flow'))}")

            # Check if this intent has a next_flow and automatically
            # transition to it
            next_flow = matching_intent["flow_data"].get("next_flow")
            if next_flow:
                logger.info(
                    f"FLOW_MANAGEMENT: Intent has next_flow, transitioning to: {next_flow.get('name')}")
                flow_state.current_step = next_flow.get("name")
                flow_state.flow_data = next_flow
                auto_save_flow_state()  # Save after step transition
                
                print_flow_status(room_name, flow_state, "🔄 AUTO-TRANSITION", 
                                f"From intent to: {next_flow.get('type')} - '{next_flow.get('text', '')}'")
                
                # Use the next_flow response instead of intent response
                response_text = next_flow.get("text", "")
                add_agent_response_to_history(flow_state, response_text)
                
                return {
                    "type": "flow_started",
                    "flow_name": matching_intent["intent"],
                    "response": response_text,
                    "next_step": next_flow.get("next_flow")
                }
            else:
                # No next_flow, use intent response
                logger.info(
                    f"FLOW_MANAGEMENT: No next_flow found for intent, using intent response")
                print_flow_status(room_name, flow_state, "🎉 FLOW STARTED", 
                                f"Intent: {matching_intent['intent']} | Flow: {matching_intent['flow_key']} | Response: '{matching_intent['flow_data'].get('text', '')}'")
                
                # Add agent response to conversation history
                response_text = matching_intent["flow_data"].get("text", "")
                if not response_text or response_text == "N/A":
                    response_text = f"I understand you want to know about {matching_intent['intent']}. How can I help you with that?"
                    logger.warning(
                        f"FLOW_MANAGEMENT: Intent response was empty or N/A, using generic fallback")
                
                # Ensure we have a valid response
                if not response_text or response_text.strip() == "":
                    response_text = f"I can help you with {matching_intent['intent']}. What would you like to know?"
                
                add_agent_response_to_history(flow_state, response_text)
                
                return {
                    "type": "flow_started",
                    "flow_name": matching_intent["intent"],
                    "response": response_text,
                    "next_step": matching_intent["flow_data"].get("next_flow")
                }
        else:
            # No matching intent found, but check for escalation before FAQ
            # fallback
            logger.info(
                "FLOW_MANAGEMENT: ❌ LLM found no matching intent, checking for escalation")
            
            # Double-check for escalation phrases that might have been missed
            um_low_fallback = (user_message or "").lower().strip()
            if _matches_any(escalate_phrases, um_low_fallback):
                response_text = "I'm connecting you with a human agent. Please hold on."
                add_agent_response_to_history(flow_state, response_text)
                auto_save_flow_state()  # Save after escalation
                logger.info(
                    "FLOW_MANAGEMENT: Escalation detected in fallback check → initiating agent handoff")
                return {
                    "type": "agent_handoff",
                    "response": response_text,
                    "flow_state": flow_state,
                    "agent_required": True,
                    "escalation_reason": "fallback_escalation"
                }
            
            # Use FAQ bot as final fallback
            print_flow_status(
                room_name,
                flow_state,
                "❌ NO INTENT FOUND",
                "Using FAQ bot fallback")
            print(f"🚨 FAQ BOT CALLED: No intent found for '{user_message}'")
            return await get_faq_response(user_message, flow_state=flow_state)
    
    # SMART MESSAGE PROCESSING: Use LLM to analyze every user message
    logger.info(
        f"🔍 FLOW STATE: {room_name} -> Flow: {
            flow_state.current_flow}, Step: {
            flow_state.current_step}")

    # 🧠 ORCHESTRATOR: Use intelligent contextual analysis for all decisions
    # (Smart Processor functionality now integrated into Orchestrator)

    # If we're already in a flow, check if this is a response to a question or
    # greeting
    if flow_state.current_flow and flow_state.current_step:
        logger.info(
            f"FLOW_MANAGEMENT: Already in flow {
                flow_state.current_flow}, step {
            flow_state.current_step}")
        logger.info(f"FLOW_MANAGEMENT: Flow data: {flow_state.flow_data}")
        
        # Check if current step is a question, greeting, or message and user
        # provided a response
        current_step_data = flow_state.flow_data
        if current_step_data and current_step_data.get(
                "type") in ["question", "greeting", "message"]:
            step_type = current_step_data.get("type")
            logger.info(
                f"FLOW_MANAGEMENT: Current step is a {step_type}, processing user response: '{user_message}'")

            # Global farewell within any step context
            um_low_q = (user_message or "").lower().strip()
            farewell_markers_q = [
                "bye", "goodbye", "that is all", "that's all", "thats all", "thanks, bye", "thank you, bye", "end call", "hang up", "we are done", "we're done", "okay, bye", "okay that's all", "ok that's all", "ok bye"
            ]
            if any(m in um_low_q for m in farewell_markers_q):
                response_text = "Thanks for calling Alive5. Have a great day! Goodbye!"
                add_agent_response_to_history(flow_state, response_text)
                logger.info(
                    "FLOW_MANAGEMENT: Farewell detected during step → conversation_end")
                return {"type": "conversation_end",
                        "response": response_text, "flow_state": flow_state}

            # Handle greeting and message steps - let orchestrator decide what to do
            if step_type in ["greeting", "message"]:
                logger.info(f"FLOW_MANAGEMENT: Processing {step_type} step response")
                print(f"🎯 GREETING FLOW: Processing user message: '{user_message}'")
                
                # For greeting/message steps, let the orchestrator decide what to do
                # instead of trying to detect intent here
                logger.info("FLOW_MANAGEMENT: Letting orchestrator handle greeting/message response")
                # Continue to orchestrator logic below - let orchestrator handle intent detection and routing

            # Handle question steps (existing logic)
            elif step_type == "question":
            # Use LLM for robust answer extraction
                interp = extract_answer_with_llm(current_step_data.get("text", ""), user_message or "")
            logger.info(f"LLM_ANSWER_EXTRACTOR: {interp}")

            # Extra yes/no fallback for special-needs/SSO style questions
            qtxt = (current_step_data.get("text") or "").lower()
            utxt = (user_message or "").lower()
            if any(k in qtxt for k in ["special needs", "sso", "salesforce", "crm integration"]) and re.search(
                    r"\b(yes|yeah|yep|yup|sure|of course|please|ok|okay|absolutely|i need|i would need)\b", utxt):
                interp = {
                    "status": "extracted",
                    "kind": "yesno",
                    "value": True,
                    "confidence": 0.95}

            # Handle uncertainty responses (user doesn't know the answer)
            # Use LLM to detect uncertainty instead of hardcoded phrases
            question_text = current_step_data.get("text", "")
            is_uncertain = detect_uncertainty_with_llm(user_message or "", question_text)
            
            if is_uncertain and current_step_data.get("next_flow"):
                # User is uncertain - acknowledge and move to next step
                logger.info("FLOW_MANAGEMENT: User expressed uncertainty, progressing to next step")
                nxt = current_step_data["next_flow"]
                old_step = flow_state.current_step
                flow_state.current_step = nxt.get("name")
                flow_state.flow_data = nxt
                step_type = nxt.get("type", "unknown")
                response_text = "That's okay. " + nxt.get("text", "")
                
                logger.info(f"FLOW_MANAGEMENT: Uncertainty handler progressed to next step {flow_state.current_step}")
                print_flow_status(
                    room_name, flow_state, "➡️ STEP TRANSITION", 
                    f"From: {old_step} → To: {flow_state.current_step} | Type: {step_type} | Response: '{response_text}'")
                
                if step_type == 'question':
                    flow_state.pending_step = flow_state.current_step
                    flow_state.pending_expected_kind = 'number' if (
                        'phone line' in response_text.lower() or 'texts' in response_text.lower()) else None
                    flow_state.pending_asked_at = time.time()
                    flow_state.pending_reask_count = 0
                else:
                    flow_state.pending_step = None
                
                add_agent_response_to_history(flow_state, response_text)
                auto_save_flow_state()
                
                return {
                    "type": step_type,
                    "response": response_text,
                    "flow_state": flow_state
                }
            
            # Handle unclear responses in main question flow
            if interp.get("status") == "unclear":
                if interp.get("kind") == "ambiguous":
                    response_text = "I didn't quite catch that. Could you please repeat your answer more clearly?"
                else:
                    response_text = "I didn't quite understand that. Could you please repeat your answer?"
                add_agent_response_to_history(flow_state, response_text)
                logger.info(
                    f"ANSWER_INTERPRETER: Handling unclear response ({
                        interp.get(
            'kind',
            'unclear')}) with clarification request")
                return {
                    "type": "message",
                    "response": response_text,
                    "flow_state": flow_state
                }

            # Process the user response and move to next step
            next_step = get_next_flow_step(flow_state, user_message, room_name)
            if next_step:
                logger.info(f"FLOW_MANAGEMENT: ✅ Next step found: {next_step}")
                
                # 🎯 MESSAGE COMBINATION: If we have answer_data, combine current answer message with next question
                response_text = next_step["step_data"].get("text", "")
                if next_step.get("answer_data"):
                    logger.info(f"FLOW_MANAGEMENT: ✅ Found answer_data, combining messages")
                    current_answer_message = next_step["answer_data"].get("text", "")
                    if current_answer_message:
                        response_text = current_answer_message + " " + response_text
                        logger.info(f"FLOW_MANAGEMENT: ✅ Combined message: '{response_text}'")
                
                old_step = flow_state.current_step
                flow_state.current_step = next_step["step_name"]
                flow_state.flow_data = next_step["step_data"]
                step_type = next_step["step_data"].get("type", "unknown")
                
                logger.info(
                    f"FLOW_MANAGEMENT: STEP TRANSITION - From: {old_step} → To: {
                        next_step['step_name']} | Type: {step_type}")
                print_flow_status(room_name, flow_state, f"➡️ STEP TRANSITION", 
                                f"From: {old_step} → To: {next_step['step_name']} | Type: {step_type} | Response: '{response_text}'")
                
                # Set pending question lock if next is a question
                if step_type == 'question':
                    flow_state.pending_step = next_step['step_name']
                    flow_state.pending_expected_kind = 'number' if (
                        'phone line' in response_text.lower() or 'texts' in response_text.lower()) else None
                    flow_state.pending_asked_at = time.time()
                    flow_state.pending_reask_count = 0
                else:
                    flow_state.pending_step = None
                add_agent_response_to_history(flow_state, response_text)
                auto_save_flow_state()
                
                return {
                    "type": step_type,
                    "response": response_text,
                    "flow_state": flow_state
                }
            else:
                # If interpreter extracted something, attempt progression even
                # if answers don't match strictly
                if interp.get("status") == "extracted" and current_step_data.get(
                        "next_flow"):
                    nxt = current_step_data["next_flow"]
                    old_step = flow_state.current_step
                    flow_state.current_step = nxt.get("name")
                    flow_state.flow_data = nxt
                    step_type = nxt.get("type", "unknown")
                    response_text = nxt.get("text", "")
                    logger.info(
                        f"FLOW_MANAGEMENT: Interpreter-based progression applied - extracted {
                            interp.get('kind')}: {
            interp.get('value')}")
                    print_flow_status(
                        room_name, flow_state, "➡️ STEP TRANSITION", f"From: {old_step} → To: {
                            flow_state.current_step} | Type: {step_type} | Response: '{response_text}'")
                    # Update pending lock
                    if step_type == 'question':
                        flow_state.pending_step = flow_state.current_step
                        flow_state.pending_expected_kind = 'number' if (
                            'phone line' in response_text.lower() or 'texts' in response_text.lower()) else None
                        flow_state.pending_asked_at = time.time()
                        flow_state.pending_reask_count = 0
                    else:
                        flow_state.pending_step = None
                    add_agent_response_to_history(flow_state, response_text)
                    return {"type": step_type, "response": response_text,
                            "flow_state": flow_state}

                # If user utterance is too short/stopwordy, re-ask the same
                # question instead of falling back
                ur = (user_message or "").lower()
                tokens = re.findall(r"\w+", ur)
                if len(tokens) <= 2:
                    response_text = current_step_data.get("text", "")
                    add_agent_response_to_history(flow_state, response_text)
                    logger.info(
                        "FLOW_MANAGEMENT: Re-asking current question due to low-information user response")
                    return {
                        "type": "question",
                        "response": response_text,
                        "flow_state": flow_state
                    }

                logger.info(
                    "FLOW_MANAGEMENT: ❌ No next step found for question response (after heuristics)")
                # Re-ask the same question instead of immediate FAQ
                response_text = current_step_data.get("text", "")
                flow_state.pending_reask_count = (
                    flow_state.pending_reask_count or 0) + 1
                flow_state.pending_asked_at = time.time()
                add_agent_response_to_history(flow_state, response_text)
                return {"type": "question", "response": response_text,
                        "flow_state": flow_state}
        else:
            logger.info(
                f"FLOW_MANAGEMENT: Current step is not a question (type: {
                    current_step_data.get('type') if current_step_data else 'None'}), checking for intent shift or answers branch")

            # Handle "N/A" or None type steps (greeting flow completion edge case)
            if not current_step_data or current_step_data.get('type') is None or current_step_data.get('name') == 'N/A':
                logger.info("FLOW_MANAGEMENT: Detected N/A or None type step - checking for intent before routing")
                
                # Check if user is expressing an intent that should trigger a flow
                matching_intent = await detect_flow_intent_with_llm(user_message)
                if matching_intent and matching_intent.get("flow_key"):
                    logger.info(f"FLOW_MANAGEMENT: Intent detected in N/A step: '{matching_intent['intent']}' -> '{matching_intent['flow_key']}' - transitioning to flow")
                    
                    # Execute the intent flow (same logic as in main intent detection)
                    flow_state.current_flow = matching_intent["flow_key"]
                    flow_state.current_step = matching_intent["flow_data"]["name"]
                    flow_state.flow_data = matching_intent["flow_data"]
                    auto_save_flow_state()
                    
                    # Check if this intent has a next_flow and automatically transition to it
                    next_flow = matching_intent["flow_data"].get("next_flow")
                    if next_flow:
                        logger.info(f"FLOW_MANAGEMENT: Intent has next_flow, transitioning to: {next_flow.get('name')}")
                        flow_state.current_step = next_flow.get("name")
                        flow_state.flow_data = next_flow
                        auto_save_flow_state()
                        
                        response_text = next_flow.get("text", "")
                        add_agent_response_to_history(flow_state, response_text)
                        
                        return {
                            "type": "flow_started",
                            "flow_name": matching_intent["intent"],
                            "response": response_text,
                            "next_step": next_flow.get("next_flow")
                        }
                    else:
                        # No next_flow, use intent response
                        response_text = matching_intent["flow_data"].get("text", "")
                        if not response_text or response_text == "N/A":
                            response_text = f"I understand you want to know about {matching_intent['intent']}. How can I help you with that?"
                        
                        add_agent_response_to_history(flow_state, response_text)
                        
                        return {
                            "type": "flow_started",
                            "flow_name": matching_intent["intent"],
                            "response": response_text,
                            "next_step": matching_intent["flow_data"].get("next_flow")
                        }
                
                # No intent detected or confirmed - route to conversational handling
                logger.info("FLOW_MANAGEMENT: No intent detected in N/A step - routing to conversational handling")
                return await get_faq_response(user_message, flow_state=flow_state)

            # If current step is a message with a next_flow of type 'faq', auto-transition so that
            # subsequent user utterances evaluate the FAQ node's answers.
            if current_step_data and current_step_data.get(
                    "type") == "message":
                nf = current_step_data.get("next_flow")
                # If next_flow is faq, auto-transition
                if isinstance(nf, dict) and nf.get("type") == "faq":
                    old = flow_state.current_step
                    flow_state.current_step = nf.get("name")
                    flow_state.flow_data = nf
                    logger.info(
                        f"FLOW_MANAGEMENT: Auto-transitioned message → faq for answers handling: {old} → {
                            flow_state.current_step}")
                    current_step_data = flow_state.flow_data
                # If there's no explicit next_flow, but the template contains a
                # faq node with the expected text, jump to it
                elif not nf and bot_template:
                    msg_text = (current_step_data.get("text") or "").strip()
                    probe = _find_step_by_text(
                        bot_template, "Feel free to ask any question!")
                    if probe and isinstance(probe.get("node"), dict) and probe["node"].get(
                            "type") == "faq":
                        old = flow_state.current_step
                        flow_state.current_step = probe["node"].get("name")
                        flow_state.flow_data = probe["node"]
                        logger.info(
                            f"FLOW_MANAGEMENT: Soft-transitioned message → faq by text match: {old} → {
                                flow_state.current_step}")
                        current_step_data = flow_state.flow_data

            # Also handle if current step IS 'faq' — emit its prompt once so
            # the user hears it
            if current_step_data and current_step_data.get("type") == "faq":
                faq_text = current_step_data.get("text", "")
                if faq_text:
                    # avoid duplicate prompt if it was the previous assistant
                    # message
                    last_msg = flow_state.conversation_history[-1]["content"] if flow_state.conversation_history else ""
                    if (last_msg or "").strip().lower(
                    ) != faq_text.strip().lower():
                        add_agent_response_to_history(flow_state, faq_text)
                        logger.info(
                            "FLOW_MANAGEMENT: Emitting FAQ prompt to user")
                        # Return the prompt so the agent actually says it;
                        # answers will be evaluated on next user turn
                        return {
                            "type": "message",
                            "response": faq_text,
                            "flow_state": flow_state
                        }

            # Handle Agent Bot - Human agent handoff
            if current_step_data and current_step_data.get("type") == "agent":
                logger.info("FLOW_MANAGEMENT: Agent bot step detected")
                print_flow_status(
                    room_name,
                    flow_state,
                    "👤 AGENT HANDOFF",
                    f"Agent: '{
                        current_step_data.get(
            'text',
            '')}'")

                # Agent bot response
                agent_response = current_step_data.get(
                    "text", "I'm connecting you with a human agent. Please hold on.")
                add_agent_response_to_history(flow_state, agent_response)
                auto_save_flow_state()

                return {
                    "type": "agent_handoff",
                    "response": agent_response,
                    "flow_state": flow_state,
                    "agent_required": True
                }

            # Handle Action Bot - Backend actions
            if current_step_data and current_step_data.get("type") == "action":
                logger.info("FLOW_MANAGEMENT: Action bot step detected")
                print_flow_status(
                    room_name,
                    flow_state,
                    "⚡ ACTION BOT",
                    f"Action: '{
                        current_step_data.get(
            'text',
            '')}'")

                # Execute action bot functionality
                action_result = await execute_action_bot(current_step_data, flow_state)
                add_agent_response_to_history(
                    flow_state, action_result["response"])
                auto_save_flow_state()

                return {
                    "type": "action_completed",
                    "response": action_result["response"],
                    "flow_state": flow_state,
                    "action_data": action_result.get("action_data")
                }

            # Handle Condition Bot - Variable-based routing
            if current_step_data and current_step_data.get(
                    "type") == "condition":
                logger.info("FLOW_MANAGEMENT: Condition bot step detected")
                print_flow_status(
                    room_name,
                    flow_state,
                    "🔀 CONDITION BOT",
                    f"Condition: '{
                        current_step_data.get(
            'text',
            '')}'")

                # Evaluate condition and route accordingly
                condition_result = await evaluate_condition_bot(current_step_data, flow_state, user_message)
                add_agent_response_to_history(
                    flow_state, condition_result["response"])
                auto_save_flow_state()

                return {
                    "type": "condition_evaluated",
                    "response": condition_result["response"],
                    "flow_state": flow_state,
                    "condition_result": condition_result.get("condition_result")
                }

            # Handle template 'answers' on FAQ/message steps (noAction /
            # moreAction)
            if current_step_data and current_step_data.get(
                    "answers") and current_step_data.get("type") in ("faq", "message"):
                answers = current_step_data.get("answers", {}) or {}
                um = (user_message or "").lower().strip()

                # Check for escalation phrases in FAQ step (escalation should
                # work everywhere)
                if _matches_any(escalate_phrases, um):
                    response_text = "I'm connecting you with a human agent. Please hold on."
                    add_agent_response_to_history(flow_state, response_text)
                    logger.info(
                        "FLOW_MANAGEMENT: Escalation detected in FAQ step → initiating agent handoff")
                    return {
                        "type": "agent_handoff",
                        "response": response_text,
                        "flow_state": flow_state,
                        "agent_required": True,
                        "escalation_reason": "faq_step_escalation"
                    }

                # Check for end phrases
                end_phrases = [
                    "thanks", "thank you", "that is all", "that's all", "thats all", "bye", "goodbye", "all good", "great, thanks", "no more"
                ]

                if _matches_any(end_phrases, um):
                    branch = "noAction"
                    node = answers.get(branch) or {}
                    response_text = node.get("text", "")
                    add_agent_response_to_history(flow_state, response_text)
                    return {
                        "type": "conversation_end" if branch == "noAction" else node.get("type", "message"),
                        "response": response_text,
                        "flow_state": flow_state
                    }

                # For FAQ steps, treat user responses as general questions and
                # use FAQ bot
                logger.info(
                    "FAQ_ANSWER_INTERPRETER: User response in FAQ step, routing to FAQ bot")
                return await get_faq_response(user_message, flow_state=flow_state)

    print_flow_status(
        room_name,
        flow_state,
        "CHECKING FOR INTENT SHIFT",
        f"Current flow: {
            flow_state.current_flow}")
    
    # 🎯 INTENT DETECTION: Check if user is expressing a new intent
    matching_intent = await detect_flow_intent_with_llm(user_message)

    # Only shift if intent has a real flow_key (ignore greetings/none)
    if matching_intent and matching_intent.get("flow_key") and matching_intent.get("type") != "greeting":
        if matching_intent["flow_key"] != flow_state.current_flow:
            old_flow = flow_state.current_flow
            logger.info(f"FLOW_MANAGEMENT: LLM detected intent shift from {flow_state.current_flow} to {matching_intent['flow_key']}")
            flow_state.current_flow = matching_intent["flow_key"]
            flow_state.user_responses = {}

        intent_node = matching_intent["flow_data"]
        flow_state.current_step = intent_node["name"]
        flow_state.flow_data = intent_node

        print_flow_status(
            room_name,
            flow_state,
            "🔄 INTENT SHIFT DETECTED",
            f"From: {old_flow} → To: {matching_intent['flow_key']} | Intent: {matching_intent['intent']}"
        )

        next_flow = intent_node.get("next_flow")
        if next_flow:
            flow_state.current_step = next_flow.get("name")
            flow_state.flow_data = next_flow

            print_flow_status(
                room_name,
                flow_state,
                "🔄 AUTO-TRANSITION",
                f"From intent to: {next_flow.get('type')} - '{next_flow.get('text', '')}'"
            )

            response_text = next_flow.get("text", "")
            add_agent_response_to_history(flow_state, response_text)
            return {
                "type": "flow_started",
                "flow_name": matching_intent["intent"],
                "response": response_text,
                "next_step": next_flow.get("next_flow")
            }
        else:
            response_text = intent_node.get("text", "") or \
                f"I understand you want to know about {matching_intent['intent']}. How can I help you with that?"
            add_agent_response_to_history(flow_state, response_text)
            return {
                "type": "flow_started",
                "flow_name": matching_intent["intent"],
                "response": response_text,
                "next_step": None
            }
    
    # If we reach here, we're in a flow but no specific handling was done
    # This should not happen with the new logic, but as a fallback
    logger.info(
        "FLOW_MANAGEMENT: Unexpected flow state, using FAQ bot as fallback")
    logger.info(
        f"FLOW_MANAGEMENT: Current flow state - flow: {flow_state.current_flow}, step: {flow_state.current_step}")
    print_flow_status(
        room_name,
        flow_state,
        "❌ UNEXPECTED STATE",
        "Using FAQ bot fallback")
    print(f"❌ FALLBACK TO FAQ: No intent found for '{user_message}'")
    return await get_faq_response(user_message, flow_state=flow_state)


# =============================================================================
# FAQ AND RESPONSE FUNCTIONS
# =============================================================================

async def get_faq_response(user_message: str, bot_id: str = None,
                           flow_state: FlowState = None) -> Dict[str, Any]:
    """Get response from FAQ bot - supports dynamic bot IDs"""
    try:
        logger.info(
            f"FAQ_RESPONSE: Called with message: '{user_message}', bot_id: {bot_id}")
        if flow_state:
            logger.info(
                f"FAQ_RESPONSE: Flow state - current_flow: {
                    flow_state.current_flow}, current_step: {
            flow_state.current_step}")
        
        # Use provided bot_id or an explicit default from env/constant
        # (template 'name' is NOT a bot_id)
        if not bot_id:
            bot_id = FAQ_BOT_ID
        
        logger.info(f"FAQ_RESPONSE: Using bot_id: {bot_id}")
        print(f"🤖 FAQ BOT CALL: Bot ID: {bot_id} | Question: '{user_message}'")
        
        # FAQ may take ~15s; set a generous timeout
        async with httpx.AsyncClient(timeout=httpx.Timeout(35.0)) as client:
            faq_endpoint = f"{A5_BASE_URL}{A5_FAQ_URL}"
            response = await client.post(
                faq_endpoint,
                headers={
                    "X-A5-APIKEY": A5_API_KEY,
                    "Content-Type": "application/json"
                },
                json={
                    "bot_id": bot_id,
                    "faq_question": user_message
                }
            )
            response.raise_for_status()
            result = response.json()
            
            # Check if result has valid data structure
            if not result or not result.get(
                    "data") or not result["data"].get("answer"):
                print(f"⚠️ FAQ BOT RESPONSE: No valid answer received from API")
                error_response = "I'm sorry, I'm having trouble processing your request. Let me connect you to a human agent."
                if flow_state:
                    add_agent_response_to_history(flow_state, error_response)
                return {
                    "type": "fallback",
                    "response": error_response,
                    "urls": [],
                    "bot_id": bot_id
                }
            
            answer = result["data"]["answer"]
            print(f"✅ FAQ BOT RESPONSE: {answer[:100]}...")
            
            # Add agent response to conversation history if flow_state is
            # provided
            if flow_state:
                add_agent_response_to_history(flow_state, answer)
            
            return {
                "type": "faq_response",
                "response": answer,
                "urls": result["data"].get("urls", []),
                "bot_id": bot_id
            }
    except Exception as e:
        logger.error(f"FLOW_MANAGEMENT: FAQ bot error: {e!r}")
        print(f"❌ FAQ BOT ERROR: {e!r}")
        # Add error response to conversation history if flow_state is provided
        error_response = "I'm sorry, I'm having trouble processing your request. Let me connect you to a human agent."
        if flow_state:
            add_agent_response_to_history(flow_state, error_response)
        
        return {
            "type": "error",
            "response": error_response
        }


async def execute_agent_bot(
        flow_state: FlowState, user_message: str, auto_save_flow_state=None) -> Dict[str, Any]:
    """Execute agent bot functionality - transfer to human agent"""
    try:
        logger.info(
            f"AGENT_BOT: Executing agent transfer for message: '{user_message}'")

        # Get the agent flow data
        agent_flow_data = flow_state.flow_data
        if not agent_flow_data:
            logger.error("AGENT_BOT: No agent flow data available")
            return {
                "type": "error",
                "response": "Sorry, I'm having trouble connecting you to an agent right now.",
                "flow_state": flow_state
            }

        # Get the agent response text
        agent_response = agent_flow_data.get(
            "text", "I'm connecting you to a human agent who can help you better.")

        # Add agent response to history
        add_agent_response_to_history(flow_state, agent_response)

        # Check if there's a next flow to transition to
        next_flow = agent_flow_data.get("next_flow")
        if next_flow:
            logger.info(
                f"AGENT_BOT: Transitioning to next flow: {
                    next_flow.get(
            'name', 'unknown')}")
            flow_state.current_step = next_flow.get("name")
            flow_state.flow_data = next_flow
            if auto_save_flow_state:
                auto_save_flow_state()

            return {
                "type": "flow_started",
                "flow_name": "agent",
                "response": agent_response,
                "next_step": next_flow.get("next_flow")
            }
        else:
            # No next flow, just return the agent response
            logger.info("AGENT_BOT: No next flow, returning agent response")
            return {
                "type": "message",
                "response": agent_response,
                "flow_state": flow_state
            }

    except Exception as e:
        logger.error(f"AGENT_BOT: Error executing agent bot: {e}")
        return {
            "type": "error",
            "response": "Sorry, I'm having trouble connecting you to an agent right now.",
            "flow_state": flow_state
        }


async def execute_action_bot(
        action_data: Dict[str, Any], flow_state: FlowState) -> Dict[str, Any]:
    """Execute action bot functionality"""
    try:
        action_type = action_data.get("action_type", "unknown")
        action_text = action_data.get("text", "Action completed.")

        logger.info(f"ACTION_BOT: Executing action type: {action_type}")

        if action_type == "webhook":
            # Execute webhook action
            webhook_url = action_data.get("webhook_url")
            if webhook_url:
                async with httpx.AsyncClient(timeout=httpx.Timeout(10.0)) as client:
                    response = await client.post(webhook_url, json={
                        "user_message": action_text,
                        "flow_state": flow_state.dict() if flow_state else None
                    })
                    response.raise_for_status()
                    result = response.json()
                    return {
                        "response": result.get("message", "Webhook executed successfully."),
                        "action_data": result
                    }

        elif action_type == "email":
            # Execute email action
            email_data = action_data.get("email_data", {})
            # Here you would integrate with your email service
            return {
                "response": "Email sent successfully.",
                "action_data": {"email_sent": True, "recipient": email_data.get("to")}
            }

        elif action_type == "url":
            # Execute URL action
            url = action_data.get("url")
            if url:
                return {
                    "response": f"Opening URL: {url}",
                    "action_data": {"url_opened": url}
                }

        # Default action response
        return {
            "response": action_text,
            "action_data": {"action_type": action_type}
        }

    except Exception as e:
        logger.error(f"ACTION_BOT: Error executing action: {e}")
        return {
            "response": "I'm sorry, there was an error executing that action. Please try again.",
            "action_data": {"error": str(e)}
        }


async def evaluate_condition_bot(
        condition_data: Dict[str, Any], flow_state: FlowState, user_message: str) -> Dict[str, Any]:
    """Evaluate condition bot and route accordingly"""
    try:
        condition_type = condition_data.get("condition_type", "variable")
        condition_text = condition_data.get("text", "Condition evaluated.")

        logger.info(
            f"CONDITION_BOT: Evaluating condition type: {condition_type}")

        if condition_type == "variable":
            # Check variable value from flow state
            variable_name = condition_data.get("variable_name")
            expected_value = condition_data.get("expected_value")

            if variable_name and flow_state.user_responses:
                actual_value = flow_state.user_responses.get(variable_name, "")
                condition_met = str(actual_value).lower() == str(
                    expected_value).lower()

                if condition_met:
                    # Condition met - follow true path
                    next_flow = condition_data.get("true_flow")
                    response = condition_data.get(
                        "true_response", "Condition met.")
                else:
                    # Condition not met - follow false path
                    next_flow = condition_data.get("false_flow")
                    response = condition_data.get(
                        "false_response", "Condition not met.")

                # Update flow state if next_flow is provided
                if next_flow:
                    flow_state.current_step = next_flow.get("name")
                    flow_state.flow_data = next_flow

                return {
                    "response": response,
                    "condition_result": {
                        "condition_met": condition_met,
                        "variable_name": variable_name,
                        "expected_value": expected_value,
                        "actual_value": actual_value
                    }
                }

        elif condition_type == "user_input":
            # Check user input against condition
            condition_pattern = condition_data.get("condition_pattern", "")
            condition_met = condition_pattern.lower() in user_message.lower()

            if condition_met:
                next_flow = condition_data.get("true_flow")
                response = condition_data.get(
                    "true_response", "Input matches condition.")
            else:
                next_flow = condition_data.get("false_flow")
                response = condition_data.get(
                    "false_response", "Input doesn't match condition.")

            if next_flow:
                flow_state.current_step = next_flow.get("name")
                flow_state.flow_data = next_flow

            return {
                "response": response,
                "condition_result": {
                    "condition_met": condition_met,
                    "pattern": condition_pattern,
                    "user_input": user_message
                }
            }

        # Default condition response
        return {
            "response": condition_text,
            "condition_result": {"condition_type": condition_type}
        }

    except Exception as e:
        logger.error(f"CONDITION_BOT: Error evaluating condition: {e}")
        return {
            "response": "I'm sorry, there was an error evaluating that condition. Please try again.",
            "condition_result": {"error": str(e)}
        }



# =============================================================================
# FLOW MANAGEMENT ENDPOINTS
# =============================================================================

@app.post("/api/process_flow_message")
async def process_flow_message_endpoint(request: ProcessFlowMessageRequest):
    """Process user message through the new flow system"""
    try:
        room_name = request.room_name
        user_message = request.user_message
        
        if not room_name or not user_message:
            raise HTTPException(status_code=400,
                                detail="Missing room_name or user_message")

        logger.info(
            f"FLOW_PROCESSING: Room {room_name}, Message: '{user_message}'")

        # Process through flow system with conversation history and custom
        # config
        flow_result = await process_flow_message(
            room_name,
            user_message,
            request.conversation_history,
            request.botchain_name,
            request.org_name
        )
        
        # Update session if we have one
        if room_name in active_sessions:
            session = active_sessions[room_name]
            session["last_updated"] = time.time()
            session["flow_state"] = flow_result.get("flow_state")
        
        # Prepare response
        response_data = {
            'status': 'processed',
            'room_name': room_name,
            'user_message': user_message,
            'flow_result': flow_result
        }
        
        logger.info(f"FLOW_RESPONSE: {response_data}")
        return response_data
        
    except Exception as e:
        logger.error(f"Error processing flow message: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# Template Management Endpoints


@app.post("/api/refresh_template")
async def refresh_template(request: dict = None):
    """Refresh the bot template from Alive5 API using direct API call"""
    try:
        # Check if custom botchain parameters are provided
        if request and request.get("botchain_name"):
            botchain_name = request.get("botchain_name")
            org_name = request.get("org_name", "alive5stage0")
            logger.info(
                f"🔄 REFRESH_TEMPLATE: Loading custom template - Botchain: {botchain_name}, Org: {org_name}")
        else:
            # Load default template using environment variables
            botchain_name = os.getenv("A5_BOTCHAIN_NAME", "voice-1")
            org_name = os.getenv("A5_ORG_NAME", "alive5stage0")
            logger.info(
                f"🔄 REFRESH_TEMPLATE: Loading default template - Botchain: {botchain_name}, Org: {org_name}")

        # Use the same function as initialization
        result = await initialize_bot_template_with_config(botchain_name, org_name)

        if result:
            return {
                "status": "success",
                "message": "Template refreshed successfully",
                "template_version": result.get("code", "unknown"),
                "template_hash": "direct_api_call",
                "last_updated": datetime.now().isoformat(),
                "botchain_name": botchain_name,
                "org_name": org_name
            }
        else:
            raise HTTPException(
                status_code=500,
                detail="Failed to refresh template")
    except Exception as e:
        logger.error(f"Template refresh error: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Template refresh failed: {
                str(e)}")


@app.get("/api/template_info")
async def get_template_info():
    """Get current template information"""
    if not bot_template:
        return {"status": "no_template", "message": "No template loaded"}
    
    # Extract flow information
    flows = []
    if bot_template.get("data"):
        for flow_key, flow_data in bot_template["data"].items():
            flows.append({
                "key": flow_key,
                "type": flow_data.get("type"),
                "text": flow_data.get("text"),
                "name": flow_data.get("name")
            })
    
    return {
        "status": "loaded",
        "template_version": bot_template.get("code", "unknown"),
        "flows": flows,
        "total_flows": len(flows)
    }


@app.get("/api/template_status")
async def get_template_status():
    """Get current template status"""
    return {
        "status": "loaded" if bot_template else "not_loaded",
        "template_available": bot_template is not None,
        "template_loaded": bot_template is not None,
        "available_intents": list(bot_template.get("data", {}).keys()) if bot_template else [],
        "template_size": len(json.dumps(bot_template)) if bot_template else 0,
        "last_updated": datetime.now().isoformat() if bot_template else None
    }


@app.post("/api/force_template_update")
async def force_template_update():
    """Manually trigger template update (POST method)"""
    try:
        # Load default template using environment variables
        default_botchain = os.getenv("A5_BOTCHAIN_NAME", "voice-1")
        default_org = os.getenv("A5_ORG_NAME", "alive5stage0")

        # Use the same function as initialization
        result = await initialize_bot_template_with_config(default_botchain, default_org)

        if result:
            return {
                "success": True,
                "message": "Template updated successfully",
                "timestamp": datetime.now().isoformat(),
                "template_hash": "direct_api_call",
                "last_updated": datetime.now().isoformat()
            }
        else:
            return {
                "success": False,
                "message": "Failed to update template",
                "timestamp": datetime.now().isoformat()
            }
            
    except Exception as e:
        logger.error(f"Force template update error: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Template update failed: {
                str(e)}")


@app.get("/api/force_template_update")
async def force_template_update_get():
    """Manually trigger template update (GET method for easy browser access)"""
    try:
        # Load default template using environment variables
        default_botchain = os.getenv("A5_BOTCHAIN_NAME", "voice-1")
        default_org = os.getenv("A5_ORG_NAME", "alive5stage0")

        # Use the same function as initialization
        result = await initialize_bot_template_with_config(default_botchain, default_org)

        if result:
            return {
                "success": True,
                "message": "Template updated successfully",
                "timestamp": datetime.now().isoformat(),
                "template_hash": "direct_api_call",
                "last_updated": datetime.now().isoformat()
            }
        else:
            return {
                "success": False,
                "message": "Failed to update template",
                "timestamp": datetime.now().isoformat()
            }
            
    except Exception as e:
        logger.error(f"Force template update error: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Template update failed: {
                str(e)}")

# Polling endpoints removed - templates loaded on-demand


@app.post("/api/validate_and_load_template")
async def validate_and_load_template(request: dict):
    """Validate and load template with proper error handling and timeout"""
    try:
        botchain_name = request.get("botchain_name")
        org_name = request.get("org_name", "alive5stage0")

        if not botchain_name:
            return {
                "status": "error",
                "message": "botchain_name is required",
                "error_type": "missing_parameter"
            }

        logger.info(
            f"🔍 TEMPLATE_VALIDATION: Validating botchain '{botchain_name}' for org '{org_name}'")

        # Load template with timeout
        import asyncio
        try:
            template_result = await asyncio.wait_for(
                initialize_bot_template_with_config(botchain_name, org_name),
                timeout=10.0  # 10 second timeout
            )
        except asyncio.TimeoutError:
            logger.error(
                f"🔍 TEMPLATE_VALIDATION: Timeout loading template for '{botchain_name}'")
            return {
                "status": "error",
                "message": f"Timeout loading bot configuration '{botchain_name}'. Please check your connection and try again.",
                "error_type": "timeout"
            }

        if not template_result:
            logger.error(
                f"🔍 TEMPLATE_VALIDATION: Failed to load template for '{botchain_name}'")
            return {
                "status": "error",
                "message": f"Bot configuration '{botchain_name}' not found. Please verify the bot name and try again.",
                "error_type": "not_found"
            }

        # Get template info
        template_data = template_result.get("data", {})
        flow_count = len(template_data)
        greeting_available = any(
            flow.get("type") == "greeting" for flow in template_data.values())

        logger.info(
            f"🔍 TEMPLATE_VALIDATION: Successfully loaded template for '{botchain_name}' - {flow_count} flows, greeting: {greeting_available}")

        return {
            "status": "success",
            "message": f"Bot configuration '{botchain_name}' loaded successfully",
            "botchain_name": botchain_name,
            "org_name": org_name,
            "flow_count": flow_count,
            "greeting_available": greeting_available,
            "template_loaded": True
        }

    except Exception as e:
        logger.error(f"🔍 TEMPLATE_VALIDATION: Error validating template: {e}")
        return {
            "status": "error",
            "message": f"Error loading bot configuration: {str(e)}",
            "error_type": "server_error"
        }


@app.get("/api/get_greeting")
async def get_greeting():
    """Get greeting from template if available - requires template to be loaded"""
    try:
        global bot_template

        # Check if template is loaded
        if not bot_template or not bot_template.get("data"):
            logger.warning(
                "🎯 GREETING API: No template loaded - greeting not available")
            return {
                "greeting_available": False,
                "greeting_text": None,
                "message": "No template loaded - please load a bot template first"
            }

        # Look for greeting bot in template
        for flow_key, flow_data in bot_template["data"].items():
            if flow_data.get("type") == "greeting":
                greeting_text = flow_data.get("text", "")
                logger.info(
                    f"🎯 GREETING API: Found greeting bot: {flow_key} - '{greeting_text}'")
                return {
                    "greeting_available": True,
                    "greeting_text": greeting_text,
                    "flow_key": flow_key
                }

        # No greeting bot found
        logger.info("🎯 GREETING API: No greeting bot found in template")
        return {
            "greeting_available": False,
            "greeting_text": None,
            "message": "No greeting bot found in template"
        }

    except Exception as e:
        logger.error(f"🎯 GREETING API: Error getting greeting: {e}")
        return {
            "greeting_available": False,
            "greeting_text": None,
            "error": str(e)
        }


@app.post("/api/initialize_greeting_flow")
async def initialize_greeting_flow(request: dict):
    """Initialize greeting bot flow in backend when worker sends greeting"""
    try:
        room_name = request.get("room_name")
        greeting_text = request.get("greeting_text")

        if not room_name or not greeting_text:
            return {
                "success": False,
                "error": "Missing room_name or greeting_text"
            }

        global bot_template

        # Find the greeting bot in template
        greeting_flow_key = None
        greeting_flow_data = None

        if bot_template and bot_template.get("data"):
            for flow_key, flow_data in bot_template["data"].items():
                if flow_data.get("type") == "greeting":
                    # Check if the greeting text matches (allowing for partial matches)
                    template_text = flow_data.get("text", "")
                    if greeting_text in template_text or template_text in greeting_text:
                        greeting_flow_key = flow_key
                        greeting_flow_data = flow_data
                        logger.info(
                            f"🎯 GREETING FLOW INIT: Found greeting bot {flow_key} with text: {template_text}")
                        break

        if not greeting_flow_key or not greeting_flow_data:
            return {
                "success": False,
                "error": f"Greeting bot not found for text: {greeting_text}"
            }

        # Initialize flow state for this room
        if room_name not in flow_states:
            # Try to load from local file first
            flow_state = load_flow_state_from_file(room_name)
            if flow_state:
                flow_states[room_name] = flow_state
            else:
                flow_states[room_name] = FlowState()

        # Get the flow state (should never be None at this point)
        flow_state = flow_states[room_name]
        if flow_state is None:
            flow_states[room_name] = FlowState()
            flow_state = flow_states[room_name]

        # Set up greeting flow
        flow_state.current_flow = greeting_flow_key
        flow_state.current_step = greeting_flow_data.get(
            "name", greeting_flow_key)
        flow_state.flow_data = greeting_flow_data
        flow_state.flow_key = greeting_flow_key

        # Add greeting to conversation history
        add_agent_response_to_history(flow_state, greeting_text)

        # Save flow state
        save_flow_state_to_file(room_name, flow_state)

        logger.info(
            f"🎯 GREETING FLOW INIT: Initialized greeting flow {greeting_flow_key} for room {room_name}")
        print(
            f"🎯 GREETING FLOW INIT: Room {room_name} -> Flow: {greeting_flow_key}, Step: {flow_state.current_step}")

        return {
            "success": True,
            "message": "Greeting flow initialized successfully",
            "flow_key": greeting_flow_key,
            "flow_data": greeting_flow_data
        }

    except Exception as e:
        logger.error(
            f"🎯 GREETING FLOW INIT: Error initializing greeting flow: {e}")
        return {
            "success": False,
            "error": str(e)
        }


@app.get("/api/flow_states")
def get_flow_states():
    """Get all current flow states for debugging"""
    states = {}
    for room_name, flow_state in flow_states.items():
        states[room_name] = {
            "current_flow": flow_state.current_flow,
            "current_step": flow_state.current_step,
            "user_responses": flow_state.user_responses,
            "flow_data": flow_state.flow_data
        }
    return {
        "active_flows": len(states),
        "flow_states": states
    }


@app.post("/api/clear_flow_states")
def clear_flow_states():
    """Clear all flow states - useful for testing and template switching"""
    try:
        clear_all_flow_states()
        return {
            "status": "success",
            "message": "All flow states cleared successfully",
            "active_flows": len(flow_states)
        }
    except Exception as e:
        logger.error(f"Error clearing flow states: {e}")
        return {
            "status": "error",
            "message": f"Failed to clear flow states: {str(e)}"
        }


def get_voice_name_from_id(voice_id: str) -> str:
    """Get human-readable voice name from voice ID - uses cached Cartesia voices"""
    # First try to get from cached Cartesia voices
    cached_voices = load_cached_voices()
    if voice_id in cached_voices:
        return cached_voices[voice_id]

    # Fallback to hardcoded mapping for backward compatibility
    fallback_voices = {
        # Keep some popular voices as fallback
        '7f423809-0011-4658-ba48-a411f5e516ba': 'Ashwin - Warm Narrator (Default)',
        'a167e0f3-df7e-4d52-a9c3-f949145efdab': 'Blake - Helpful Agent',
        'e07c00bc-4134-4eae-9ea4-1a55fb45746b': 'Brooke - Big Sister',
        'f786b574-daa5-4673-aa0c-cbe3e8534c02': 'Katie - Friendly Fixer',
        '9626c31c-bec5-4cca-baa8-f8ba9e84c8bc': 'Jacqueline - Reassuring Agent',
        '8832a0b5-47b2-4751-bb22-6a8e2149303d': 'French Narrator Lady',
        '3b554273-4299-48b9-9aaf-eefd438e3941': 'Simi - Support Specialist',
        '95d51f79-c397-46f9-b49a-23763d3eaa2d': 'Arushi - Hinglish Speaker',
    }

    return fallback_voices.get(voice_id, f'Voice ({voice_id[:8]}...)')


# =============================================================================
# VOICE MANAGEMENT ENDPOINTS
# =============================================================================

@app.post("/api/update_voice_cache")
async def update_voice_cache_endpoint():
    """Update the voice cache with latest Cartesia voices"""
    try:
        logger.info("🔄 API: Updating voice cache...")
        voices = update_voice_cache()
        return {
            "status": "success",
            "message": f"Voice cache updated with {len(voices)} voices",
            "voice_count": len(voices)
        }
    except Exception as e:
        logger.error(f"Failed to update voice cache: {e}")
        return {
            "status": "error",
            "message": f"Failed to update voice cache: {str(e)}"
        }


@app.get("/api/available_voices")
async def get_available_voices_endpoint():
    """Get list of available voices"""
    try:
        voices = get_available_voices()
        return {
            "status": "success",
            "voices": voices,
            "voice_count": len(voices)
        }
    except Exception as e:
        logger.error(f"Failed to get available voices: {e}")
        return {
            "status": "error",
            "message": f"Failed to get available voices: {str(e)}"
        }


@app.post("/api/change_voice")
async def change_voice(request: dict):
    """Change the TTS voice for a specific room"""
    try:
        room_name = request.get("room_name")
        voice_id = request.get("voice_id")

        logger.info(
            f"🎤 VOICE_CHANGE API: Received request - room: {room_name}, voice: {voice_id}")

        if not room_name or not voice_id:
            logger.error("🎤 VOICE_CHANGE API: Missing parameters")
            return {"status": "error",
                    "message": "room_name and voice_id are required"}

        # Update local session
        if room_name in active_sessions:
            active_sessions[room_name]["selected_voice"] = voice_id
            active_sessions[room_name]["voice_id"] = voice_id
            active_sessions[room_name]["last_updated"] = time.time()
            logger.info(f"🎤 VOICE UPDATED: {voice_id} for {room_name}")
        else:
            logger.warning(f"🎤 VOICE CHANGE FAILED: Room {room_name} not found")

        # Send signal to worker via LiveKit
        livekit_api = api.LiveKitAPI(
            url=os.getenv("LIVEKIT_URL"),
            api_key=os.getenv("LIVEKIT_API_KEY"),
            api_secret=os.getenv("LIVEKIT_API_SECRET"),
        )
        try:
            payload = {
                "type": "voice_change",
                "voice_id": voice_id,
                "timestamp": time.time(),
            }
            send_req = room_service.SendDataRequest(
                room=room_name,
                data=json.dumps(payload).encode("utf-8"),
                kind=DataPacketKind.KIND_RELIABLE,
                topic="lk.voice.change",
            )
            await livekit_api.room.send_data(send_req)
            logger.info(f"🎤 VOICE CHANGE SENT: {voice_id} to {room_name}")
        finally:
            await livekit_api.aclose()  # ✅ stop "Unclosed client session" warnings

        return {
            "status": "success",
            "message": f"Voice changed to {get_voice_name_from_id(voice_id)}",
            "room_name": room_name,
            "voice_id": voice_id,
            "voice_name": get_voice_name_from_id(voice_id),
        }

    except Exception as e:
        logger.error(f"Error changing voice: {e}")
        return {"status": "error",
                "message": f"Failed to change voice: {str(e)}"}


# =============================================================================
# DEBUG AND TESTING ENDPOINTS
# =============================================================================

@app.get("/api/flow_debug/{room_name}")
def get_flow_debug(room_name: str):
    """Get detailed flow debug information for a specific room"""
    if room_name not in flow_states:
        return {"error": "Room not found"}
    
    flow_state = flow_states[room_name]
    return {
        "room_name": room_name,
        "current_flow": flow_state.current_flow,
        "current_step": flow_state.current_step,
        "user_responses": flow_state.user_responses,
        "flow_data": flow_state.flow_data,
        "template_available": bot_template is not None
    }


@app.post("/api/test_intent_detection")
async def test_intent_detection(request: Dict[str, Any]):
    """Test intent detection with a sample message"""
    user_message = request.get("message", "")
    conversation_history = request.get("conversation_history", [])
    
    if not conversation_history:
        conversation_history = [{"role": "user", "content": user_message}]
    
    logger.info(
        f"TEST_INTENT: Testing intent detection for message: '{user_message}'")
    logger.info(f"TEST_INTENT: Conversation history: {conversation_history}")
    
    result = await detect_flow_intent_with_llm(user_message)
    
    return {
        "user_message": user_message,
        "conversation_history": conversation_history,
        "detected_intent": result,
        "available_intents": list(bot_template.get("data", {}).keys()) if bot_template else []
    }

# No automatic template loading on startup - templates loaded on demand


# =============================================================================
# APPLICATION LIFECYCLE EVENTS
# =============================================================================

@app.on_event("startup")
async def startup_event():
    """Startup event - no automatic template loading"""
    logger.info("🚀 Backend started - Templates will be loaded on demand")

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
