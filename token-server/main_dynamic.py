import os
from datetime import timedelta, datetime
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from livekit import api
from dotenv import load_dotenv
import random
import time
import uuid
import uvicorn
from pydantic import BaseModel, Field
from typing import Optional, Dict, Any, List
import json
import logging
import re
import openai
import httpx


# Load environment variables
load_dotenv(dotenv_path="../.env")

app = FastAPI(title="Alive5 Voice Agent Server", version="2.0")
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

print(f"Loaded credentials:")
print(f"API_KEY: {LIVEKIT_API_KEY}")
print(f"API_SECRET: {LIVEKIT_API_SECRET[:10] if LIVEKIT_API_SECRET else 'None'}...")
print(f"URL: {LIVEKIT_URL}")
print(f"OPENAI_KEY: {OPENAI_API_KEY[:10] if OPENAI_API_KEY else 'None'}...")
print(f"A5_BASE_URL: {A5_BASE_URL}")
print(f"A5_API_KEY: {A5_API_KEY}")

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

# Request models
class ConnectionRequest(BaseModel):
    participant_name: str
    room_name: Optional[str] = None
    intent: Optional[str] = None  # sales, support, billing
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

# Flow management models
class FlowState(BaseModel):
    current_flow: Optional[str] = None
    current_step: Optional[str] = None
    flow_data: Optional[Dict[str, Any]] = None
    user_responses: Optional[Dict[str, str]] = None
    conversation_history: List[Dict[str, str]] = Field(default_factory=list)
    # Pending question lock for strict flow
    pending_step: Optional[str] = None
    pending_expected_kind: Optional[str] = None  # 'number'|'zip'|'yesno'|'text'
    pending_asked_at: Optional[float] = None
    pending_reask_count: int = 0
    deferred_intent: Optional[str] = None

class FlowResponse(BaseModel):
    room_name: str
    user_message: str
    current_flow_state: Optional[FlowState] = None
    conversation_history: Optional[List[Dict[str, str]]] = None

def interpret_answer(question_text: str, user_text: str) -> Dict[str, Any]:
    """Extract structured answers from natural speech for common question types."""
    q = (question_text or "").lower()
    u = (user_text or "").lower().strip()

    # Yes/No
    if any(k in q for k in ["special needs", "sso", "salesforce", "crm integration", "yes", "no"]):
        if re.search(r"\b(yes|yeah|yup|sure|affirmative)\b", u):
            return {"status": "extracted", "kind": "yesno", "value": True, "confidence": 0.9}
        if re.search(r"\b(no|nope|nah|negative)\b", u):
            return {"status": "extracted", "kind": "yesno", "value": False, "confidence": 0.9}

    # ZIP
    if "zip" in q:
        words_map = {"zero":"0","one":"1","two":"2","three":"3","four":"4","five":"5","six":"6","seven":"7","eight":"8","nine":"9"}
        parts = re.findall(r"\d|zero|one|two|three|four|five|six|seven|eight|nine", u)
        digits = "".join(words_map.get(p, p) for p in parts)
        if len(digits) >= 5:
            return {"status": "extracted", "kind": "zip", "value": digits[:5], "confidence": 0.85}

    # Phone lines quantity
    if "phone line" in q or "lines" in q:
        # direct digits
        m = re.search(r"\b(\d{1,3})\b", u)
        if m:
            return {"status": "extracted", "kind": "number", "value": int(m.group(1)), "confidence": 0.9}
        # hyphenated or spaced tens-composite (twenty four, twenty-four)
        tens_map = {
            "twenty":20, "thirty":30, "forty":40, "fifty":50, "sixty":60, "seventy":70, "eighty":80, "ninety":90
        }
        units_map = {"one":1,"two":2,"three":3,"four":4,"five":5,"six":6,"seven":7,"eight":8,"nine":9}
        m2 = re.search(r"\b(twenty|thirty|forty|fifty|sixty|seventy|eighty|ninety)[ -]?(one|two|three|four|five|six|seven|eight|nine)?\b", u)
        if m2:
            tens = tens_map.get(m2.group(1), 0)
            unit = units_map.get(m2.group(2), 0) if m2.group(2) else 0
            val = tens + unit
            if val > 0:
                return {"status": "extracted", "kind": "number", "value": val, "confidence": 0.85}
        # basic units (fallback)
        words_to_num = {"one":1,"two":2,"three":3,"four":4,"five":5,"six":6,"seven":7,"eight":8,"nine":9,"ten":10,
                        "eleven":11,"twelve":12,"thirteen":13,"fourteen":14,"fifteen":15,"sixteen":16,"seventeen":17,
                        "eighteen":18,"nineteen":19,"twenty":20}
        for w, n in words_to_num.items():
            if re.search(rf"\b{w}\b", u):
                return {"status": "extracted", "kind": "number", "value": n, "confidence": 0.8}

    # Texts-per-month quantity
    if "texts" in q:
        m = re.search(r"\b(\d{1,5})\b", u)
        if m:
            return {"status": "extracted", "kind": "number", "value": int(m.group(1)), "confidence": 0.85}

    return {"status": "unclear", "kind": "text", "value": u, "confidence": 0.0}

def llm_extract_answer(question_text: str, user_text: str) -> Dict[str, Any]:
    """LLM-based extractor for natural responses when deterministic parsing is unclear.
    Returns the same schema as interpret_answer. Uses strict JSON output instructions.
    """
    try:
        if not OPENAI_API_KEY:
            return {"status": "unclear", "kind": "text", "value": user_text, "confidence": 0.0}
        client = openai.OpenAI(api_key=OPENAI_API_KEY)
        system = (
            "You extract structured answers from a user's natural reply. "
            "Return a JSON object only, no prose, with keys: status ('extracted'|'unclear'), "
            "kind ('number'|'zip'|'yesno'|'text'), value, confidence (0..1)."
        )
        user = (
            "Question: " + (question_text or "") + "\n"
            "User reply: " + (user_text or "") + "\n"
            "Rules:\n"
            "- If user gives a quantity like 'two phone lines' or 'twenty four', set kind='number' and value as integer.\n"
            "- If it's a ZIP like 'two five nine six three', set kind='zip' and 5-digit value.\n"
            "- If yes/no ('yes', 'no', etc.), set kind='yesno' and value true/false.\n"
            "- Otherwise set status='unclear'.\n"
            "Respond with JSON only."
        )
        resp = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            temperature=0.2,
            max_tokens=120
        )
        content = resp.choices[0].message.content or "{}"
        data = json.loads(content)
        # Basic validation
        if isinstance(data, dict) and data.get("status") in ("extracted", "unclear"):
            return data
    except Exception as e:
        logger.warning(f"ANSWER_LLM: extractor error {e}")
    return {"status": "unclear", "kind": "text", "value": user_text, "confidence": 0.0}


async def detect_flow_intent_with_llm(user_message: str) -> Optional[Dict[str, Any]]:
    """Detect flow intent using LLM - simple and direct approach"""
    try:
        if not bot_template or not bot_template.get("data"):
            logger.warning("INTENT_DETECTION: No bot template available")
            return None
        
        # Extract available intents from template
        available_intents = []
        intent_mapping = {}
        
        for flow_key, flow_data in bot_template["data"].items():
            if flow_data.get("type") == "intent_bot":
                intent_name = flow_data.get("text", "")
                if intent_name:
                    available_intents.append(intent_name)
                    intent_mapping[intent_name] = {
                        "flow_key": flow_key,
                        "flow_data": flow_data,
                        "intent": intent_name
                    }
        
        if not available_intents:
            logger.warning("INTENT_DETECTION: No intents found in template")
            return None
        
        # Simple prompt - just compare user message with available intents
        intent_list = ", ".join(available_intents)
        
        prompt = f"""
You are an intent classifier. Compare the user's message with the available intents and find the best match.

Available intents: {intent_list}

User message: "{user_message}"

Instructions:
1. Look at the user's message and compare it with each available intent
2. Find the intent that best matches what the user is asking about
3. Consider synonyms and related terms
4. Be flexible with matching - partial matches are okay

Respond with ONLY the exact intent name from the list above, or "none" if no intent matches.

Examples:
- User says "weather" or "weather today" → match with "weather" intent
- User says "pricing" or "cost" → match with "Pricing" intent  
- User says "agent" or "human" → match with "Agent" intent
"""
        
        logger.info(f"INTENT_DETECTION: Analyzing message '{user_message}' for intents: {intent_list}")

        client = openai.OpenAI(api_key=OPENAI_API_KEY)
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[{"role": "user", "content": prompt}],
            max_tokens=20,
            temperature=0.1
        )
        
        detected_intent = response.choices[0].message.content.strip()
        logger.info(f"INTENT_DETECTION: LLM response: '{detected_intent}'")
        
        # Find matching intent
        for intent_name, intent_data in intent_mapping.items():
            if detected_intent.lower() == intent_name.lower():
                logger.info(f"INTENT_DETECTION: ✅ Intent found: '{intent_name}'")
                return intent_data
        
        logger.info(f"INTENT_DETECTION: ❌ No intent found, will use FAQ bot")
        return None
            
    except Exception as e:
        logger.error(f"INTENT_DETECTION: Error using LLM: {e}")
        return None

def extract_user_data(message: str) -> Dict[str, Any]:
    """Extract user information from message"""
    extracted_data = {}
    
    # Extract email
    email_pattern = r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b'
    emails = re.findall(email_pattern, message)
    if emails:
        extracted_data['email'] = emails[0]
        
    # Extract phone numbers
    phone_pattern = r'\b\d{3}[-.]?\d{3}[-.]?\d{4}\b'
    phones = re.findall(phone_pattern, message)
    if phones:
        extracted_data['phone'] = phones[0]
        
    # Extract names (simple pattern)
    name_indicators = ['my name is', "i'm", 'this is', 'name:', 'i am']
    for indicator in name_indicators:
        if indicator in message.lower():
            parts = message.lower().split(indicator)
            if len(parts) > 1:
                potential_name = parts[1].strip().split()[0]
                if len(potential_name) > 1 and potential_name.isalpha():
                    extracted_data['name'] = potential_name.title()
            break
    
    return extracted_data

def generate_truly_unique_room_name(participant_name: str = None, intent: str = None) -> str:
    """Generate a truly unique room name with intent context"""
    unique_id = str(uuid.uuid4())
    timestamp = int(time.time())
    
    # Include intent in room name for better organization
    intent_prefix = f"{intent}_" if intent else ""
    
    if participant_name:
        # Sanitize participant name (remove special characters)
        clean_name = ''.join(c for c in participant_name if c.isalnum()).lower()[:8]
        return f"alive5_{intent_prefix}{clean_name}_{timestamp}_{unique_id[:8]}"
    else:
        return f"alive5_{intent_prefix}user_{timestamp}_{unique_id[:8]}"

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
        
        logger.info(f"Generating token for {participant_name} in room {room_name}")
        
        # Create token with appropriate permissions
        token = api.AccessToken(LIVEKIT_API_KEY, LIVEKIT_API_SECRET)
        token.with_identity(participant_name)
        token.with_grants(api.VideoGrants(
            room_join=True,
            room=room_name,
            can_publish=True,
            can_subscribe=True,
        ))
        token.with_ttl(timedelta(minutes=45))  # Extended TTL for longer conversations
        
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
        
        logger.info(f"Generating token for {participant_name} in room {room_name} with intent: {intent}")
        
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
        token.with_ttl(timedelta(minutes=45))  # Extended for complex conversations
        
        jwt_token = token.to_jwt()
        
        # Track session with intent and user data
        session_data = {
            "participant_name": participant_name,
            "room_name": room_name,
            "created_at": time.time(),
            "intent": intent,
            "status": "created",
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
            logger.warning(f"Session {room_name} not found, creating new session")
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
            logger.info(f"Session {room_name}: Intent updated to {request.intent}")
        
        if request.user_data:
            session["user_data"].update(request.user_data)
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
        "last_updated": session.get("last_updated")
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
            logger.info(f"Session completed: {json.dumps(final_summary, indent=2)}")
            
            # Remove from active sessions
            del active_sessions[room_name]
            
            return {
                "message": f"Room {room_name} cleaned up successfully",
                "session_summary": final_summary
            }
        else:
            return {"message": f"Room {room_name} cleanup requested (no session data found)"}
            
    except Exception as e:
        logger.error(f"Cleanup error for {room_name}: {e}")
        return {"error": str(e)}

@app.post("/api/sessions/{room_name}/transfer")
def initiate_transfer(room_name: str, department: str = "sales"):
    """Initiate transfer to human agent (sales, support, billing)"""
    try:
        if room_name not in active_sessions:
            raise HTTPException(status_code=404, detail="Session not found")
        
        session = active_sessions[room_name]
        session["status"] = f"transferring_to_{department}"
        session["transfer_requested_at"] = time.time()
        
        logger.info(f"Transfer initiated for session {room_name} to {department}")
        
        return {
            "message": f"Transfer to {department} initiated",
            "session_id": room_name,
            "transfer_status": "initiated",
            "department": department
        }
        
    except Exception as e:
        logger.error(f"Transfer error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post('/api/process_transcript')
async def process_transcript(request: TranscriptRequest):
    """Process transcript and return intent detection results"""
    try:
        room_name = request.room_name
        transcript = request.transcript
        session_id = request.session_id
        
        if not room_name or not transcript:
            raise HTTPException(status_code=400, detail="Missing room_name or transcript")
        
        logger.info(f"TRANSCRIPT_PROCESSING: Room {room_name}, Message: '{transcript}'")
        
        # Legacy intent detection removed; keep only user data extraction
        detected_intent = None
        user_data = extract_user_data(transcript)
        
        # Update session if we have one
        if room_name in active_sessions:
            session = active_sessions[room_name]
            
            # Update intent if detected
            if detected_intent and detected_intent != session.get("intent"):
                session["intent"] = detected_intent
                session["intent_detected_at"] = time.time()
                logger.info(f"INTENT_UPDATE: Session {room_name} intent updated to '{detected_intent}'")
            
            # Update user data
            if user_data:
                session["user_data"].update(user_data)
                logger.info(f"USER_DATA_UPDATE: Session {room_name} data updated: {user_data}")
            
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
@app.post("/api/alive5/generate-template")
async def generate_template(request: GenerateTemplateRequest):
    """
    Generate a template using the Alive5 API
    """
    try:
        logger.info(f"ALIVE5_API: Generating template for {request.botchain_name} in org {request.org_name}")
        
        async with httpx.AsyncClient() as client:
            response = await client.post(
                f"{A5_BASE_URL}/1.0/org-botchain/generate-template",
                headers={
                    "X-A5-APIKEY": A5_API_KEY,
                    "Content-Type": "application/json"
                },
                json={
                    "botchain_name": request.botchain_name,
                    "org_name": request.org_name
                }
            )
            response.raise_for_status()
            result = response.json()
            logger.info(f"ALIVE5_API: Template generation successful")
            return result
    except httpx.HTTPError as e:
        logger.error(f"ALIVE5_API: HTTP error in template generation: {str(e)}")
        raise HTTPException(status_code=400, detail=f"Alive5 API request failed: {str(e)}")
    except Exception as e:
        logger.error(f"ALIVE5_API: Error in template generation: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")

@app.post("/api/alive5/get-faq-bot-response")
async def get_faq_bot_response(request: GetFAQResponseRequest):
    """
    Get FAQ bot response using the Alive5 API
    """
    try:
        logger.info(f"ALIVE5_API: Getting FAQ response for bot {request.bot_id} with question: {request.faq_question}")
        
        async with httpx.AsyncClient() as client:
            response = await client.post(
                f"{A5_BASE_URL}/public/1.0/get-faq-bot-response-by-bot-id",
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
        raise HTTPException(status_code=400, detail=f"Alive5 API request failed: {str(e)}")
    except Exception as e:
        logger.error(f"ALIVE5_API: Error in FAQ response: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")

# Flow Management Functions
async def initialize_bot_template():
    """Initialize the bot template on startup"""
    global bot_template
    try:
        print("\n" + "="*80)
        print("🚀 INITIALIZING BOT TEMPLATE")
        print("="*80)
        logger.info("FLOW_MANAGEMENT: Initializing bot template...")
        
        print(f"🔧 TEMPLATE LOADING: Making request to {A5_BASE_URL}/1.0/org-botchain/generate-template")
        print(f"🔧 TEMPLATE LOADING: API Key: {A5_API_KEY[:10] if A5_API_KEY else 'None'}...")
        print(f"🔧 TEMPLATE LOADING: Request payload: {{'botchain_name': 'dustin-gpt', 'org_name': 'alive5stage0'}}")
        
        async with httpx.AsyncClient() as client:
            response = await client.post(
                f"{A5_BASE_URL}/1.0/org-botchain/generate-template",
                headers={
                    "X-A5-APIKEY": A5_API_KEY,
                    "Content-Type": "application/json"
                },
                json={
                    "botchain_name": "dustin-gpt",
                    "org_name": "alive5stage0"
                }
            )
            
            print(f"🔧 TEMPLATE LOADING: Response status: {response.status_code}")
            print(f"🔧 TEMPLATE LOADING: Response text: {response.text}")
            response.raise_for_status()
            result = response.json()
            bot_template = result
            
            if result.get("code") == 200 and result.get("data"):
                logger.info("FLOW_MANAGEMENT: Bot template initialized successfully")
                
                # Print template structure
                print("✅ TEMPLATE LOADED SUCCESSFULLY")
                print(f"📊 Available Flows: {len(result['data'])}")
                for flow_key, flow_data in result["data"].items():
                    flow_type = flow_data.get("type", "unknown")
                    flow_text = flow_data.get("text", "")
                    print(f"   🔹 {flow_key}: {flow_type} - '{flow_text}'")
                
                print("="*80 + "\n")
                return bot_template
            else:
                logger.error(f"FLOW_MANAGEMENT: Invalid template response: {result}")
                print(f"❌ TEMPLATE LOAD FAILED: {result}")
                return None
    except Exception as e:
        logger.error(f"FLOW_MANAGEMENT: Failed to initialize bot template: {str(e)}")
        print(f"❌ TEMPLATE INITIALIZATION ERROR: {str(e)}")
        return None

    # Removed mock template: always fetch from Alive5 API per client requirement

# Removed find_matching_intent - now using LLM-based detection

def get_next_flow_step(current_flow_state: FlowState, user_response: str = None) -> Optional[Dict[str, Any]]:
    """Get the next step in the current flow - fully dynamic"""
    logger.info(f"FLOW_NAVIGATION: Getting next step for flow {current_flow_state.current_flow}, step {current_flow_state.current_step}")
    logger.info(f"FLOW_NAVIGATION: User response: '{user_response}'")
    
    if not current_flow_state.current_flow or not bot_template:
        logger.info("FLOW_NAVIGATION: ❌ No current flow or bot template")
        return None
    
    flow_data = bot_template["data"].get(current_flow_state.current_flow)
    if not flow_data:
        logger.info(f"FLOW_NAVIGATION: ❌ Flow data not found for {current_flow_state.current_flow}")
        return None
    
    logger.info(f"FLOW_NAVIGATION: Flow data type: {flow_data.get('type')}, text: '{flow_data.get('text')}'")
    
    # If we have a user response, store it
    if user_response and current_flow_state.current_step:
        if not current_flow_state.user_responses:
            current_flow_state.user_responses = {}
        current_flow_state.user_responses[current_flow_state.current_step] = user_response
        logger.info(f"FLOW_NAVIGATION: Stored user response for step {current_flow_state.current_step}")
    
    # Navigate through the flow
    current_step_data = flow_data
    if current_flow_state.current_step:
        # Find the current step in the flow
        current_step_data = find_step_in_flow(flow_data, current_flow_state.current_step)
        logger.info(f"FLOW_NAVIGATION: Found current step data: {current_step_data}")
    
    if not current_step_data:
        logger.info("FLOW_NAVIGATION: ❌ Current step data not found")
        return None
    
    logger.info(f"FLOW_NAVIGATION: Current step type: {current_step_data.get('type')}, text: '{current_step_data.get('text')}'")
    logger.info(f"FLOW_NAVIGATION: Has answers: {bool(current_step_data.get('answers'))}")
    logger.info(f"FLOW_NAVIGATION: Has next_flow: {bool(current_step_data.get('next_flow'))}")
    
    # Check if current step has answers and user provided a response
    if user_response and current_step_data.get("answers"):
        logger.info(f"FLOW_NAVIGATION: Checking answers: {list(current_step_data['answers'].keys())}")
        # Helper: normalize numeric phrases (e.g., "two" -> 2)
        def _extract_normalized_quantity(text: str) -> Optional[int]:
            try:
                import re
                words_to_num = {
                    "one": 1, "two": 2, "three": 3, "four": 4, "five": 5,
                    "six": 6, "seven": 7, "eight": 8, "nine": 9, "ten": 10,
                    "eleven": 11, "twelve": 12, "thirteen": 13, "fourteen": 14,
                    "fifteen": 15, "sixteen": 16, "seventeen": 17, "eighteen": 18,
                    "nineteen": 19, "twenty": 20
                }
                t = text.lower()
                m = re.search(r"\b(\d{1,3})\b", t)
                if m:
                    return int(m.group(1))
                for w, v in words_to_num.items():
                    if f" {w} " in f" {t} ":
                        return v
            except Exception:
                pass
            return None

        normalized_qty = _extract_normalized_quantity(user_response)
        # Find matching answer - more flexible matching
        for answer_key, answer_data in current_step_data["answers"].items():
            logger.info(f"FLOW_NAVIGATION: Checking answer '{answer_key}' against user response '{user_response}'")
            ak = answer_key.lower()
            ur = user_response.lower()
            match = ak in ur or ur in ak
            if not match and normalized_qty is not None:
                try:
                    if "+" in ak:
                        base = int(ak.replace("+", "").strip())
                        match = normalized_qty >= base
                    elif "-" in ak:
                        low, high = ak.split("-", 1)
                        match = int(low.strip()) <= normalized_qty <= int(high.strip())
                except Exception:
                    match = False

            if match:
                logger.info(f"FLOW_NAVIGATION: ✅ Answer match found: {answer_key}")
                if answer_data.get("next_flow"):
                    logger.info(f"FLOW_NAVIGATION: ✅ Next flow found for answer {answer_key}")
                    return {
                        "type": "next_step",
                        "step_data": answer_data["next_flow"],
                        "step_name": answer_data["name"]
                    }
                else:
                    logger.info(f"FLOW_NAVIGATION: ❌ No next_flow for answer {answer_key}")
    
    # Check for next_flow
    if current_step_data.get("next_flow"):
        logger.info(f"FLOW_NAVIGATION: ✅ Found next_flow: {current_step_data['next_flow'].get('name')}")
        return {
            "type": "next_step",
            "step_data": current_step_data["next_flow"],
            "step_name": current_step_data["next_flow"].get("name")
        }
    
    logger.info("FLOW_NAVIGATION: ❌ No next step found")
    return None

def find_step_in_flow(flow_data: Dict[str, Any], step_name: str) -> Optional[Dict[str, Any]]:
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

def print_flow_status(room_name: str, flow_state: FlowState, action: str, details: str = ""):
    """Print visual flow status to console"""
    print("\n" + "="*80)
    print(f"🎯 FLOW TRACKING - Room: {room_name}")
    print(f"📋 Action: {action}")
    print(f"📍 Current Flow: {flow_state.current_flow or 'None'}")
    print(f"🔢 Current Step: {flow_state.current_step or 'None'}")
    if flow_state.user_responses:
        print(f"💬 User Responses: {flow_state.user_responses}")
    if details:
        print(f"📝 Details: {details}")
    print("="*80 + "\n")

async def process_flow_message(room_name: str, user_message: str, frontend_conversation_history: List[Dict[str, str]] = None) -> Dict[str, Any]:
    """Process user message through the flow system"""
    logger.info(f"FLOW_MANAGEMENT: Processing message for room {room_name}: '{user_message}'")
    
    # Get or create flow state for this room
    if room_name not in flow_states:
        flow_states[room_name] = FlowState()
        print_flow_status(room_name, flow_states[room_name], "NEW SESSION CREATED", f"User message: '{user_message}'")
    
    flow_state = flow_states[room_name]
    
    # Initialize conversation history
    if flow_state.conversation_history is None:
        flow_state.conversation_history = []
    
    # Use frontend conversation history if provided (more complete)
    if frontend_conversation_history and len(frontend_conversation_history) > 0:
        flow_state.conversation_history = frontend_conversation_history.copy()
        logger.info(f"CONVERSATION_HISTORY: Using frontend history with {len(frontend_conversation_history)} messages")
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
    
    # If no current flow, try to find matching intent using LLM
    if not flow_state.current_flow:
        print_flow_status(room_name, flow_state, "SEARCHING FOR INTENT", f"Analyzing message: '{user_message}'")
        logger.info(f"FLOW_MANAGEMENT: Bot template available: {bot_template is not None}")
        if bot_template:
            logger.info(f"FLOW_MANAGEMENT: Bot template data keys: {list(bot_template.get('data', {}).keys())}")
            # Debug: Show all available intents
            for flow_key, flow_data in bot_template.get('data', {}).items():
                if flow_data.get('type') == 'intent_bot':
                    logger.info(f"FLOW_MANAGEMENT: Available intent '{flow_data.get('text', '')}' in flow {flow_key}")
        
        matching_intent = await detect_flow_intent_with_llm(user_message)
        logger.info(f"FLOW_MANAGEMENT: Intent detection result: {matching_intent}")
        print(f"🔍 INTENT DETECTION: '{user_message}' -> {matching_intent}")
        
        if matching_intent:
            logger.info(f"FLOW_MANAGEMENT: ✅ INTENT DETECTED - {matching_intent['intent']} -> {matching_intent['flow_key']}")
            logger.info(f"FLOW_MANAGEMENT: Flow data: {matching_intent['flow_data']}")
            
            flow_state.current_flow = matching_intent["flow_key"]
            flow_state.current_step = matching_intent["flow_data"]["name"]
            flow_state.flow_data = matching_intent["flow_data"]
            
            logger.info(f"FLOW_MANAGEMENT: LLM started flow {flow_state.current_flow} for intent: {matching_intent['intent']}")
            logger.info(f"FLOW_MANAGEMENT: Current step set to: {flow_state.current_step}")
            logger.info(f"FLOW_MANAGEMENT: Flow data type: {matching_intent['flow_data'].get('type')}")
            logger.info(f"FLOW_MANAGEMENT: Has next_flow: {bool(matching_intent['flow_data'].get('next_flow'))}")
            
            # Check if this intent has a next_flow and automatically transition to it
            next_flow = matching_intent["flow_data"].get("next_flow")
            if next_flow:
                logger.info(f"FLOW_MANAGEMENT: Intent has next_flow, transitioning to: {next_flow.get('name')}")
                flow_state.current_step = next_flow.get("name")
                flow_state.flow_data = next_flow
                
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
                logger.info(f"FLOW_MANAGEMENT: No next_flow found for intent, using intent response")
                print_flow_status(room_name, flow_state, "🎉 FLOW STARTED", 
                                f"Intent: {matching_intent['intent']} | Flow: {matching_intent['flow_key']} | Response: '{matching_intent['flow_data'].get('text', '')}'")
                
                # Add agent response to conversation history
                response_text = matching_intent["flow_data"].get("text", "")
                if not response_text or response_text == "N/A":
                    response_text = f"I understand you want to know about {matching_intent['intent']}. How can I help you with that?"
                    logger.warning(f"FLOW_MANAGEMENT: Intent response was empty or N/A, using generic fallback")
                
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
            # No matching intent, use FAQ bot
            logger.info("FLOW_MANAGEMENT: ❌ LLM found no matching intent, using FAQ bot")
            print_flow_status(room_name, flow_state, "❌ NO INTENT FOUND", "Using FAQ bot fallback")
            print(f"🚨 FAQ BOT CALLED: No intent found for '{user_message}'")
            return await get_faq_response(user_message, flow_state=flow_state)
    
    # If we're already in a flow, check if this is a response to a question
    if flow_state.current_flow and flow_state.current_step:
        logger.info(f"FLOW_MANAGEMENT: Already in flow {flow_state.current_flow}, step {flow_state.current_step}")
        logger.info(f"FLOW_MANAGEMENT: Flow data: {flow_state.flow_data}")
        
        # Check if current step is a question and user provided a response
        current_step_data = flow_state.flow_data
        if current_step_data and current_step_data.get("type") == "question":
            logger.info(f"FLOW_MANAGEMENT: Current step is a question, processing user response: '{user_message}'")
            
            # Try interpreter first to handle natural speech
            interp = interpret_answer(current_step_data.get("text", ""), user_message or "")
            logger.info(f"ANSWER_INTERPRETER: {interp}")
            if interp.get("status") != "extracted":
                # Gated LLM extraction if unclear
                llm_interp = llm_extract_answer(current_step_data.get("text", ""), user_message or "")
                logger.info(f"ANSWER_LLM: {llm_interp}")
                # Prefer LLM only if it extracted with reasonable confidence
                if llm_interp.get("status") == "extracted" and float(llm_interp.get("confidence", 0)) >= 0.6:
                    interp = llm_interp

            # Process the user response and move to next step
            next_step = get_next_flow_step(flow_state, user_message)
            if next_step:
                logger.info(f"FLOW_MANAGEMENT: ✅ Next step found: {next_step}")
                old_step = flow_state.current_step
                flow_state.current_step = next_step["step_name"]
                flow_state.flow_data = next_step["step_data"]
                step_type = next_step["step_data"].get("type", "unknown")
                
                logger.info(f"FLOW_MANAGEMENT: STEP TRANSITION - From: {old_step} → To: {next_step['step_name']} | Type: {step_type}")
                print_flow_status(room_name, flow_state, f"➡️ STEP TRANSITION", 
                                f"From: {old_step} → To: {next_step['step_name']} | Type: {step_type} | Response: '{next_step['step_data'].get('text', '')}'")
                
                # Handle different step types
                response_text = next_step["step_data"].get("text", "")
                # Set pending question lock if next is a question
                if step_type == 'question':
                    flow_state.pending_step = next_step['step_name']
                    flow_state.pending_expected_kind = 'number' if ('phone line' in response_text.lower() or 'texts' in response_text.lower()) else None
                    flow_state.pending_asked_at = time.time()
                    flow_state.pending_reask_count = 0
                else:
                    flow_state.pending_step = None
                add_agent_response_to_history(flow_state, response_text)
                
                return {
                    "type": step_type,
                    "response": response_text,
                    "flow_state": flow_state
                }
            else:
                # If interpreter extracted something, attempt progression even if answers don't match strictly
                if interp.get("status") == "extracted" and current_step_data.get("next_flow"):
                    nxt = current_step_data["next_flow"]
                    old_step = flow_state.current_step
                    flow_state.current_step = nxt.get("name")
                    flow_state.flow_data = nxt
                    step_type = nxt.get("type", "unknown")
                    response_text = nxt.get("text", "")
                    logger.info("FLOW_MANAGEMENT: Interpreter-based progression applied")
                    print_flow_status(room_name, flow_state, "➡️ STEP TRANSITION", f"From: {old_step} → To: {flow_state.current_step} | Type: {step_type} | Response: '{response_text}'")
                    # Update pending lock
                    if step_type == 'question':
                        flow_state.pending_step = flow_state.current_step
                        flow_state.pending_expected_kind = 'number' if ('phone line' in response_text.lower() or 'texts' in response_text.lower()) else None
                        flow_state.pending_asked_at = time.time()
                        flow_state.pending_reask_count = 0
                    else:
                        flow_state.pending_step = None
                    add_agent_response_to_history(flow_state, response_text)
                    return {"type": step_type, "response": response_text, "flow_state": flow_state}

                # Heuristics: try to interpret common free-form answers to advance flow instead of falling back
                qtext = (current_step_data.get("text") or "").lower()
                ur = (user_message or "").lower()

                def _extract_digits_from_words(t: str) -> str:
                    words_map = {
                        "zero": "0", "one": "1", "two": "2", "three": "3", "four": "4",
                        "five": "5", "six": "6", "seven": "7", "eight": "8", "nine": "9"
                    }
                    import re
                    parts = re.findall(r"\d|zero|one|two|three|four|five|six|seven|eight|nine", t)
                    return "".join(words_map.get(p, p) for p in parts)

                def _extract_quantity(t: str) -> Optional[int]:
                    import re
                    m = re.search(r"\b(\d{1,3})\b", t)
                    if m:
                        return int(m.group(1))
                    words_to_num = {
                        "one": 1, "two": 2, "three": 3, "four": 4, "five": 5,
                        "six": 6, "seven": 7, "eight": 8, "nine": 9, "ten": 10,
                        "eleven": 11, "twelve": 12, "thirteen": 13, "fourteen": 14,
                        "fifteen": 15, "sixteen": 16, "seventeen": 17, "eighteen": 18,
                        "nineteen": 19, "twenty": 20
                    }
                    for w, v in words_to_num.items():
                        if f" {w} " in f" {t} ":
                            return v
                    return None

                advanced = False
                if "zip" in qtext or "zipcode" in qtext or "zip code" in qtext:
                    zip_digits = _extract_digits_from_words(ur)
                    if len(zip_digits) >= 5 and current_step_data.get("next_flow"):
                        # Proceed to next step (FAQ for weather flow)
                        nxt = current_step_data["next_flow"]
                        old_step = flow_state.current_step
                        flow_state.current_step = nxt.get("name")
                        flow_state.flow_data = nxt
                        step_type = nxt.get("type", "unknown")
                        response_text = nxt.get("text", "")
                        logger.info(f"FLOW_MANAGEMENT: Heuristic progressed ZIP question to next step {flow_state.current_step}")
                        print_flow_status(room_name, flow_state, "➡️ STEP TRANSITION", f"From: {old_step} → To: {flow_state.current_step} | Type: {step_type} | Response: '{response_text}'")
                        add_agent_response_to_history(flow_state, response_text)
                        advanced = True
                        return {
                            "type": step_type,
                            "response": response_text,
                            "flow_state": flow_state
                        }

                qty = _extract_quantity(ur)
                if not advanced and qty is not None and current_step_data.get("next_flow"):
                    # Proceed to next question in pricing regardless of specific answer bucket
                    nxt = current_step_data["next_flow"]
                    old_step = flow_state.current_step
                    flow_state.current_step = nxt.get("name")
                    flow_state.flow_data = nxt
                    step_type = nxt.get("type", "unknown")
                    response_text = nxt.get("text", "")
                    logger.info(f"FLOW_MANAGEMENT: Heuristic progressed numeric answer to next step {flow_state.current_step}")
                    print_flow_status(room_name, flow_state, "➡️ STEP TRANSITION", f"From: {old_step} → To: {flow_state.current_step} | Type: {step_type} | Response: '{response_text}'")
                    add_agent_response_to_history(flow_state, response_text)
                    return {
                        "type": step_type,
                        "response": response_text,
                        "flow_state": flow_state
                    }

                # If user utterance is too short/stopwordy, re-ask the same question instead of falling back
                import re
                tokens = re.findall(r"\w+", ur)
                if len(tokens) <= 2:
                    response_text = current_step_data.get("text", "")
                    add_agent_response_to_history(flow_state, response_text)
                    logger.info("FLOW_MANAGEMENT: Re-asking current question due to low-information user response")
                    return {
                        "type": "question",
                        "response": response_text,
                        "flow_state": flow_state
                    }

                logger.info("FLOW_MANAGEMENT: ❌ No next step found for question response (after heuristics)")
                # Re-ask the same question instead of immediate FAQ
                response_text = current_step_data.get("text", "")
                flow_state.pending_reask_count = (flow_state.pending_reask_count or 0) + 1
                flow_state.pending_asked_at = time.time()
                add_agent_response_to_history(flow_state, response_text)
                return {"type": "question", "response": response_text, "flow_state": flow_state}
        else:
            logger.info(f"FLOW_MANAGEMENT: Current step is not a question (type: {current_step_data.get('type') if current_step_data else 'None'}), checking for intent shift")
    
    # Check for intent shift even when in a flow using LLM
    print_flow_status(room_name, flow_state, "CHECKING FOR INTENT SHIFT", f"Current flow: {flow_state.current_flow}")
    matching_intent = await detect_flow_intent_with_llm(user_message)
    if matching_intent and matching_intent["flow_key"] != flow_state.current_flow:
        # User shifted to a different intent - start new flow and auto-transition to its first actionable step
        old_flow = flow_state.current_flow
        logger.info(f"FLOW_MANAGEMENT: LLM detected intent shift from {flow_state.current_flow} to {matching_intent['flow_key']}")
        flow_state.current_flow = matching_intent["flow_key"]
        flow_state.user_responses = {}  # Reset user responses for new flow

        # Set to intent node first
        intent_node = matching_intent["flow_data"]
        flow_state.current_step = intent_node["name"]
        flow_state.flow_data = intent_node

        print_flow_status(room_name, flow_state, "🔄 INTENT SHIFT DETECTED", 
                        f"From: {old_flow} → To: {matching_intent['flow_key']} | Intent: {matching_intent['intent']}")

        # If the intent has a next_flow (e.g., a question), auto-transition to it (same behavior as initial detection)
        next_flow = intent_node.get("next_flow")
        if next_flow:
            flow_state.current_step = next_flow.get("name")
            flow_state.flow_data = next_flow

            print_flow_status(room_name, flow_state, "🔄 AUTO-TRANSITION", 
                            f"From intent to: {next_flow.get('type')} - '{next_flow.get('text', '')}'")

            response_text = next_flow.get("text", "")
            add_agent_response_to_history(flow_state, response_text)

            return {
                "type": "flow_started",
                "flow_name": matching_intent["intent"],
                "response": response_text,
                "next_step": next_flow.get("next_flow")
            }
        else:
            # No next_flow on intent node; reply with the intent text
            response_text = intent_node.get("text", "")
            if not response_text or response_text == "N/A":
                response_text = f"I understand you want to know about {matching_intent['intent']}. How can I help you with that?"
            add_agent_response_to_history(flow_state, response_text)
            return {
                "type": "flow_started",
                "flow_name": matching_intent["intent"],
                "response": response_text,
                "next_step": None
            }
    
    # If we reach here, we're in a flow but no specific handling was done
    # This should not happen with the new logic, but as a fallback
    logger.info("FLOW_MANAGEMENT: Unexpected flow state, using FAQ bot as fallback")
    logger.info(f"FLOW_MANAGEMENT: Current flow state - flow: {flow_state.current_flow}, step: {flow_state.current_step}")
    print_flow_status(room_name, flow_state, "❌ UNEXPECTED STATE", "Using FAQ bot fallback")
    return await get_faq_response(user_message, flow_state=flow_state)

async def get_faq_response(user_message: str, bot_id: str = None, flow_state: FlowState = None) -> Dict[str, Any]:
    """Get response from FAQ bot - supports dynamic bot IDs"""
    try:
        logger.info(f"FAQ_RESPONSE: Called with message: '{user_message}', bot_id: {bot_id}")
        if flow_state:
            logger.info(f"FAQ_RESPONSE: Flow state - current_flow: {flow_state.current_flow}, current_step: {flow_state.current_step}")
        
        # Use provided bot_id or an explicit default from env/constant (template 'name' is NOT a bot_id)
        if not bot_id:
            env_bot_id = os.getenv("A5_FAQ_BOT_ID")
            if env_bot_id and env_bot_id.strip():
                bot_id = env_bot_id.strip()
            else:
                # Fallback to known bot id shared by client
                bot_id = "faq_b9952a56-fc7b-41c9-b0a0-5c662ddb039e"
        
        logger.info(f"FAQ_RESPONSE: Using bot_id: {bot_id}")
        print(f"🤖 FAQ BOT CALL: Bot ID: {bot_id} | Question: '{user_message}'")
        
        # FAQ may take ~15s; set a generous timeout
        async with httpx.AsyncClient(timeout=httpx.Timeout(35.0)) as client:
            response = await client.post(
                f"{A5_BASE_URL}/public/1.0/get-faq-bot-response-by-bot-id",
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
            
            print(f"✅ FAQ BOT RESPONSE: {result['data']['answer'][:100]}...")
            
            # Add agent response to conversation history if flow_state is provided
            if flow_state:
                add_agent_response_to_history(flow_state, result["data"]["answer"])
            
            return {
                "type": "faq_response",
                "response": result["data"]["answer"],
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

# New Flow-based Processing Endpoint
@app.post("/api/process_flow_message")
async def process_flow_message_endpoint(request: ProcessFlowMessageRequest):
    """Process user message through the new flow system"""
    try:
        room_name = request.room_name
        user_message = request.user_message
        
        if not room_name or not user_message:
            raise HTTPException(status_code=400, detail="Missing room_name or user_message")
        
        logger.info(f"FLOW_PROCESSING: Room {room_name}, Message: '{user_message}'")
        
        # Process through flow system with conversation history
        flow_result = await process_flow_message(room_name, user_message, request.conversation_history)
        
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
async def refresh_template():
    """Refresh the bot template from Alive5 API"""
    try:
        global bot_template
        new_template = await initialize_bot_template()
        if new_template:
            return {
                "status": "success",
                "message": "Template refreshed successfully",
                "template_version": new_template.get("code", "unknown")
            }
        else:
            raise HTTPException(status_code=500, detail="Failed to refresh template")
    except Exception as e:
        logger.error(f"Template refresh error: {e}")
        raise HTTPException(status_code=500, detail=f"Template refresh failed: {str(e)}")

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
    
    logger.info(f"TEST_INTENT: Testing intent detection for message: '{user_message}'")
    logger.info(f"TEST_INTENT: Conversation history: {conversation_history}")
    
    result = await detect_flow_intent_with_llm_from_conversation(conversation_history)
    
    return {
        "user_message": user_message,
        "conversation_history": conversation_history,
        "detected_intent": result,
        "available_intents": list(bot_template.get("data", {}).keys()) if bot_template else []
    }

# Initialize bot template on startup
@app.on_event("startup")
async def startup_event():
    """Initialize bot template on startup"""
    await initialize_bot_template()

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)