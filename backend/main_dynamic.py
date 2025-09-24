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
import hashlib
import threading


# Load environment variables
# Try multiple possible paths for .env file
import os
from pathlib import Path

# Get the current file's directory
current_dir = Path(__file__).parent
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
        print(f"✅ Loaded .env from: {env_path}")
        env_loaded = True
        break

if not env_loaded:
    print("⚠️ No .env file found in any expected location")
    print(f"   Searched paths: {[str(p) for p in env_paths]}")
    # Fallback to default behavior
    load_dotenv()

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

# Alive5 API endpoints - fully configurable
A5_TEMPLATE_URL = os.getenv("A5_TEMPLATE_URL", "/1.0/org-botchain/generate-template")
A5_FAQ_URL = os.getenv("A5_FAQ_URL", "/public/1.0/get-faq-bot-response-by-bot-id")
A5_BOTCHAIN_NAME = os.getenv("A5_BOTCHAIN_NAME", "dustin-gpt")
A5_ORG_NAME = os.getenv("A5_ORG_NAME", "alive5stage0")
FAQ_BOT_ID = os.getenv("FAQ_BOT_ID", "default-bot-id")

# Template polling configuration
TEMPLATE_POLLING_INTERVAL = int(os.getenv("TEMPLATE_POLLING_INTERVAL", "1"))  # hours
TEMPLATE_POLLING_ENABLED = os.getenv("TEMPLATE_POLLING_ENABLED", "true").lower() == "true"

print(f"Loaded credentials:")
print(f"API_KEY: {LIVEKIT_API_KEY}")
print(f"API_SECRET: {LIVEKIT_API_SECRET[:10] if LIVEKIT_API_SECRET else 'None'}...")
print(f"URL: {LIVEKIT_URL}")
print(f"OPENAI_KEY: {OPENAI_API_KEY[:10] if OPENAI_API_KEY else 'None'}...")
print(f"A5_BASE_URL: {A5_BASE_URL}")
print(f"A5_API_KEY: {A5_API_KEY}")
print(f"A5_TEMPLATE_URL: {A5_TEMPLATE_URL}")
print(f"A5_FAQ_URL: {A5_FAQ_URL}")
print(f"A5_BOTCHAIN_NAME: {A5_BOTCHAIN_NAME}")
print(f"A5_ORG_NAME: {A5_ORG_NAME}")
print(f"FAQ_BOT_ID: {FAQ_BOT_ID}")

# Debug: Show current working directory and file locations
print(f"🔍 DEBUG: Current working directory: {os.getcwd()}")
print(f"🔍 DEBUG: Backend file location: {__file__}")
print(f"🔍 DEBUG: Environment variables loaded: {env_loaded}")

# Create persistence directory
PERSISTENCE_DIR = "flow_states"
if not os.path.exists(PERSISTENCE_DIR):
    os.makedirs(PERSISTENCE_DIR)
    print(f"✅ Created persistence directory: {PERSISTENCE_DIR}")
else:
    print(f"✅ Using existing persistence directory: {PERSISTENCE_DIR}")

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

# Helper: find a step in the template by its exact text (case-insensitive)
def _find_step_by_text(template: Dict[str, Any], target_text: str) -> Optional[Dict[str, Any]]:
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


class TemplateManager:
    """Manages template storage, hashing, and scheduled polling"""
    
    def __init__(self):
        self.template_data = None
        self.template_hash = None
        self.last_updated = None
        self.polling_active = False
        self.polling_thread = None
        self.polling_interval = TEMPLATE_POLLING_INTERVAL
        self.polling_enabled = TEMPLATE_POLLING_ENABLED
        
        # Load environment variables directly to ensure they're available
        self.a5_base_url = os.getenv("A5_BASE_URL")
        self.a5_template_url = os.getenv("A5_TEMPLATE_URL", "/1.0/org-botchain/generate-template")
        self.a5_api_key = os.getenv("A5_API_KEY")
        self.a5_botchain_name = os.getenv("A5_BOTCHAIN_NAME", "dustin-gpt")
        self.a5_org_name = os.getenv("A5_ORG_NAME", "alive5stage0")
        
        # Debug: Print loaded values
        print(f"🔍 TEMPLATE_MANAGER: A5_BASE_URL = {repr(self.a5_base_url)}")
        print(f"🔍 TEMPLATE_MANAGER: A5_TEMPLATE_URL = {repr(self.a5_template_url)}")
        print(f"🔍 TEMPLATE_MANAGER: A5_API_KEY = {repr(self.a5_api_key)}")
        
    def generate_template_hash(self, template_data: Dict[str, Any]) -> str:
        """Generate SHA-256 hash of template data"""
        if not template_data:
            return ""
        
        # Sort keys for consistent hashing
        template_json = json.dumps(template_data, sort_keys=True, separators=(',', ':'))
        return hashlib.sha256(template_json.encode('utf-8')).hexdigest()
    
    async def fetch_template_from_api(self) -> Optional[Dict[str, Any]]:
        """Fetch template from Alive5 API"""
        try:
            # Check if required variables are loaded
            if not self.a5_base_url or not self.a5_api_key:
                print(f"❌ TEMPLATE_POLLING: Missing required environment variables")
                print(f"   A5_BASE_URL: {repr(self.a5_base_url)}")
                print(f"   A5_API_KEY: {repr(self.a5_api_key)}")
                return None
            
            template_endpoint = f"{self.a5_base_url}{self.a5_template_url}"
            print(f"🔄 TEMPLATE_POLLING: Fetching template from {template_endpoint}")
            
            async with httpx.AsyncClient() as client:
                response = await client.post(
                    template_endpoint,
                    headers={
                        "X-A5-APIKEY": self.a5_api_key,
                        "Content-Type": "application/json"
                    },
                    json={
                        "botchain_name": self.a5_botchain_name,
                        "org_name": self.a5_org_name
                    },
                    timeout=30.0
                )
                
                if response.status_code == 200:
                    result = response.json()
                    print(f"✅ TEMPLATE_POLLING: Successfully fetched template")
                    return result
                else:
                    print(f"❌ TEMPLATE_POLLING: API returned status {response.status_code}: {response.text}")
                    return None
                    
        except Exception as e:
            print(f"❌ TEMPLATE_POLLING: Failed to fetch template: {str(e)}")
            logger.error(f"TEMPLATE_POLLING: Failed to fetch template: {str(e)}")
            return None
    
    async def fetch_and_store_template(self) -> bool:
        """Fetch template from Alive5 API and store with hash"""
        try:
            # Fetch from Alive5 API
            template_data = await self.fetch_template_from_api()
            
            if not template_data:
                return False
            
            # Generate hash
            new_hash = self.generate_template_hash(template_data)
            
            # Store template and hash
            self.template_data = template_data
            self.template_hash = new_hash
            self.last_updated = datetime.now()
            
            print(f"✅ TEMPLATE_POLLING: Template updated - Hash: {new_hash[:8]}...")
            logger.info(f"TEMPLATE_POLLING: Template updated - Hash: {new_hash[:8]}...")
            return True
            
        except Exception as e:
            print(f"❌ TEMPLATE_POLLING: Failed to fetch and store template: {str(e)}")
            logger.error(f"TEMPLATE_POLLING: Failed to fetch and store template: {str(e)}")
            return False
    
    async def check_template_updates(self) -> bool:
        """Check if template has changed by comparing hashes"""
        try:
            # Fetch current template from API
            current_template = await self.fetch_template_from_api()
            
            if not current_template:
                return False
            
            current_hash = self.generate_template_hash(current_template)
            
            # Compare with stored hash
            if current_hash != self.template_hash:
                print(f"🔄 TEMPLATE_POLLING: Template changed - Old: {self.template_hash[:8] if self.template_hash else 'None'}... New: {current_hash[:8]}...")
                logger.info(f"TEMPLATE_POLLING: Template changed - Old: {self.template_hash[:8] if self.template_hash else 'None'}... New: {current_hash[:8]}...")
                
                # Update stored template
                self.template_data = current_template
                self.template_hash = current_hash
                self.last_updated = datetime.now()
                
                return True  # Template updated
            else:
                print(f"✅ TEMPLATE_POLLING: Template unchanged - Hash: {current_hash[:8]}...")
                logger.debug(f"TEMPLATE_POLLING: Template unchanged - Hash: {current_hash[:8]}...")
                return False  # No changes
                
        except Exception as e:
            print(f"❌ TEMPLATE_POLLING: Failed to check template updates: {str(e)}")
            logger.error(f"TEMPLATE_POLLING: Failed to check template updates: {str(e)}")
            return False
    
    # Polling mechanism removed - templates loaded on-demand
    
    # stop_polling method removed - no longer needed
    
    # _polling_worker method removed - no longer needed
    
    def get_status(self) -> Dict[str, Any]:
        """Get current template status"""
        return {
            "template_loaded": self.template_data is not None,
            "last_updated": self.last_updated.isoformat() if self.last_updated else None,
            "template_hash": self.template_hash[:8] + "..." if self.template_hash else None,
            "template_size": len(json.dumps(self.template_data)) if self.template_data else 0,
            "template_available": self.template_data is not None,
            "available_intents": self._get_available_intents() if self.template_data else []
        }
    
    def _get_available_intents(self) -> List[str]:
        """Get list of available intents from template"""
        if not self.template_data or not self.template_data.get("data"):
            return []
        
        intents = []
        for flow_key, flow_data in self.template_data["data"].items():
            if flow_data.get("type") == "intent_bot":
                intent_name = flow_data.get("text", "")
                if intent_name:
                    intents.append(intent_name)
        return intents


# Global template manager instance
template_manager = None

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

# Local file-based persistence functions
def save_flow_state_to_file(room_name: str, flow_state: FlowState) -> bool:
    """Save flow state to local JSON file"""
    try:
        # Sanitize room name for filename
        safe_room_name = "".join(c for c in room_name if c.isalnum() or c in ('-', '_')).rstrip()
        file_path = os.path.join(PERSISTENCE_DIR, f"{safe_room_name}.json")
        
        # Convert FlowState to dict
        data = {
            "current_flow": flow_state.current_flow,
            "current_step": flow_state.current_step,
            "flow_data": flow_state.flow_data,
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
        
        logger.info(f"PERSISTENCE: Saved flow state for room {room_name} to {file_path}")
        return True
        
    except Exception as e:
        logger.error(f"PERSISTENCE: Error saving flow state: {e}")
        return False

def load_flow_state_from_file(room_name: str) -> Optional[FlowState]:
    """Load flow state from local JSON file"""
    try:
        # Sanitize room name for filename
        safe_room_name = "".join(c for c in room_name if c.isalnum() or c in ('-', '_')).rstrip()
        file_path = os.path.join(PERSISTENCE_DIR, f"{safe_room_name}.json")
        
        if not os.path.exists(file_path):
            logger.info(f"PERSISTENCE: No flow state file found for room {room_name}")
            return None
        
        with open(file_path, 'r') as f:
            data = json.load(f)
        
        # Check if file is too old (older than 24 hours)
        saved_at = data.get("saved_at", 0)
        if time.time() - saved_at > 24 * 60 * 60:  # 24 hours
            logger.info(f"PERSISTENCE: Flow state for room {room_name} is too old, ignoring")
            os.remove(file_path)  # Clean up old file
            return None
        
        flow_state = FlowState(
            current_flow=data.get("current_flow"),
            current_step=data.get("current_step"),
            flow_data=data.get("flow_data"),
            conversation_history=data.get("conversation_history", []),
            user_responses=data.get("user_responses"),
            pending_step=data.get("pending_step"),
            pending_expected_kind=data.get("pending_expected_kind"),
            pending_asked_at=data.get("pending_asked_at"),
            pending_reask_count=data.get("pending_reask_count", 0)
        )
        
        logger.info(f"PERSISTENCE: Loaded flow state for room {room_name} from {file_path}")
        return flow_state
        
    except Exception as e:
        logger.error(f"PERSISTENCE: Error loading flow state: {e}")
        return None

def delete_flow_state_from_file(room_name: str) -> bool:
    """Delete flow state from local JSON file"""
    try:
        # Sanitize room name for filename
        safe_room_name = "".join(c for c in room_name if c.isalnum() or c in ('-', '_')).rstrip()
        file_path = os.path.join(PERSISTENCE_DIR, f"{safe_room_name}.json")
        
        if os.path.exists(file_path):
            os.remove(file_path)
            logger.info(f"PERSISTENCE: Deleted flow state file for room {room_name}")
        else:
            logger.info(f"PERSISTENCE: No flow state file found to delete for room {room_name}")
        
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
                        logger.info(f"PERSISTENCE: Cleaned up old flow state file: {filename}")
                        
                except Exception as e:
                    logger.warning(f"PERSISTENCE: Error processing file {filename}: {e}")
        
        if cleaned_count > 0:
            logger.info(f"PERSISTENCE: Cleaned up {cleaned_count} old flow state files")
        
    except Exception as e:
        logger.error(f"PERSISTENCE: Error during cleanup: {e}")

# Clean up old files on startup
cleanup_old_flow_states()

def is_ambiguous_transcription(user_text: str) -> bool:
    """Detect if the user text appears to be a garbled/ambiguous transcription."""
    u = (user_text or "").lower().strip()
    
    # Check for common garbled patterns
    garbled_patterns = [
        r"\bthrough\s+\w+\s+lines?\b",  # "through phone lines"
        r"\bwe\s+use\s*$",  # "we use" at end
        r"\babout\s+to\s*$",  # "about to" (likely "about two")
        r"\bto\s+fifty\s*\??",  # "to fifty?" (likely "two fifty")
        r"\buh\s*$",  # "uh" at end
        r"\bum\s*$",  # "um" at end
        r"\buh\s+can\s+i\b",  # "uh can i"
        r"\bthat\s+is\s+a\s+question\s+i\s+i\s+think\b",  # repeated words
        r"\bsome\s+some\b",  # repeated words
        r"\babout\s+two\s+two\b",  # repeated words
    ]
    
    # Check for incomplete sentences (ends with articles/prepositions)
    incomplete_endings = [
        r"\bthe\s*$", r"\ba\s*$", r"\ban\s*$", r"\bto\s*$", r"\bfor\s*$", 
        r"\bwith\s*$", r"\bin\s*$", r"\bon\s*$", r"\bat\s*$", r"\bby\s*$",
        r"\bwe\s*$", r"\buse\s*$", r"\bthrough\s*$"
    ]
    
    # Check for very short responses that don't make sense
    if len(u.split()) <= 2 and not re.search(r"\b(yes|no|ok|okay|thanks|bye|hello|hi)\b", u):
        # If it's very short and doesn't contain common words, it might be garbled
        if not re.search(r"\b\d+\b", u):  # Unless it contains numbers
            return True
    
    # Check for garbled patterns
    for pattern in garbled_patterns:
        if re.search(pattern, u):
            return True
    
    # Check for incomplete endings
    for pattern in incomplete_endings:
        if re.search(pattern, u):
            return True
    
    # Check for excessive repetition of words
    words = u.split()
    if len(words) > 2:
        word_counts = {}
        for word in words:
            word_counts[word] = word_counts.get(word, 0) + 1
        # If any word appears more than once in a short phrase, it might be garbled
        if max(word_counts.values()) > 1 and len(words) <= 5:
            return True
    
    return False

def interpret_answer(question_text: str, user_text: str) -> Dict[str, Any]:
    """Extract structured answers from natural speech for common question types."""
    q = (question_text or "").lower()
    u = (user_text or "").lower().strip()

    # First check for ambiguous/garbled transcriptions
    if is_ambiguous_transcription(u):
        return {"status": "unclear", "kind": "ambiguous", "value": u, "confidence": 0.0}

    # Yes/No
    yes_triggers = ["special needs", "sso", "salesforce", "crm integration", "do you", "would you", "are you", "is it", "should we", "can you"]
    if any(k in q for k in yes_triggers):
        if re.search(r"\b(yes|yeah|yep|yup|sure|of course|please|affirmative|ok|okay|absolutely)\b", u):
            return {"status": "extracted", "kind": "yesno", "value": True, "confidence": 0.9}
        if re.search(r"\b(no|nope|nah|negative|not really|don\'t|do not)\b", u):
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
        # Look for numbers in various formats
        m = re.search(r"\b(\d{1,5})\b", u)
        if m:
            return {"status": "extracted", "kind": "number", "value": int(m.group(1)), "confidence": 0.85}
        
        # Look for word numbers (five hundred, two fifty, etc.)
        words_to_num = {
            "one":1,"two":2,"three":3,"four":4,"five":5,"six":6,"seven":7,"eight":8,"nine":9,"ten":10,
            "eleven":11,"twelve":12,"thirteen":13,"fourteen":14,"fifteen":15,"sixteen":16,"seventeen":17,
            "eighteen":18,"nineteen":19,"twenty":20,"thirty":30,"forty":40,"fifty":50,"sixty":60,
            "seventy":70,"eighty":80,"ninety":90,"hundred":100,"thousand":1000
        }
        
        # Handle "five hundred" pattern
        if re.search(r"\bfive\s+hundred\b", u):
            return {"status": "extracted", "kind": "number", "value": 500, "confidence": 0.9}
        if re.search(r"\btwo\s+fifty\b", u):
            return {"status": "extracted", "kind": "number", "value": 250, "confidence": 0.9}
        if re.search(r"\btwo\s+hundred\b", u):
            return {"status": "extracted", "kind": "number", "value": 200, "confidence": 0.9}
        if re.search(r"\bthree\s+hundred\b", u):
            return {"status": "extracted", "kind": "number", "value": 300, "confidence": 0.9}
        if re.search(r"\bfour\s+hundred\b", u):
            return {"status": "extracted", "kind": "number", "value": 400, "confidence": 0.9}
        if re.search(r"\bsix\s+hundred\b", u):
            return {"status": "extracted", "kind": "number", "value": 600, "confidence": 0.9}
        if re.search(r"\bseven\s+hundred\b", u):
            return {"status": "extracted", "kind": "number", "value": 700, "confidence": 0.9}
        if re.search(r"\beight\s+hundred\b", u):
            return {"status": "extracted", "kind": "number", "value": 800, "confidence": 0.9}
        if re.search(r"\bnine\s+hundred\b", u):
            return {"status": "extracted", "kind": "number", "value": 900, "confidence": 0.9}
        
        # Handle single word numbers
        for w, n in words_to_num.items():
            if re.search(rf"\b{w}\b", u) and n <= 1000:  # Only reasonable text message counts
                return {"status": "extracted", "kind": "number", "value": n, "confidence": 0.8}

    return {"status": "unclear", "kind": "text", "value": u, "confidence": 0.0}

def gated_llm_extract_answer(question_text: str, user_text: str) -> Dict[str, Any]:
    """Always use LLM for answer extraction with deterministic parser context.
    
    This is the new robust approach that:
    1. First runs deterministic parser for initial analysis
    2. Always calls LLM with parser context for final decision
    3. Provides more reliable extraction than either method alone
    
    Args:
        question_text: The question being asked
        user_text: The user's natural language response
        
    Returns:
        Dict with status, kind, value, confidence
    """
    # Step 1: Run deterministic parser for initial analysis
    parser_result = interpret_answer(question_text, user_text)
    
    # Step 2: Always call LLM with parser context
    llm_result = llm_extract_answer(question_text, user_text, parser_result)
    
    # Log the comparison for debugging
    logger.info(f"GATED_LLM: Parser={parser_result}, LLM={llm_result}")
    
    # Return LLM result (it has the final decision)
    return llm_result

def llm_extract_answer(question_text: str, user_text: str, parser_context: Dict[str, Any] = None) -> Dict[str, Any]:
    """LLM-based extractor for natural responses with deterministic parser context.
    Returns the same schema as interpret_answer. Uses strict JSON output instructions.
    """
    try:
        if not OPENAI_API_KEY:
            return {"status": "unclear", "kind": "text", "value": user_text, "confidence": 0.0}
        client = openai.OpenAI(api_key=OPENAI_API_KEY)
        system = (
            "You extract structured answers from a user's natural reply. "
            "You will receive both the raw user input and a deterministic parser's analysis. "
            "Use the parser's analysis as context, but make your own intelligent decision. "
            "Return a JSON object only, no prose, with keys: status ('extracted'|'unclear'), "
            "kind ('number'|'zip'|'yesno'|'text'|'ambiguous'), value, confidence (0..1)."
        )
        
        # Build context from parser analysis
        parser_info = ""
        if parser_context:
            parser_info = f"\nParser Analysis: {parser_context.get('status', 'unknown')} - {parser_context.get('kind', 'unknown')} - {parser_context.get('value', 'none')} (confidence: {parser_context.get('confidence', 0)})"
        
        user = (
            "Question: " + (question_text or "") + "\n"
            "User reply: " + (user_text or "") + parser_info + "\n"
            "Rules:\n"
            "- If user gives a quantity like 'two phone lines' or 'twenty four', set kind='number' and value as integer.\n"
            "- If it's a ZIP like 'two five nine six three', set kind='zip' and 5-digit value.\n"
            "- If yes/no ('yes', 'no', etc.), set kind='yesno' and value true/false.\n"
            "- If the reply appears garbled, incomplete, or nonsensical (like 'through phone lines we use', 'about to', 'uh can i'), set status='unclear' and kind='ambiguous'.\n"
            "- If the reply is incomplete or ends with articles/prepositions ('the', 'to', 'we', 'use'), set status='unclear'.\n"
            "- If words are repeated unnecessarily ('some some', 'two two'), set status='unclear'.\n"
            "- Consider the parser's analysis but make your own decision - you may override the parser if it seems wrong.\n"
            "- For ambiguous transcriptions, prioritize asking for clarification over guessing.\n"
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


async def smart_message_processor(user_message: str, current_flow_context: Dict[str, Any] = None) -> Dict[str, Any]:
    """
    Smart LLM processor that analyzes every user message to determine:
    - Intent detection
    - Context understanding
    - Appropriate response strategy
    - Whether to continue current flow or switch
    """
    try:
        # Get current flow context
        current_flow = current_flow_context.get("current_flow") if current_flow_context else None
        current_step = current_flow_context.get("current_step") if current_flow_context else None
        current_step_type = current_flow_context.get("current_step_type") if current_flow_context else None
        
        # Create context-aware prompt
        context_info = ""
        if current_flow and current_step:
            context_info = f"""
CURRENT CONVERSATION CONTEXT:
- Current Flow: {current_flow}
- Current Step: {current_step}
- Step Type: {current_step_type}
- User is responding to a question or in a conversation flow
"""
        
        # Get available intents dynamically from bot template
        available_intents = []
        if bot_template and bot_template.get("data"):
            for flow_key, flow_data in bot_template["data"].items():
                if flow_data.get("type") == "intent_bot":
                    intent_name = flow_data.get("text", flow_key)  # Use text field for intent name
                    available_intents.append(intent_name)
        
        intents_list = ", ".join(available_intents) if available_intents else "none available"
        
        prompt = f"""You are a smart conversation analyzer. Analyze the user's message and determine the best response strategy.

{context_info}

USER MESSAGE: "{user_message}"

AVAILABLE INTENTS: {intents_list}

ANALYSIS TASKS:
1. INTENT DETECTION: Does this message indicate a clear intent from the available list?
2. CONTEXT UNDERSTANDING: Is this a response to a question, filler/stuttering, or a new topic?
3. RESPONSE STRATEGY: What should the agent do next?

RESPONSE FORMAT (JSON):
{{
    "intent_detected": "intent_name|none",
    "message_type": "intent_request|question_response|filler|unclear|new_topic",
    "confidence": "high|medium|low",
    "action": "continue_flow|switch_intent|ask_clarification|ignore|respond_naturally",
    "reasoning": "brief explanation of the analysis"
}}

EXAMPLES:
- "Yeah, I'm looking for someone to help" → {{"intent_detected": "agent", "message_type": "intent_request", "confidence": "high", "action": "switch_intent", "reasoning": "Clear request for human help"}}
- "Yeah" → {{"intent_detected": "none", "message_type": "question_response", "confidence": "medium", "action": "continue_flow", "reasoning": "Simple affirmation to current question"}}
- "Uh, I, uh, I was asking" → {{"intent_detected": "none", "message_type": "filler", "confidence": "high", "action": "ignore", "reasoning": "Stuttering/filler, not meaningful content"}}
- "Can I speak with someone?" → {{"intent_detected": "agent", "message_type": "intent_request", "confidence": "high", "action": "switch_intent", "reasoning": "Direct request for human agent"}}

Respond with ONLY the JSON object, no other text."""

        logger.info(f"🧠 SMART PROCESSOR: Analyzing message: '{user_message}'")
        
        client = openai.OpenAI(api_key=OPENAI_API_KEY)
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[{"role": "user", "content": prompt}],
            max_tokens=200,
            temperature=0.1
        )
        response_text = response.choices[0].message.content.strip()
        logger.info(f"🧠 SMART PROCESSOR: LLM response: {response_text}")
        
        # Parse JSON response
        try:
            analysis = json.loads(response_text.strip())
            logger.info(f"🧠 SMART PROCESSOR: Analysis result: {analysis}")
            return analysis
        except json.JSONDecodeError:
            logger.error(f"🧠 SMART PROCESSOR: Failed to parse JSON: {response_text}")
            return {
                "intent_detected": "none",
                "message_type": "unclear",
                "confidence": "low",
                "action": "ask_clarification",
                "reasoning": "Failed to parse LLM response"
            }
            
    except Exception as e:
        logger.error(f"🧠 SMART PROCESSOR: Error: {e}")
        return {
            "intent_detected": "none",
            "message_type": "unclear",
            "confidence": "low",
            "action": "ask_clarification",
            "reasoning": f"Error in processing: {e}"
        }

async def detect_flow_intent_with_llm(user_message: str) -> Optional[Dict[str, Any]]:
    """Detect flow intent using LLM - simple and direct approach"""
    try:
        print(f"🔍 INTENT_DETECTION: Starting detection for: '{user_message}'")
        if not bot_template or not bot_template.get("data"):
            logger.warning("INTENT_DETECTION: No bot template available")
            print(f"❌ INTENT_DETECTION: No bot template available")
            return None
        
        # Extract available intents from template
        available_intents = []
        intent_mapping = {}
        
        for flow_key, flow_data in bot_template["data"].items():
            if flow_data.get("type") == "intent_bot":
                intent_name = flow_data.get("text", flow_key)  # Use text field for intent name
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
You are an intent classifier. Your job is to match the user's message to one of the available intents.

Available intents: {intent_list}

User message: "{user_message}"

ANALYSIS STEPS:
1. Read the user's message carefully
2. Identify what the user is asking for or wants to do
3. Match it to the most appropriate intent from the list above
4. Consider the meaning and intent behind the words, not just exact matches

SPECIAL CASES:
- Greetings like "Hello", "Hi", "How are you?" → respond with "greeting"
- Agent/human requests like "speak with someone", "talk to an agent", "connect me with a human", "over the phone", "real person", "human agent", "talk to someone", "can I speak with", "I want to speak with", "I need to speak with", "get me someone", "transfer me", "put me through" → match with the appropriate intent from the available list
- Pricing questions like "cost", "price", "plans", "how much" → match with the appropriate intent from the available list
- Weather questions → match with the appropriate intent from the available list

CRITICAL: If the user is asking to speak with a human, agent, or person in ANY way, they want the "agent" intent (if available in the list).

IMPORTANT: The user said: "{user_message}"
Think about what they really want. Are they asking to speak to a person? Do they want pricing information? Are they asking about weather?

Respond with ONLY the exact intent name from the list above (case-insensitive), "greeting", or "none" if no intent matches.

Examples:
- "Can I speak with someone over the phone?" → agent (if available, they want to talk to a human)
- "Can I speak with someone, please?" → agent (if available, they want human help)
- "I wanna talk to a real person" → agent (if available, they want human help)
- "get me connected with someone else" → agent (if available, they want human help)
- "I need to speak with an agent" → agent (if available, they want human help)
- "Can you transfer me to someone?" → agent (if available, they want human help)
- "What's the weather like?" → weather (if available)
- "How much does it cost?" → pricing (if available)
- "Hello there" → greeting
- "I need help with billing" → agent (if available, they want human help)
"""
        
        logger.info(f"INTENT_DETECTION: Analyzing message '{user_message}' for intents: {intent_list}")
        logger.info(f"INTENT_DETECTION: Available intents mapping: {list(intent_mapping.keys())}")
        print(f"🔍 INTENT DETECTION: Available intents: {intent_list}")
        print(f"🔍 INTENT DETECTION: User message: '{user_message}'")

        client = openai.OpenAI(api_key=OPENAI_API_KEY)
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[{"role": "user", "content": prompt}],
            max_tokens=50,
            temperature=0.0
        )
        
        detected_intent = response.choices[0].message.content.strip()
        logger.info(f"INTENT_DETECTION: LLM response: '{detected_intent}'")
        print(f"🔍 INTENT DETECTION: LLM returned: '{detected_intent}'")
        
        # Handle special "greeting" response
        if detected_intent == "greeting":
            logger.info(f"INTENT_DETECTION: ✅ Greeting detected")
            return {"type": "greeting", "intent": "greeting"}
        
        # Find matching intent
        for intent_name, intent_data in intent_mapping.items():
            if detected_intent.lower() == intent_name.lower():
                logger.info(f"INTENT_DETECTION: ✅ Intent found: '{intent_name}'")
                print(f"✅ INTENT MATCHED: '{detected_intent}' -> '{intent_name}'")
                return intent_data
        
        logger.info(f"INTENT_DETECTION: ❌ No intent found, will use FAQ bot")
        print(f"❌ INTENT NOT FOUND: '{detected_intent}' not in {list(intent_mapping.keys())}")
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
    """Initiate transfer to human agent"""
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
        logger.info(f"ALIVE5_API: Generating template for {A5_BOTCHAIN_NAME} in org {A5_ORG_NAME}")
        
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
        raise HTTPException(status_code=400, detail=f"Alive5 API request failed: {str(e)}")
    except Exception as e:
        logger.error(f"ALIVE5_API: Error in FAQ response: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")

# Flow Management Functions
async def initialize_bot_template():
    """Initialize the bot template on startup using TemplateManager"""
    global bot_template, template_manager
    
    try:
        print("\n" + "="*80)
        print("🚀 INITIALIZING BOT TEMPLATE WITH POLLING SYSTEM")
        print("="*80)
        logger.info("FLOW_MANAGEMENT: Initializing bot template with polling system...")
        
        # Initialize template manager
        template_manager = TemplateManager()
        
        # Fetch initial template
        success = await template_manager.fetch_and_store_template()
        
        if success:
            # Set the global bot_template for backward compatibility
            bot_template = template_manager.template_data
            
            logger.info("FLOW_MANAGEMENT: Bot template initialized successfully with polling")
            print("✅ TEMPLATE LOADED SUCCESSFULLY WITH POLLING SYSTEM")
            print(f"📊 Template contains {len(bot_template['data'])} flows:")
            for flow_key, flow_data in bot_template["data"].items():
                flow_type = flow_data.get("type", "unknown")
                flow_text = flow_data.get("text", "")
                print(f"   🔹 {flow_key}: {flow_type} - '{flow_text}'")
            print(f"🔐 Template Hash: {template_manager.template_hash[:8]}...")
            print(f"⏰ Last Updated: {template_manager.last_updated}")
            print(f"🔄 Polling Enabled: {template_manager.polling_enabled}")
            print(f"⏱️ Polling Interval: {template_manager.polling_interval} hour(s)")
            print("="*80)
            
            # Polling removed - templates loaded on-demand
            
            # Verify global variable is set
            print(f"🔍 VERIFICATION: bot_template is {'✅ SET' if bot_template is not None else '❌ NONE'}")
            logger.info(f"FLOW_MANAGEMENT: Global bot_template set: {bot_template is not None}")
            
            return bot_template
        else:
            logger.error("FLOW_MANAGEMENT: Failed to initialize bot template")
            print("❌ TEMPLATE LOAD FAILED")
            return None
            
    except Exception as e:
        logger.error(f"FLOW_MANAGEMENT: Failed to initialize bot template: {str(e)}")
        print(f"❌ TEMPLATE INITIALIZATION ERROR: {str(e)}")
        import traceback
        print(f"❌ TRACEBACK: {traceback.format_exc()}")
        return None

    # Removed mock template: always fetch from Alive5 API per client requirement

async def initialize_bot_template_with_config(botchain_name: str, org_name: str):
    """Initialize bot template with custom configuration"""
    global bot_template, template_manager
    
    logger.info(f"🚀 INITIALIZING BOT TEMPLATE WITH CUSTOM CONFIG: {botchain_name}/{org_name}")
    print(f"🚀 INITIALIZING BOT TEMPLATE WITH CUSTOM CONFIG: {botchain_name}/{org_name}")
    
    try:
        # Create temporary template manager with custom config
        temp_template_manager = TemplateManager()
        temp_template_manager.a5_botchain_name = botchain_name
        temp_template_manager.a5_org_name = org_name
        
        # Fetch template with custom config
        success = await temp_template_manager.fetch_and_store_template()
        
        if success:
            bot_template = temp_template_manager.template_data
            template_manager = temp_template_manager  # Update global manager
            
            logger.info("✅ CUSTOM TEMPLATE LOADED SUCCESSFULLY")
            print("✅ CUSTOM TEMPLATE LOADED SUCCESSFULLY")
            
            # Print template summary
            if bot_template and bot_template.get("data"):
                print(f"📊 Custom template contains {len(bot_template['data'])} flows:")
                for flow_key, flow_data in bot_template["data"].items():
                    flow_type = flow_data.get("type", "unknown")
                    flow_text = flow_data.get("text", "N/A")
                    print(f"   🔹 {flow_key}: {flow_type} - '{flow_text}'")
            
            return bot_template
        else:
            logger.error("❌ CUSTOM TEMPLATE LOAD FAILED")
            print("❌ CUSTOM TEMPLATE LOAD FAILED")
            return None
            
    except Exception as e:
        logger.error(f"❌ CUSTOM TEMPLATE INITIALIZATION ERROR: {e}")
        print(f"❌ CUSTOM TEMPLATE INITIALIZATION ERROR: {e}")
        return None

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

async def process_flow_message(room_name: str, user_message: str, frontend_conversation_history: List[Dict[str, str]] = None, botchain_name: str = None, org_name: str = None) -> Dict[str, Any]:
    """Process user message through the flow system"""
    global bot_template
    
    # Check if we need to load template with custom configuration
    if botchain_name and org_name:
        logger.info(f"FLOW_MANAGEMENT: Loading template with custom config - Botchain: {botchain_name}, Org: {org_name}")
        try:
            await initialize_bot_template_with_config(botchain_name, org_name)
        except Exception as e:
            logger.error(f"FLOW_MANAGEMENT: Error loading template with custom config: {e}")
            return {
                "status": "error",
                "message": "Failed to load bot configuration",
                "flow_result": {
                    "type": "error",
                    "response": "I'm experiencing technical difficulties. Please try again in a moment."
                }
            }
    
    # Ensure bot template is loaded before processing
    if bot_template is None:
        logger.warning("FLOW_MANAGEMENT: Bot template not loaded, attempting to initialize...")
        print("⚠️ BOT TEMPLATE NOT LOADED - ATTEMPTING INITIALIZATION")
        try:
            await initialize_bot_template()
            if bot_template is None:
                logger.error("FLOW_MANAGEMENT: Failed to initialize bot template during request processing")
                return {
                    "status": "error",
                    "message": "Bot template not available. Please try again.",
                    "flow_result": {
                        "type": "error",
                        "response": "I'm experiencing technical difficulties. Please try again in a moment."
                    }
                }
        except Exception as e:
            logger.error(f"FLOW_MANAGEMENT: Error initializing bot template during request: {e}")
            return {
                "status": "error", 
                "message": "Bot initialization failed",
                "flow_result": {
                    "type": "error",
                    "response": "I'm experiencing technical difficulties. Please try again in a moment."
                }
            }
    
    logger.info(f"FLOW_MANAGEMENT: Processing message for room {room_name}: '{user_message}'")
    
    # Note: Greeting bot is now handled by the worker in on_enter() method
    # This ensures the greeting is sent immediately when the user joins the room
    
    # Get or create flow state for this room
    if room_name not in flow_states:
        # Try to load from local file first
        flow_state = load_flow_state_from_file(room_name)
        if flow_state:
            flow_states[room_name] = flow_state
            print_flow_status(room_name, flow_state, "SESSION RESTORED FROM FILE", f"User message: '{user_message}'")
            logger.info(f"FLOW_MANAGEMENT: Restored flow state from file for room {room_name}")
        else:
            flow_states[room_name] = FlowState()
            print_flow_status(room_name, flow_states[room_name], "NEW SESSION CREATED", f"User message: '{user_message}'")
            logger.info(f"FLOW_MANAGEMENT: Created new flow state for room {room_name}")
    else:
        logger.info(f"FLOW_MANAGEMENT: Using existing flow state for room {room_name}")
    
    flow_state = flow_states[room_name]
    
    # Auto-save flow state to file after any changes
    def auto_save_flow_state():
        save_flow_state_to_file(room_name, flow_state)
    
    # Define escalation phrases and helper function here
    escalate_phrases = [
        "agent", "human", "representative", "connect me", "talk to", "speak to", "speak with", "someone", "person", "escalate", "transfer", "over the phone", "over the line"
    ]

    def _matches_any(phrases: list[str], text: str) -> bool:
        return any(p in text for p in phrases)

    # Check for escalation phrases immediately
    um_low = (user_message or "").lower().strip()
    if _matches_any(escalate_phrases, um_low):
        response_text = "I'm connecting you with a human agent. Please hold on."
        add_agent_response_to_history(flow_state, response_text)
        auto_save_flow_state()  # Save after escalation
        logger.info("FLOW_MANAGEMENT: Global escalation detected → initiating agent handoff")
        return {
            "type": "agent_handoff",
            "response": response_text,
            "flow_state": flow_state,
            "agent_required": True,
            "escalation_reason": "user_requested_agent"
        }

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
    
    # Global farewell detection to gracefully end calls regardless of step type
    um_low = (user_message or "").lower().strip()
    farewell_markers = [
        "bye", "goodbye", "that is all", "that's all", "thats all", "thanks, bye", "thank you, bye", "end call", "hang up", "we are done", "we're done", "okay, bye", "ok bye", "okay that's all", "ok that's all", "i think that's all", "that's all goodbye"
    ]
    if any(m in um_low for m in farewell_markers):
        response_text = "Thanks for calling Alive5. Have a great day! Goodbye!"
        add_agent_response_to_history(flow_state, response_text)
        logger.info("FLOW_MANAGEMENT: Global farewell detected → conversation_end")
        return {
            "type": "conversation_end",
            "response": response_text,
            "flow_state": flow_state
        }

    # If no current flow, try to find matching intent using LLM
    if not flow_state.current_flow:
        # Check if we can recover flow state from conversation history
        if flow_state.conversation_history and len(flow_state.conversation_history) > 0:
            # Look for recent agent messages that might indicate we were in a flow
            recent_agent_messages = [
                msg for msg in flow_state.conversation_history[-5:] 
                if msg.get("role") == "assistant" and msg.get("content")
            ]
            
            # Check if any recent agent message looks like a flow question
            for msg in recent_agent_messages:
                content = msg.get("content", "").lower()
                if any(keyword in content for keyword in ["how many phone lines", "how many texts", "special needs", "sso", "crm"]):
                    logger.info(f"FLOW_MANAGEMENT: Detected potential flow recovery from conversation history: '{content[:50]}...'")
                    # Try to recover by starting pricing flow
                    matching_intent = await detect_flow_intent_with_llm("pricing information")
                    if matching_intent:
                        logger.info(f"FLOW_MANAGEMENT: Recovered flow state for pricing intent")
                        flow_state.current_flow = matching_intent["flow_key"]
                        flow_state.current_step = matching_intent["flow_data"]["name"]
                        flow_state.flow_data = matching_intent["flow_data"]
                        break
            
            # Also check if the current user message looks like a flow response
            user_msg_lower = (user_message or "").lower()
            if any(keyword in user_msg_lower for keyword in ["phone lines", "text messages", "texts", "three hundred", "two hundred", "five hundred"]):
                logger.info(f"FLOW_MANAGEMENT: User message looks like flow response, attempting recovery")
                # Try to recover by starting pricing flow
                matching_intent = await detect_flow_intent_with_llm("pricing information")
                if matching_intent:
                    logger.info(f"FLOW_MANAGEMENT: Recovered flow state from user message context")
                    flow_state.current_flow = matching_intent["flow_key"]
                    flow_state.current_step = matching_intent["flow_data"]["name"]
                    flow_state.flow_data = matching_intent["flow_data"]
                    # Skip intent detection and go directly to flow processing
                    logger.info(f"FLOW_MANAGEMENT: Skipping intent detection, processing as flow response")
                    # Continue to flow processing below
        
        # Only run intent detection if we still don't have a flow
        matching_intent = None
        print(f"🔍 FLOW CHECK: current_flow = '{flow_state.current_flow}', current_step = '{flow_state.current_step}'")
        if not flow_state.current_flow:
            print_flow_status(room_name, flow_state, "SEARCHING FOR INTENT", f"Analyzing message: '{user_message}'")
            logger.info(f"FLOW_MANAGEMENT: Bot template available: {bot_template is not None}")
            if bot_template:
                logger.info(f"FLOW_MANAGEMENT: Bot template data keys: {list(bot_template.get('data', {}).keys())}")
                # Debug: Show all available intents
                for flow_key, flow_data in bot_template.get('data', {}).items():
                    if flow_data.get('type') == 'intent_bot':
                        logger.info(f"FLOW_MANAGEMENT: Available intent '{flow_data.get('text', '')}' in flow {flow_key}")
            
            print(f"🔍 CALLING INTENT DETECTION for: '{user_message}'")
            matching_intent = await detect_flow_intent_with_llm(user_message)
            logger.info(f"FLOW_MANAGEMENT: Intent detection result: {matching_intent}")
            print(f"🔍 INTENT DETECTION: '{user_message}' -> {matching_intent}")
        else:
            print(f"🔍 SKIPPING INTENT DETECTION: Already in flow '{flow_state.current_flow}'")
        
        if matching_intent:
            print(f"✅ INTENT FOUND: {matching_intent}")
            
            # Skip greeting intent detection - greeting is handled by worker
            if matching_intent.get("type") == "greeting":
                logger.info("FLOW_MANAGEMENT: Greeting intent detected, but greeting is handled by worker - skipping")
                # Don't send another greeting response, just continue with normal flow
                pass
            
            # Handle regular intent flows
            logger.info(f"FLOW_MANAGEMENT: ✅ INTENT DETECTED - {matching_intent['intent']} -> {matching_intent['flow_key']}")
            logger.info(f"FLOW_MANAGEMENT: Flow data: {matching_intent['flow_data']}")
            
            flow_state.current_flow = matching_intent["flow_key"]
            flow_state.current_step = matching_intent["flow_data"]["name"]
            flow_state.flow_data = matching_intent["flow_data"]
            auto_save_flow_state()  # Save after flow state changes
            
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
            # No matching intent found, but check for escalation before FAQ fallback
            logger.info("FLOW_MANAGEMENT: ❌ LLM found no matching intent, checking for escalation")
            
            # Double-check for escalation phrases that might have been missed
            um_low_fallback = (user_message or "").lower().strip()
            if _matches_any(escalate_phrases, um_low_fallback):
                response_text = "I'm connecting you with a human agent. Please hold on."
                add_agent_response_to_history(flow_state, response_text)
                auto_save_flow_state()  # Save after escalation
                logger.info("FLOW_MANAGEMENT: Escalation detected in fallback check → initiating agent handoff")
                return {
                    "type": "agent_handoff",
                    "response": response_text,
                    "flow_state": flow_state,
                    "agent_required": True,
                    "escalation_reason": "fallback_escalation"
                }
            
            # Use FAQ bot as final fallback
            print_flow_status(room_name, flow_state, "❌ NO INTENT FOUND", "Using FAQ bot fallback")
            print(f"🚨 FAQ BOT CALLED: No intent found for '{user_message}'")
            return await get_faq_response(user_message, flow_state=flow_state)
    
    # SMART MESSAGE PROCESSING: Use LLM to analyze every user message
    current_flow_context = {
        "current_flow": flow_state.current_flow,
        "current_step": flow_state.current_step,
        "current_step_type": flow_state.flow_data.get("type") if flow_state.flow_data else None
    }
    
    # Process message with smart LLM analyzer
    message_analysis = await smart_message_processor(user_message, current_flow_context)
    logger.info(f"🧠 SMART ANALYSIS: {message_analysis}")
    
    # Handle based on analysis
    if message_analysis.get("action") == "ignore":
        logger.info(f"🧠 IGNORING MESSAGE: {message_analysis.get('reasoning')}")
        return {
            "type": "ignored",
            "response": "",
            "flow_state": flow_state
        }
    
    # If we're already in a flow, check if this is a response to a question or greeting
    if flow_state.current_flow and flow_state.current_step:
        logger.info(f"FLOW_MANAGEMENT: Already in flow {flow_state.current_flow}, step {flow_state.current_step}")
        logger.info(f"FLOW_MANAGEMENT: Flow data: {flow_state.flow_data}")
        
        
        # Check if current step is a question, greeting, or message and user provided a response
        current_step_data = flow_state.flow_data
        if current_step_data and current_step_data.get("type") in ["question", "greeting", "message"]:
            step_type = current_step_data.get("type")
            logger.info(f"FLOW_MANAGEMENT: Current step is a {step_type}, processing user response: '{user_message}'")
            
            # Global farewell within any step context
            um_low_q = (user_message or "").lower().strip()
            farewell_markers_q = [
                "bye", "goodbye", "that is all", "that's all", "thats all", "thanks, bye", "thank you, bye", "end call", "hang up", "we are done", "we're done", "okay, bye", "okay that's all", "ok that's all", "ok bye"
            ]
            if any(m in um_low_q for m in farewell_markers_q):
                response_text = "Thanks for calling Alive5. Have a great day! Goodbye!"
                add_agent_response_to_history(flow_state, response_text)
                logger.info("FLOW_MANAGEMENT: Farewell detected during step → conversation_end")
                return {"type": "conversation_end", "response": response_text, "flow_state": flow_state}
            
            # Handle greeting and message steps differently from question steps
            if step_type in ["greeting", "message"]:
                logger.info(f"FLOW_MANAGEMENT: Processing {step_type} step response")
                print(f"🎯 GREETING FLOW: Processing user message: '{user_message}'")
                # For greeting/message steps, check if user wants to continue to next step or change intent
                
                # Check for intent shift - if user mentions something that matches an intent, switch flows
                # But don't detect greeting intents when we're already in a greeting flow
                print(f"🎯 GREETING FLOW: Calling intent detection for: '{user_message}'")
                matching_intent = await detect_flow_intent_with_llm(user_message)
                print(f"🎯 GREETING FLOW: Intent detection result: {matching_intent}")
                if matching_intent and matching_intent.get("type") != "greeting":
                    logger.info(f"FLOW_MANAGEMENT: Intent shift detected from {step_type} to {matching_intent['intent']}")
                    print_flow_status(room_name, flow_state, "🔄 INTENT SHIFT DETECTED", 
                                    f"From: {flow_state.current_flow} → To: {matching_intent['flow_key']} | Intent: {matching_intent['intent']}")
                    
                    # Update flow state to new intent
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
                
                # Special case: If we're in a greeting flow and user mentions something that should trigger intent detection
                # but it's not detected as a specific intent, complete the greeting flow and allow intent detection
                if step_type == "greeting" and not matching_intent:
                    # Check if user message contains keywords that suggest they want to discuss a topic or speak with someone
                    intent_keywords = ["marketing", "sales", "campaign", "strategy", "business", "service", "help", "information", "about", "speak with", "talk to", "connect", "agent", "human", "someone", "person"]
                    if any(keyword in user_message.lower() for keyword in intent_keywords):
                        logger.info("FLOW_MANAGEMENT: User mentioned topic keywords during greeting flow, completing greeting and allowing intent detection")
                        # Complete the greeting flow
                        flow_state.current_flow = None
                        flow_state.current_step = None
                        flow_state.flow_data = None
                        flow_state.pending_step = None
                        flow_state.pending_expected_kind = None
                        flow_state.pending_asked_at = None
                        auto_save_flow_state()
                        
                        # Now try to detect intent again
                        matching_intent = await detect_flow_intent_with_llm(user_message)
                        if matching_intent and matching_intent.get("type") != "greeting":
                            logger.info(f"FLOW_MANAGEMENT: Intent detected after greeting completion: {matching_intent['intent']}")
                            # Switch to intent flow
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
                
                # No intent shift detected, but let's be more aggressive about intent detection ONLY for greeting flows
                # If we're in a greeting flow and the user says something that sounds like they want to discuss a topic,
                # force intent detection even if the LLM didn't detect it initially
                if step_type == "greeting" and not matching_intent:
                    # Check for agent-related phrases more aggressively
                    agent_phrases = ["speak with", "talk to", "connect", "agent", "human", "someone", "person", "over the phone", "else"]
                    if any(phrase in user_message.lower() for phrase in agent_phrases):
                        logger.info("FLOW_MANAGEMENT: Detected agent-related phrases in greeting flow, forcing agent intent detection")
                        # Force agent intent
                        matching_intent = {
                            "type": "intent_bot",
                            "intent": "agent",
                            "flow_key": "Flow_4",
                            "flow_data": bot_template["data"]["Flow_4"] if bot_template and bot_template.get("data", {}).get("Flow_4") else None
                        }
                        if matching_intent["flow_data"]:
                            logger.info(f"FLOW_MANAGEMENT: Forced agent intent detection for: '{user_message}'")
                            print_flow_status(room_name, flow_state, "🔄 FORCED AGENT INTENT", 
                                            f"From: {flow_state.current_flow} → To: {matching_intent['flow_key']} | Intent: {matching_intent['intent']}")
                            
                            # Update flow state to agent intent
                            flow_state.current_flow = matching_intent["flow_key"]
                            flow_state.current_step = matching_intent["flow_data"]["name"]
                            flow_state.flow_data = matching_intent["flow_data"]
                            auto_save_flow_state()
                            
                            # Execute agent bot
                            return await execute_agent_bot(flow_state, user_message, auto_save_flow_state)
                
                # No intent shift detected, continue with current flow
                logger.info(f"FLOW_MANAGEMENT: No intent shift detected, continuing with {step_type} flow")
                
                # For greeting steps, automatically progress to next step if available
                if step_type == "greeting" and current_step_data.get("next_flow"):
                    logger.info("FLOW_MANAGEMENT: Greeting step - automatically progressing to next step")
                    next_step_data = current_step_data["next_flow"]
                    old_step = flow_state.current_step
                    flow_state.current_step = next_step_data.get("name")
                    flow_state.flow_data = next_step_data
                    step_type = next_step_data.get("type", "unknown")
                    
                    logger.info(f"FLOW_MANAGEMENT: GREETING STEP TRANSITION - From: {old_step} → To: {next_step_data.get('name')} | Type: {step_type}")
                    print_flow_status(room_name, flow_state, f"➡️ GREETING STEP TRANSITION", 
                                    f"From: {old_step} → To: {next_step_data.get('name')} | Type: {step_type} | Response: '{next_step_data.get('text', '')}'")
                    
                    # Handle different step types
                    response_text = next_step_data.get("text", "")
                    # Set pending question lock if next is a question
                    if step_type == 'question':
                        flow_state.pending_step = next_step_data.get('name')
                        flow_state.pending_expected_kind = 'number' if ('phone line' in response_text.lower() or 'texts' in response_text.lower()) else None
                        flow_state.pending_asked_at = time.time()
                    
                    add_agent_response_to_history(flow_state, response_text)
                    auto_save_flow_state()
                    
                    return {
                        "type": "flow_continued",
                        "response": response_text,
                        "next_step": next_step_data.get("next_flow"),
                        "flow_state": flow_state
                    }
                
                # Special handling for greeting flow completion
                if step_type == "greeting" and not current_step_data.get("next_flow"):
                    logger.info("FLOW_MANAGEMENT: Greeting flow completed, resetting to intent detection mode")
                    flow_state.current_flow = None
                    flow_state.current_step = None
                    flow_state.flow_data = None
                    flow_state.pending_step = None
                    flow_state.pending_expected_kind = None
                    flow_state.pending_asked_at = None
                    auto_save_flow_state()
                    
                    response_text = "I'm here to help you with information about our business communication services. What would you like to know about?"
                    add_agent_response_to_history(flow_state, response_text)
                    
                    return {
                        "type": "message",
                        "response": response_text,
                        "flow_state": flow_state
                    }
                
                # Process the user response and move to next step using get_next_flow_step
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
                    
                    add_agent_response_to_history(flow_state, response_text)
                    auto_save_flow_state()
                    
                    return {
                        "type": "flow_continued",
                        "response": response_text,
                        "next_step": next_step["step_data"].get("next_flow"),
                        "flow_state": flow_state
                    }
                else:
                    # No next step found, check if this is a greeting that should continue to intent detection
                    if step_type == "greeting":
                        logger.info("FLOW_MANAGEMENT: Greeting step completed, checking for intent in user response")
                        # Try to detect intent from user response
                        matching_intent = await detect_flow_intent_with_llm(user_message)
                        if matching_intent and matching_intent.get("type") != "greeting":
                            logger.info(f"FLOW_MANAGEMENT: Intent detected after greeting: {matching_intent['intent']}")
                            # Switch to intent flow
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
                        else:
                            # No intent detected, check if we should progress to next step in greeting flow
                            if current_step_data.get("next_flow"):
                                logger.info("FLOW_MANAGEMENT: No intent detected, progressing to next step in greeting flow")
                                next_step_data = current_step_data["next_flow"]
                                flow_state.current_step = next_step_data.get("name")
                                flow_state.flow_data = next_step_data
                                auto_save_flow_state()
                                
                                response_text = next_step_data.get("text", "")
                                add_agent_response_to_history(flow_state, response_text)
                                
                                return {
                                    "type": "flow_continued",
                                    "response": response_text,
                                    "next_step": next_step_data.get("next_flow"),
                                    "flow_state": flow_state
                                }
                            else:
                                # Greeting flow completed - reset to intent detection mode
                                logger.info("FLOW_MANAGEMENT: Greeting flow completed, resetting to intent detection mode")
                                flow_state.current_flow = None
                                flow_state.current_step = None
                                flow_state.flow_data = None
                                flow_state.pending_step = None
                                flow_state.pending_expected_kind = None
                                flow_state.pending_asked_at = None
                                auto_save_flow_state()
                                
                                response_text = "I'm here to help you with information about our business communication services. What would you like to know about?"
                            add_agent_response_to_history(flow_state, response_text)
                            auto_save_flow_state()
                            
                            return {
                                "type": "message",
                                "response": response_text,
                                "flow_state": flow_state
                            }
                    else:
                        # No next step and not a greeting, use FAQ fallback
                        logger.info("FLOW_MANAGEMENT: No next step found, using FAQ fallback")
                        return await get_faq_response(user_message, flow_state=flow_state)
            
            # Handle question steps (existing logic)
            if step_type == "question":
                # Use gated LLM approach for robust answer extraction
                interp = gated_llm_extract_answer(current_step_data.get("text", ""), user_message or "")
                logger.info(f"GATED_LLM_EXTRACT: {interp}")

                # Extra yes/no fallback for special-needs/SSO style questions
                qtxt = (current_step_data.get("text") or "").lower()
                utxt = (user_message or "").lower()
                if any(k in qtxt for k in ["special needs", "sso", "salesforce", "crm integration"]) and re.search(r"\b(yes|yeah|yep|yup|sure|of course|please|ok|okay|absolutely|i need|i would need)\b", utxt):
                    interp = {"status": "extracted", "kind": "yesno", "value": True, "confidence": 0.95}

                # Handle unclear responses in main question flow
                if interp.get("status") == "unclear":
                    if interp.get("kind") == "ambiguous":
                        response_text = "I didn't quite catch that. Could you please repeat your answer more clearly?"
                    else:
                        response_text = "I didn't quite understand that. Could you please repeat your answer?"
                    add_agent_response_to_history(flow_state, response_text)
                    logger.info(f"ANSWER_INTERPRETER: Handling unclear response ({interp.get('kind', 'unclear')}) with clarification request")
                    return {
                        "type": "message",
                        "response": response_text,
                        "flow_state": flow_state
                    }

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
                        logger.info(f"FLOW_MANAGEMENT: Interpreter-based progression applied - extracted {interp.get('kind')}: {interp.get('value')}")
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
                        parts = re.findall(r"\d|zero|one|two|three|four|five|six|seven|eight|nine", t)
                        return "".join(words_map.get(p, p) for p in parts)

                    def _extract_quantity(t: str) -> Optional[int]:
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
                        logger.info(f"FLOW_MANAGEMENT: Heuristic progressed numeric answer ({qty}) to next step {flow_state.current_step}")
                        print_flow_status(room_name, flow_state, "➡️ STEP TRANSITION", f"From: {old_step} → To: {flow_state.current_step} | Type: {step_type} | Response: '{response_text}'")
                        add_agent_response_to_history(flow_state, response_text)
                        return {
                            "type": step_type,
                            "response": response_text,
                            "flow_state": flow_state
                        }

                    # If user utterance is too short/stopwordy, re-ask the same question instead of falling back
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
            logger.info(f"FLOW_MANAGEMENT: Current step is not a question (type: {current_step_data.get('type') if current_step_data else 'None'}), checking for intent shift or answers branch")

            # If current step is a message with a next_flow of type 'faq', auto-transition so that
            # subsequent user utterances evaluate the FAQ node's answers.
            if current_step_data and current_step_data.get("type") == "message":
                nf = current_step_data.get("next_flow")
                # If next_flow is faq, auto-transition
                if isinstance(nf, dict) and nf.get("type") == "faq":
                    old = flow_state.current_step
                    flow_state.current_step = nf.get("name")
                    flow_state.flow_data = nf
                    logger.info(f"FLOW_MANAGEMENT: Auto-transitioned message → faq for answers handling: {old} → {flow_state.current_step}")
                    current_step_data = flow_state.flow_data
                # If there's no explicit next_flow, but the template contains a faq node with the expected text, jump to it
                elif not nf and bot_template:
                    msg_text = (current_step_data.get("text") or "").strip()
                    probe = _find_step_by_text(bot_template, "Feel free to ask any question!")
                    if probe and isinstance(probe.get("node"), dict) and probe["node"].get("type") == "faq":
                        old = flow_state.current_step
                        flow_state.current_step = probe["node"].get("name")
                        flow_state.flow_data = probe["node"]
                        logger.info(f"FLOW_MANAGEMENT: Soft-transitioned message → faq by text match: {old} → {flow_state.current_step}")
                        current_step_data = flow_state.flow_data

            # Also handle if current step IS 'faq' — emit its prompt once so the user hears it
            if current_step_data and current_step_data.get("type") == "faq":
                faq_text = current_step_data.get("text", "")
                if faq_text:
                    # avoid duplicate prompt if it was the previous assistant message
                    last_msg = flow_state.conversation_history[-1]["content"] if flow_state.conversation_history else ""
                    if (last_msg or "").strip().lower() != faq_text.strip().lower():
                        add_agent_response_to_history(flow_state, faq_text)
                        logger.info("FLOW_MANAGEMENT: Emitting FAQ prompt to user")
                        # Return the prompt so the agent actually says it; answers will be evaluated on next user turn
                        return {
                            "type": "message",
                            "response": faq_text,
                            "flow_state": flow_state
                        }

            # Handle Agent Bot - Human agent handoff
            if current_step_data and current_step_data.get("type") == "agent":
                logger.info("FLOW_MANAGEMENT: Agent bot step detected")
                print_flow_status(room_name, flow_state, "👤 AGENT HANDOFF", f"Agent: '{current_step_data.get('text', '')}'")
                
                # Agent bot response
                agent_response = current_step_data.get("text", "I'm connecting you with a human agent. Please hold on.")
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
                print_flow_status(room_name, flow_state, "⚡ ACTION BOT", f"Action: '{current_step_data.get('text', '')}'")
                
                # Execute action bot functionality
                action_result = await execute_action_bot(current_step_data, flow_state)
                add_agent_response_to_history(flow_state, action_result["response"])
                auto_save_flow_state()
                
                return {
                    "type": "action_completed",
                    "response": action_result["response"],
                    "flow_state": flow_state,
                    "action_data": action_result.get("action_data")
                }
            
            # Handle Condition Bot - Variable-based routing
            if current_step_data and current_step_data.get("type") == "condition":
                logger.info("FLOW_MANAGEMENT: Condition bot step detected")
                print_flow_status(room_name, flow_state, "🔀 CONDITION BOT", f"Condition: '{current_step_data.get('text', '')}'")
                
                # Evaluate condition and route accordingly
                condition_result = await evaluate_condition_bot(current_step_data, flow_state, user_message)
                add_agent_response_to_history(flow_state, condition_result["response"])
                auto_save_flow_state()
                
                return {
                    "type": "condition_evaluated",
                    "response": condition_result["response"],
                    "flow_state": flow_state,
                    "condition_result": condition_result.get("condition_result")
                        }

            # Handle template 'answers' on FAQ/message steps (noAction / moreAction)
            if current_step_data and current_step_data.get("answers") and current_step_data.get("type") in ("faq", "message"):
                answers = current_step_data.get("answers", {}) or {}
                um = (user_message or "").lower().strip()

                # Check for escalation phrases in FAQ step (escalation should work everywhere)
                if _matches_any(escalate_phrases, um):
                    response_text = "I'm connecting you with a human agent. Please hold on."
                    add_agent_response_to_history(flow_state, response_text)
                    logger.info("FLOW_MANAGEMENT: Escalation detected in FAQ step → initiating agent handoff")
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

                # For FAQ steps, treat user responses as general questions and use FAQ bot
                logger.info("FAQ_ANSWER_INTERPRETER: User response in FAQ step, routing to FAQ bot")
                return await get_faq_response(user_message, flow_state=flow_state)

                # ... existing code ...
    
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
    print(f"❌ FALLBACK TO FAQ: No intent found for '{user_message}'")
    return await get_faq_response(user_message, flow_state=flow_state)

async def get_faq_response(user_message: str, bot_id: str = None, flow_state: FlowState = None) -> Dict[str, Any]:
    """Get response from FAQ bot - supports dynamic bot IDs"""
    try:
        logger.info(f"FAQ_RESPONSE: Called with message: '{user_message}', bot_id: {bot_id}")
        if flow_state:
            logger.info(f"FAQ_RESPONSE: Flow state - current_flow: {flow_state.current_flow}, current_step: {flow_state.current_step}")
        
        # Use provided bot_id or an explicit default from env/constant (template 'name' is NOT a bot_id)
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
            if not result or not result.get("data") or not result["data"].get("answer"):
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
            
            # Add agent response to conversation history if flow_state is provided
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

async def execute_agent_bot(flow_state: FlowState, user_message: str, auto_save_flow_state=None) -> Dict[str, Any]:
    """Execute agent bot functionality - transfer to human agent"""
    try:
        logger.info(f"AGENT_BOT: Executing agent transfer for message: '{user_message}'")
        
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
        agent_response = agent_flow_data.get("text", "I'm connecting you to a human agent who can help you better.")
        
        # Add agent response to history
        add_agent_response_to_history(flow_state, agent_response)
        
        # Check if there's a next flow to transition to
        next_flow = agent_flow_data.get("next_flow")
        if next_flow:
            logger.info(f"AGENT_BOT: Transitioning to next flow: {next_flow.get('name', 'unknown')}")
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

async def execute_action_bot(action_data: Dict[str, Any], flow_state: FlowState) -> Dict[str, Any]:
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

async def evaluate_condition_bot(condition_data: Dict[str, Any], flow_state: FlowState, user_message: str) -> Dict[str, Any]:
    """Evaluate condition bot and route accordingly"""
    try:
        condition_type = condition_data.get("condition_type", "variable")
        condition_text = condition_data.get("text", "Condition evaluated.")
        
        logger.info(f"CONDITION_BOT: Evaluating condition type: {condition_type}")
        
        if condition_type == "variable":
            # Check variable value from flow state
            variable_name = condition_data.get("variable_name")
            expected_value = condition_data.get("expected_value")
            
            if variable_name and flow_state.user_responses:
                actual_value = flow_state.user_responses.get(variable_name, "")
                condition_met = str(actual_value).lower() == str(expected_value).lower()
                
                if condition_met:
                    # Condition met - follow true path
                    next_flow = condition_data.get("true_flow")
                    response = condition_data.get("true_response", "Condition met.")
                else:
                    # Condition not met - follow false path
                    next_flow = condition_data.get("false_flow")
                    response = condition_data.get("false_response", "Condition not met.")
                
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
                response = condition_data.get("true_response", "Input matches condition.")
            else:
                next_flow = condition_data.get("false_flow")
                response = condition_data.get("false_response", "Input doesn't match condition.")
            
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
        
        # Process through flow system with conversation history and custom config
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
async def refresh_template():
    """Refresh the bot template from Alive5 API using TemplateManager"""
    try:
        if not template_manager:
            raise HTTPException(status_code=500, detail="Template manager not initialized")
        
        success = await template_manager.fetch_and_store_template()
        
        if success:
            # Update global bot_template for backward compatibility
            global bot_template
            bot_template = template_manager.template_data
            
            return {
                "status": "success",
                "message": "Template refreshed successfully",
                "template_version": bot_template.get("code", "unknown"),
                "template_hash": template_manager.template_hash[:8] + "...",
                "last_updated": template_manager.last_updated.isoformat()
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

@app.get("/api/template_status")
async def get_template_status():
    """Get current template status and polling info"""
    if not template_manager:
        return {
            "status": "not_initialized",
            "message": "Template manager not initialized"
        }
    
    status = template_manager.get_status()
    
    # Add additional info
    status.update({
        "template_available": bot_template is not None,
        "available_intents": list(bot_template.get("data", {}).keys()) if bot_template else []
    })
    
    return status

@app.post("/api/force_template_update")
async def force_template_update():
    """Manually trigger template update (POST method)"""
    if not template_manager:
        raise HTTPException(status_code=500, detail="Template manager not initialized")
    
    try:
        success = await template_manager.fetch_and_store_template()
        
        if success:
            # Update global bot_template for backward compatibility
            global bot_template
            bot_template = template_manager.template_data
            
            return {
                "success": True,
                "message": "Template updated successfully",
                "timestamp": datetime.now().isoformat(),
                "template_hash": template_manager.template_hash[:8] + "...",
                "last_updated": template_manager.last_updated.isoformat()
            }
        else:
            return {
                "success": False,
                "message": "Failed to update template",
                "timestamp": datetime.now().isoformat()
            }
            
    except Exception as e:
        logger.error(f"Force template update error: {e}")
        raise HTTPException(status_code=500, detail=f"Template update failed: {str(e)}")

@app.get("/api/force_template_update")
async def force_template_update_get():
    """Manually trigger template update (GET method for easy browser access)"""
    if not template_manager:
        raise HTTPException(status_code=500, detail="Template manager not initialized")
    
    try:
        success = await template_manager.fetch_and_store_template()
        
        if success:
            # Update global bot_template for backward compatibility
            global bot_template
            bot_template = template_manager.template_data
            
            return {
                "success": True,
                "message": "Template updated successfully",
                "timestamp": datetime.now().isoformat(),
                "template_hash": template_manager.template_hash[:8] + "...",
                "last_updated": template_manager.last_updated.isoformat()
            }
        else:
            return {
                "success": False,
                "message": "Failed to update template",
                "timestamp": datetime.now().isoformat()
            }
            
    except Exception as e:
        logger.error(f"Force template update error: {e}")
        raise HTTPException(status_code=500, detail=f"Template update failed: {str(e)}")

# Polling endpoints removed - templates loaded on-demand

@app.get("/api/get_greeting")
async def get_greeting():
    """Get greeting from template if available"""
    try:
        global bot_template
        
        # Check if template is loaded
        if not bot_template or not bot_template.get("data"):
            return {
                "greeting_available": False,
                "greeting_text": None,
                "message": "No template loaded"
            }
        
        # Look for greeting bot in template
        for flow_key, flow_data in bot_template["data"].items():
            if flow_data.get("type") == "greeting":
                greeting_text = flow_data.get("text", "")
                logger.info(f"🎯 GREETING API: Found greeting bot: {flow_key} - '{greeting_text}'")
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
                        logger.info(f"🎯 GREETING FLOW INIT: Found greeting bot {flow_key} with text: {template_text}")
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
        flow_state.current_step = greeting_flow_data.get("name", greeting_flow_key)
        flow_state.flow_data = greeting_flow_data
        
        # Add greeting to conversation history
        add_agent_response_to_history(flow_state, greeting_text)
        
        # Save flow state
        save_flow_state_to_file(room_name, flow_state)
        
        logger.info(f"🎯 GREETING FLOW INIT: Initialized greeting flow {greeting_flow_key} for room {room_name}")
        print(f"🎯 GREETING FLOW INIT: Room {room_name} -> Flow: {greeting_flow_key}, Step: {flow_state.current_step}")
        
        return {
            "success": True,
            "flow_key": greeting_flow_key,
            "flow_data": greeting_flow_data
        }
        
    except Exception as e:
        logger.error(f"🎯 GREETING FLOW INIT: Error initializing greeting flow: {e}")
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
    
    result = await detect_flow_intent_with_llm(user_message)
    
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