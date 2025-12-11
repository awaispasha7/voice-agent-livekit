"""
Alive5 Simple Voice Agent Backend
Simplified backend for the simple-agent worker and frontend
"""

import os
import json
import logging
import asyncio
import uuid
import subprocess
from pathlib import Path
from typing import Dict, Any, Optional
from datetime import datetime
from contextlib import asynccontextmanager

import httpx
import uvicorn
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from livekit import api

# Try to import psutil for better resource monitoring
try:
    import psutil
    PSUTIL_AVAILABLE = True
except ImportError:
    PSUTIL_AVAILABLE = False

# Load environment variables
current_dir = Path(__file__).parent
env_paths = [
    current_dir / "../../.env",
    current_dir / "../../../.env",
    Path("/home/ubuntu/alive5-voice-agent/.env"),
    Path(".env"),
]

env_loaded = False
for env_path in env_paths:
    if env_path.exists():
        load_dotenv(dotenv_path=str(env_path), override=True)
        env_loaded = True
        break
if not env_loaded:
    load_dotenv()

# Setup logging
logging.basicConfig(level=logging.INFO, format="%(message)s", force=True)
logger = logging.getLogger("simple-backend")

# Resource monitoring functions
def get_system_resources():
    """Get CPU and RAM usage"""
    try:
        if PSUTIL_AVAILABLE:
            # Use psutil for accurate monitoring
            cpu_percent = psutil.cpu_percent(interval=0.1)
            memory = psutil.virtual_memory()
            return {
                "cpu_percent": cpu_percent,
                "memory_total_gb": round(memory.total / (1024**3), 2),
                "memory_used_gb": round(memory.used / (1024**3), 2),
                "memory_available_gb": round(memory.available / (1024**3), 2),
                "memory_percent": memory.percent
            }
        else:
            # Fallback to system commands
            # Get CPU load
            try:
                cpu_result = subprocess.run(
                    ["uptime"], capture_output=True, text=True, timeout=2
                )
                if cpu_result.returncode == 0:
                    # Parse load average from uptime output
                    load_avg = cpu_result.stdout.split("load average:")[-1].strip().split(",")[0].strip()
                    cpu_percent = float(load_avg) * 100  # Rough estimate
                else:
                    cpu_percent = 0.0
            except:
                cpu_percent = 0.0
            
            # Get memory
            try:
                mem_result = subprocess.run(
                    ["free", "-m"], capture_output=True, text=True, timeout=2
                )
                if mem_result.returncode == 0:
                    lines = mem_result.stdout.strip().split("\n")
                    mem_line = lines[1].split()
                    total_mb = int(mem_line[1])
                    used_mb = int(mem_line[2])
                    available_mb = int(mem_line[6]) if len(mem_line) > 6 else total_mb - used_mb
                    memory_percent = (used_mb / total_mb) * 100 if total_mb > 0 else 0
                    
                    return {
                        "cpu_percent": cpu_percent,
                        "memory_total_gb": round(total_mb / 1024, 2),
                        "memory_used_gb": round(used_mb / 1024, 2),
                        "memory_available_gb": round(available_mb / 1024, 2),
                        "memory_percent": round(memory_percent, 1)
                    }
            except:
                pass
            
            return {
                "cpu_percent": cpu_percent,
                "memory_total_gb": 0,
                "memory_used_gb": 0,
                "memory_available_gb": 0,
                "memory_percent": 0
            }
    except Exception as e:
        logger.warning(f"Failed to get system resources: {e}")
        return None

async def monitor_resources():
    """Background task to monitor and log system resources"""
    logger.info("📊 Resource monitoring started (logging every 30 seconds)")
    
    while True:
        try:
            await asyncio.sleep(30)  # Log every 30 seconds
            resources = get_system_resources()
            
            if resources:
                # Format memory status
                mem_status = "🟢 OK"
                if resources["memory_percent"] > 90:
                    mem_status = "🔴 CRITICAL"
                elif resources["memory_percent"] > 80:
                    mem_status = "🟡 WARNING"
                
                # Format CPU status
                cpu_status = "🟢 OK"
                if resources["cpu_percent"] > 80:
                    cpu_status = "🟡 HIGH"
                
                logger.info(
                    f"📊 RESOURCES | CPU: {resources['cpu_percent']:.1f}% {cpu_status} | "
                    f"RAM: {resources['memory_used_gb']:.2f}GB/{resources['memory_total_gb']:.2f}GB "
                    f"({resources['memory_percent']:.1f}%) {mem_status} | "
                    f"Available: {resources['memory_available_gb']:.2f}GB"
                )
                
                # Log warning if memory is critical
                if resources["memory_percent"] > 90:
                    logger.warning(
                        f"⚠️ CRITICAL: Memory usage is {resources['memory_percent']:.1f}%! "
                        f"Only {resources['memory_available_gb']:.2f}GB available. "
                        f"Server may become unresponsive. Consider disabling VAD (USE_VAD=false) or upgrading instance."
                    )
            else:
                logger.warning("⚠️ Could not retrieve system resources")
                
        except asyncio.CancelledError:
            logger.info("📊 Resource monitoring stopped")
            break
        except Exception as e:
            logger.error(f"Error in resource monitoring: {e}")

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifespan context manager for startup and shutdown"""
    # Startup
    logger.info("🚀 Backend starting up...")
    
    # Start resource monitoring task
    monitor_task = asyncio.create_task(monitor_resources())
    
    yield
    
    # Shutdown
    logger.info("🛑 Backend shutting down...")
    monitor_task.cancel()
    try:
        await monitor_task
    except asyncio.CancelledError:
        pass

# FastAPI app with lifespan
app = FastAPI(
    title="Alive5 Simple Voice Agent Backend",
    version="1.0",
    lifespan=lifespan
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Pydantic models
class ConnectionDetailsRequest(BaseModel):
    room_name: str
    user_name: str = "User"
    botchain_name: str = "voice-1"
    org_name: str = "alive5stage0"
    faq_isVoice: bool = True
    selected_voice: Optional[str] = None
    faq_bot_id: Optional[str] = None
    special_instructions: Optional[str] = None

class SessionUpdateRequest(BaseModel):
    room_name: str
    user_data: Dict[str, Any] = {}

class VoiceChangeRequest(BaseModel):
    room_name: str
    voice_id: str

class CRMSubmissionRequest(BaseModel):
    room_name: str
    botchain_name: str
    org_name: str
    full_name: Optional[str] = None
    email: Optional[str] = None
    phone: Optional[str] = None
    notes: Optional[str] = None

class TelnyxWebhookRequest(BaseModel):
    """Telnyx webhook payload structure"""
    data: Dict[str, Any]

class TelnyxTransferRequest(BaseModel):
    """Request to transfer a Telnyx call"""
    room_name: str
    call_control_id: str
    transfer_to: str


# In-memory storage (simple approach)
sessions: Dict[str, Dict[str, Any]] = {}
rooms: Dict[str, Dict[str, Any]] = {}

# Load available voices from cached_voices.json
def load_available_voices():
    """Load voices from cached_voices.json or fallback to minimal list"""
    voices_path = current_dir / "cached_voices.json"
    if voices_path.exists():
        try:
            with open(voices_path, "r") as f:
                voices_dict = json.load(f)
            # Return as dictionary format for frontend compatibility
            return voices_dict
        except Exception as e:
            logger.error(f"Failed to load cached_voices.json: {e}")
    
    # Fallback minimal voices as dictionary
    return {
        "f114a467-c40a-4db8-964d-aaba89cd08fa": "Miles - Yogi",
        "98a34ef2-2140-4c28-9c71-663dc4dd7022": "Clyde - Calm Narrator", 
        "c99d36f3-5ffd-4253-803a-535c1bc9c306": "Griffin - Narrator",
    }

AVAILABLE_VOICES = load_available_voices()

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {"status": "healthy", "timestamp": datetime.now().isoformat()}

@app.get("/api/available_voices")
async def get_available_voices():
    """Get available voices for the frontend"""
    return {"status": "success", "voices": AVAILABLE_VOICES}

@app.options("/api/available_voices")
async def options_available_voices():
    return {"message": "OK"}

@app.options("/api/change_voice")
async def options_change_voice():
    return {"message": "OK"}

@app.post("/api/connection_details")
async def get_connection_details(request: ConnectionDetailsRequest):
    """Get LiveKit connection details for the frontend"""
    try:
        # Create or get room
        room_name = request.room_name
        
        # Use provided values from frontend or defaults
        selected_voice_id = request.selected_voice or list(AVAILABLE_VOICES.keys())[0]
        selected_voice_name = AVAILABLE_VOICES.get(selected_voice_id, "Unknown Voice")
        faq_bot_id = request.faq_bot_id or "faq_b9952a56-fc7b-41c9-b0a0-5c662ddb039e"
        org_name = request.org_name or "alive5stage0"
        
        logger.info(f"🌐 Web session configuration from frontend:")
        logger.info(f"   - Org Name: {org_name}")
        logger.info(f"   - FAQ Bot ID: {faq_bot_id}")
        logger.info(f"   - Botchain: {request.botchain_name}")
        
        # Get LiveKit credentials
        livekit_url = os.getenv("LIVEKIT_URL")
        livekit_api_key = os.getenv("LIVEKIT_API_KEY")
        livekit_api_secret = os.getenv("LIVEKIT_API_SECRET")
        
        if not all([livekit_url, livekit_api_key, livekit_api_secret]):
            raise HTTPException(status_code=500, detail="LiveKit configuration missing")
        
        # CRITICAL: Create the LiveKit room BEFORE returning connection details
        # This ensures the room exists and the worker can be dispatched when the frontend connects
        # This is the same approach used for phone calls (line 593)
        async with api.LiveKitAPI(livekit_url, livekit_api_key, livekit_api_secret) as lk_api:
            try:
                # Create room (or get existing room)
                room = await lk_api.room.create_room(
                    api.CreateRoomRequest(
                        name=room_name,
                        empty_timeout=300,  # 5 minutes
                        max_participants=10  # Allow multiple participants for web sessions
                    )
                )
                logger.info(f"✅ Created LiveKit room for web session: {room_name}")
            except Exception as room_error:
                # Room might already exist, which is fine
                logger.debug(f"Room {room_name} might already exist: {room_error}")
                # Try to get existing room info
                try:
                    rooms = await lk_api.room.list_rooms(api.ListRoomsRequest(names=[room_name]))
                    if rooms.rooms:
                        logger.info(f"✅ Room {room_name} already exists")
                    else:
                        logger.warning(f"⚠️ Could not create or find room {room_name}, but continuing...")
                except Exception as list_error:
                    logger.warning(f"⚠️ Could not verify room existence: {list_error}, but continuing...")
        
        # Store session data
        sessions[room_name] = {
            "room_name": room_name,
            "user_name": request.user_name,
            "user_data": {
                "botchain_name": request.botchain_name,
                "org_name": org_name,
                "faq_isVoice": request.faq_isVoice,
                "selected_voice": selected_voice_id,
                "selected_voice_name": selected_voice_name,
                "faq_bot_id": faq_bot_id,
                "special_instructions": request.special_instructions or ""
            },
            "created_at": datetime.now().isoformat()
        }
        
        # Create room token
        token = api.AccessToken(livekit_api_key, livekit_api_secret)
        token.with_identity(request.user_name)
        token.with_name(request.user_name)
        token.with_grants(api.VideoGrants(
            room_join=True,
            room=room_name,
            can_publish=True,
            can_subscribe=True,
            can_publish_data=True
        ))
        
        return {
            "url": livekit_url,
            "token": token.to_jwt(),
            "room_name": room_name
        }
        
    except Exception as e:
        logger.error(f"Error getting connection details: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/sessions/{room_name}")
async def get_session(room_name: str):
    """Get session data for the simple-agent worker"""
    # FastAPI automatically URL-decodes the path parameter
    from urllib.parse import unquote
    
    # URL-decode the room name in case it's encoded
    room_name_decoded = unquote(room_name)
    
    # Check for double prefix and clean it up
    room_name_clean = room_name_decoded
    if room_name_clean.startswith("telnyx_call__telnyx_call_"):
        # Remove double prefix
        room_name_clean = room_name_clean.replace("telnyx_call__telnyx_call_", "telnyx_call_", 1)
        logger.info(f"⚠️ Fixed double-prefixed room name in session lookup: {room_name_decoded} -> {room_name_clean}")
    
    # Log for debugging
    logger.debug(f"📋 Session lookup - Room name: {room_name_decoded}, Cleaned: {room_name_clean}")
    logger.debug(f"   Available sessions: {list(sessions.keys())[:5]}...")  # Show first 5
    
    # Try cleaned name first
    if room_name_clean in sessions:
        return sessions[room_name_clean]
    
    # Try original decoded name
    if room_name_decoded in sessions:
        return sessions[room_name_decoded]
    
    # Try original (might be encoded)
    if room_name in sessions:
        return sessions[room_name]
    
    # Try to find a matching session (in case of encoding issues)
    for key in sessions.keys():
        # Check if keys match after cleaning
        key_clean = key
        if key_clean.startswith("telnyx_call__telnyx_call_"):
            key_clean = key_clean.replace("telnyx_call__telnyx_call_", "telnyx_call_", 1)
        
        if key_clean == room_name_clean or key == room_name_decoded:
            logger.info(f"✅ Found session by matching: {key}")
            return sessions[key]
        
        # Also try URL encoding/decoding variations
        if key.replace(':', '%3A') == room_name or room_name.replace('%3A', ':') == key:
            logger.info(f"✅ Found session by encoding match: {key}")
            return sessions[key]
    
    raise HTTPException(status_code=404, detail=f"Session not found: {room_name} (tried: {room_name_clean}, {room_name_decoded})")

@app.post("/api/sessions/update")
async def update_session(request: SessionUpdateRequest):
    """Update session data"""
    if request.room_name not in sessions:
        raise HTTPException(status_code=404, detail="Session not found")
    
    sessions[request.room_name]["user_data"].update(request.user_data)
    sessions[request.room_name]["updated_at"] = datetime.now().isoformat()
    
    return {"status": "updated"}

@app.post("/api/change_voice")
async def change_voice(request: VoiceChangeRequest):
    """Change voice for a session"""
    logger.info(f"Voice change request: room={request.room_name}, voice={request.voice_id}")
    
    if request.room_name not in sessions:
        raise HTTPException(status_code=404, detail="Session not found")
    
    # Validate voice_id exists
    if request.voice_id not in AVAILABLE_VOICES:
        raise HTTPException(status_code=400, detail="Invalid voice ID")
    
    sessions[request.room_name]["user_data"]["selected_voice"] = request.voice_id
    sessions[request.room_name]["selected_voice"] = request.voice_id
    sessions[request.room_name]["voice_id"] = request.voice_id
    sessions[request.room_name]["updated_at"] = datetime.now().isoformat()
    
    return {"status": "success", "voice_name": AVAILABLE_VOICES[request.voice_id]}

@app.delete("/api/rooms/{room_name}")
async def delete_room(room_name: str):
    """Delete room and clean up session - closes LiveKit room to notify worker"""
    try:
        logger.info(f"🗑️ Deleting room: {room_name}")
        
        # Close LiveKit room to trigger worker cleanup
        try:
            livekit_url = os.getenv("LIVEKIT_URL")
            livekit_api_key = os.getenv("LIVEKIT_API_KEY")
            livekit_api_secret = os.getenv("LIVEKIT_API_SECRET")
            
            if all([livekit_url, livekit_api_key, livekit_api_secret]):
                async with api.LiveKitAPI(livekit_url, livekit_api_key, livekit_api_secret) as lk_api:
                    try:
                        await lk_api.room.delete_room(api.DeleteRoomRequest(room=room_name))
                        logger.info(f"✅ LiveKit room closed: {room_name}")
                    except Exception as e:
                        # Room might already be closed, that's fine
                        if "not_found" in str(e).lower() or "does not exist" in str(e).lower():
                            logger.debug(f"ℹ️ Room {room_name} already closed")
                        else:
                            logger.warning(f"⚠️ Could not close LiveKit room: {e}")
            else:
                logger.warning("⚠️ LiveKit credentials not configured - cannot close room")
        except Exception as e:
            logger.warning(f"⚠️ Error closing LiveKit room: {e}")
        
        # Clean up session data
        if room_name in sessions:
            del sessions[room_name]
        if room_name in rooms:
            del rooms[room_name]
        
        logger.info(f"✅ Room deleted and cleaned up: {room_name}")
        return {"status": "deleted", "room_name": room_name}
        
    except Exception as e:
        logger.error(f"❌ Error deleting room: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/init_livechat")
async def init_livechat(room_name: str, org_name: str, botchain_name: str):
    """Initialize voice agent session - sends socket config to frontend via response
    
    Frontend will establish socket connection (whitelisted origin) and emit init_voice_agent
    """
    try:
        logger.info(f"🚀 Initializing voice agent session for room: {room_name}")
        
        # Get API key and widget ID from environment
        api_key = os.getenv("A5_API_KEY")
        widget_id = os.getenv("A5_WIDGET_ID")
        if not api_key:
            raise Exception("A5_API_KEY not configured in .env")
        if not widget_id:
            raise Exception("A5_WIDGET_ID not configured in .env")
        
        # Verify API key is loaded
        logger.info(f"✅ API key loaded: {api_key[:8]}...{api_key[-4:] if len(api_key) > 12 else '***'}")
        
        # Get widget data to get channel_id and crm_id
        widget_api_url = f"https://api-v2-stage.alive5.com/1.0/widget-code/get-by-widget-id"
        
        async with httpx.AsyncClient() as client:
            response = await client.get(widget_api_url, params={"id": widget_id})
            if response.status_code != 200:
                raise Exception(f"Failed to get widget data: {response.status_code}")
            
            widget_data = response.json()
            data = widget_data.get("data", widget_data)
            channel_id = data.get("channel_id")
            crm_id = data.get("crm_id") or ""
            
            if not channel_id:
                raise Exception("channel_id not found in widget data")
        
        # Generate new thread_id for this session (standard UUID format)
        import uuid
        final_thread_id = str(uuid.uuid4())  # Format: xxxxxxxx-xxxx-xxxx-xxxx-xxxxxxxxxxxx
        
        # Use CRM ID from widget API or generate new one (standard UUID format)
        final_crm_id = crm_id
        if not final_crm_id:
            final_crm_id = str(uuid.uuid4())  # Standard UUID format: xxxxxxxx-xxxx-xxxx-xxxx-xxxxxxxxxxxx
        
        # Store in session for worker to use
        if room_name in sessions:
            sessions[room_name]["thread_id"] = final_thread_id
            sessions[room_name]["crm_id"] = final_crm_id
            sessions[room_name]["api_key"] = api_key
            sessions[room_name]["channel_id"] = channel_id
            sessions[room_name]["widget_id"] = widget_id
        
        logger.info(f"✅ Voice agent session configured - Thread: {final_thread_id[:20]}..., CRM: {final_crm_id[:20]}...")
        
        # Return socket config for frontend to connect
        return {
            "status": "success",
            "thread_id": final_thread_id,
            "crm_id": final_crm_id,
            "channel_id": channel_id,
            "api_key": api_key,
            "socket_config": {
                "thread_id": final_thread_id,
                "crm_id": final_crm_id,
                "channel_id": channel_id,
                "api_key": api_key,
                "org_name": org_name  # Include org_name for end_voice_chat
            }
        }
        
    except Exception as e:
        logger.error(f"❌ Error initializing voice agent session: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/submit_crm")
async def submit_crm(request: CRMSubmissionRequest):
    """Submit collected CRM data via voice agent socket save_crm_data events
    
    Uses existing thread_id and crm_id from init_livechat
    """
    try:
        logger.info(f"📝 CRM submission received for room: {request.room_name}")
        logger.info(f"   Name: {request.full_name}")
        logger.info(f"   Email: {request.email}")
        logger.info(f"   Notes: {request.notes}")
        
        # Get session data (includes api_key, thread_id, crm_id from init)
        session = sessions.get(request.room_name, {})
        api_key = session.get("api_key") or os.getenv("A5_API_KEY")
        thread_id = session.get("thread_id")
        crm_id = session.get("crm_id")
        channel_id = session.get("channel_id")
        
        if not api_key:
            raise Exception("A5_API_KEY not configured - cannot submit CRM data")
        if not crm_id:
            raise Exception("Session not initialized - call init_livechat first")
        
        # Store CRM data in session (worker will send via frontend socket)
        if "crm_data" not in sessions[request.room_name]:
            sessions[request.room_name]["crm_data"] = {}
        
        sessions[request.room_name]["crm_data"].update({
            "full_name": request.full_name,
            "email": request.email,
            "phone": request.phone,
            "notes": request.notes,
            "submitted_at": datetime.now().isoformat()
        })
        
        return {
            "status": "success",
            "message": "CRM data stored - will be sent via frontend socket",
            "thread_id": thread_id,
            "crm_id": crm_id
        }
        
    except Exception as e:
        logger.error(f"❌ Error submitting CRM data: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/end_livechat")
async def end_livechat(room_name: str):
    """End livechat session with visitor-chat-end event"""
    try:
        logger.info(f"👋 Ending livechat for room: {room_name}")
        
        session = sessions.get(room_name, {})
        thread_id = session.get("thread_id")
        crm_id = session.get("crm_id")
        
        if not thread_id:
            logger.warning("No active session to end")
            return {"status": "no_session"}
        
        # Clean up session
        if room_name in sessions:
            pass  # Session cleanup handled by session expiration
        
        logger.info(f"✅ Voice agent session ended - Thread: {thread_id}, CRM: {crm_id}")
        
        return {"status": "success", "message": "Session ended"}
        
    except Exception as e:
        logger.error(f"❌ Error ending livechat: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/telnyx/webhook")
async def telnyx_webhook(request: Request):
    """Handle Telnyx webhook events for incoming calls
    
    This endpoint receives call events from Telnyx and creates LiveKit SIP participants
    """
    try:
        payload = await request.json()
        
        # Try different payload structures (API v1 vs v2)
        data = payload.get("data") or payload.get("payload") or payload
        event_type = data.get("event_type") or payload.get("event_type")
        
        logger.info(f"📞 Telnyx webhook: {event_type}")
        
        if event_type == "call.initiated":
            # Incoming call - create LiveKit room and SIP participant
            # Telnyx webhook structure: data.payload contains the actual call data
            payload_data = data.get("payload") or data
            
            # Extract call_control_id from payload (not from event id)
            call_control_id = payload_data.get("call_control_id") or payload_data.get("id")
            caller_number = payload_data.get("from") or payload_data.get("from_number") or payload_data.get("caller_number") or payload_data.get("caller_id_number")
            called_number = payload_data.get("to") or payload_data.get("to_number") or payload_data.get("called_number") or payload_data.get("called_id_number")
            direction = payload_data.get("direction")  # "inbound" or "outbound"
            
            # If still None, try top-level data
            if not call_control_id:
                call_control_id = data.get("call_control_id") or data.get("id")
            if not caller_number:
                caller_number = data.get("from") or data.get("from_number")
            if not called_number:
                called_number = data.get("to") or data.get("to_number")
            if not direction:
                direction = data.get("direction")
            
            logger.info(f"📞 Call: {caller_number} → {called_number} (ID: {call_control_id[:20]}...)")
            
            # Only process inbound calls - skip outbound calls (these are our dial-out calls to LiveKit)
            # Telnyx uses "incoming" for inbound and "outgoing" for outbound
            # Also check if "to" is a SIP URI to our LiveKit domain - that's definitely an outbound call we initiated
            livekit_sip_domain = os.getenv("LIVEKIT_SIP_DOMAIN", "")
            is_outbound_call = (
                direction in ["outbound", "outgoing"] or
                (called_number and livekit_sip_domain and livekit_sip_domain in str(called_number))
            )
            
            if is_outbound_call:
                logger.info(f"⏭️ Skipping outbound call (direction={direction}, to={called_number}) - this is our dial-out to LiveKit")
                return {"status": "ok"}
            
            # Validate we have call_control_id (only for inbound calls)
            if not call_control_id:
                logger.error(f"❌ Missing call_control_id in webhook payload. Full payload: {payload}")
                raise HTTPException(status_code=400, detail="Missing call_control_id in webhook payload")
            
            # Create unique room name for this call
            # Replace colons with dashes to avoid URL-encoding issues in SIP URI
            # This prevents duplicate sessions when LiveKit dispatch rule extracts room name
            safe_call_control_id = call_control_id.replace(':', '-')
            room_name = f"telnyx_call_{safe_call_control_id}"
            
            # Get LiveKit credentials
            livekit_url = os.getenv("LIVEKIT_URL")
            livekit_api_key = os.getenv("LIVEKIT_API_KEY")
            livekit_api_secret = os.getenv("LIVEKIT_API_SECRET")
            livekit_sip_domain = os.getenv("LIVEKIT_SIP_DOMAIN")
            
            if not all([livekit_url, livekit_api_key, livekit_api_secret, livekit_sip_domain]):
                raise HTTPException(status_code=500, detail="LiveKit SIP configuration missing")
            
            # Create LiveKit room
            try:
                async with api.LiveKitAPI(livekit_url, livekit_api_key, livekit_api_secret) as lk_api:
                    # Create room
                    room = await lk_api.room.create_room(
                        api.CreateRoomRequest(
                            name=room_name,
                            empty_timeout=300,  # 5 minutes
                            max_participants=2
                        )
                    )
                    logger.info(f"✅ Created LiveKit room: {room_name}")
                    
                    # Note: For inbound calls, LiveKit SIP trunk will automatically connect
                    # the call to this room if trunk is configured with matching room name pattern.
                    # No need to create SIP participant manually for inbound calls.
                    logger.info(f"✅ Room created - LiveKit SIP trunk will auto-connect call to room")
                
                # Get default voice from env or use first available voice
                default_voice_id = os.getenv("TELNYX_DEFAULT_VOICE")
                if default_voice_id and default_voice_id in AVAILABLE_VOICES:
                    selected_voice_id = default_voice_id
                    selected_voice_name = AVAILABLE_VOICES[default_voice_id]
                    logger.info(f"🎤 Using configured voice from TELNYX_DEFAULT_VOICE: {selected_voice_name} ({selected_voice_id})")
                else:
                    # Fallback to first available voice
                    selected_voice_id = list(AVAILABLE_VOICES.keys())[0]
                    selected_voice_name = AVAILABLE_VOICES.get(selected_voice_id, "Unknown")
                    if default_voice_id:
                        logger.warning(f"⚠️ TELNYX_DEFAULT_VOICE '{default_voice_id}' not found in available voices, using default: {selected_voice_name}")
                    else:
                        logger.info(f"🎤 Using default voice (first available): {selected_voice_name} ({selected_voice_id})")
                
                # Get phone call configuration from .env
                phone_org_name = os.getenv("TELNYX_DEFAULT_ORG", "alive5stage0")
                phone_faq_bot_id = os.getenv("TELNYX_DEFAULT_FAQ_BOT", "faq_b9952a56-fc7b-41c9-b0a0-5c662ddb039e")
                phone_botchain = os.getenv("TELNYX_DEFAULT_BOTCHAIN", "voice-1")
                
                logger.info(f"📞 Phone call configuration from .env:")
                logger.info(f"   - Org Name: {phone_org_name}")
                logger.info(f"   - FAQ Bot ID: {phone_faq_bot_id}")
                logger.info(f"   - Botchain: {phone_botchain}")
                
                # Store call session
                session_data = {
                    "room_name": room_name,
                    "user_name": f"Caller_{caller_number}",
                    "call_control_id": call_control_id,
                    "caller_number": caller_number,
                    "called_number": called_number,
                    "user_data": {
                        "botchain_name": phone_botchain,
                        "org_name": phone_org_name,
                        "faq_isVoice": True,
                        "selected_voice": selected_voice_id,
                        "selected_voice_name": selected_voice_name,
                        "faq_bot_id": phone_faq_bot_id,
                        "special_instructions": "",
                        "source": "telnyx_phone"
                    },
                    "created_at": datetime.now().isoformat()
                }
                
                # Store with correct room name
                sessions[room_name] = session_data
                
                # Also store with double-prefixed name (workaround for dispatch rule issue)
                # This allows the session to be found even if LiveKit creates room with wrong name
                double_prefixed_name = f"telnyx_call__{room_name}"
                sessions[double_prefixed_name] = session_data
                logger.info(f"📝 Stored session under both names: {room_name} and {double_prefixed_name}")
                
                # Answer the call and transfer to LiveKit
                # APPROACH: Telnyx transfers the call to LiveKit SIP URI (no outbound calls needed)
                # LiveKit receives via inbound trunk and routes to room
                telnyx_api_key = os.getenv("TELNYX_API_KEY")
                livekit_sip_domain = os.getenv("LIVEKIT_SIP_DOMAIN")
                
                if not telnyx_api_key:
                    logger.error(f"❌ Missing TELNYX_API_KEY")
                    return {"status": "error", "message": "Missing TELNYX_API_KEY"}
                
                if not livekit_sip_domain:
                    logger.error(f"❌ Missing LIVEKIT_SIP_DOMAIN")
                    return {"status": "error", "message": "Missing LIVEKIT_SIP_DOMAIN"}
                
                # Room name no longer contains colons (replaced with dashes), so no URL-encoding needed
                # This prevents duplicate sessions when LiveKit dispatch rule extracts room name
                livekit_sip_uri = f"sip:{room_name}@{livekit_sip_domain}:5060"
                
                logger.info(f"📞 Answering call and transferring to LiveKit")
                logger.info(f"   Room name: {room_name}")
                logger.info(f"   SIP URI: {livekit_sip_uri}")
                logger.info(f"   ✅ Using transfer (not dial) - avoids Telnyx outbound call limits")
                logger.info(f"   ⚠️  LiveKit inbound trunk must be configured to route calls to room")
                
                try:
                    import httpx
                    async with httpx.AsyncClient(timeout=10.0) as client:
                        # Step 1: Answer the inbound call on Telnyx (this doesn't count as outbound)
                        answer_response = await client.post(
                            f"https://api.telnyx.com/v2/calls/{call_control_id}/actions/answer",
                            headers={
                                "Authorization": f"Bearer {telnyx_api_key}",
                                "Content-Type": "application/json",
                                "Accept": "application/json"
                            }
                        )
                        if answer_response.status_code in [200, 201]:
                            logger.info(f"✅ Call answered successfully on Telnyx")
                        else:
                            logger.error(f"❌ Failed to answer call: {answer_response.status_code} - {answer_response.text}")
                            return {"status": "error", "message": "Failed to answer call"}
                        
                        # Step 2: Transfer the call to LiveKit SIP URI
                        # This is a transfer, not a dial, so it doesn't count as an outbound call
                        transfer_response = await client.post(
                            f"https://api.telnyx.com/v2/calls/{call_control_id}/actions/transfer",
                            headers={
                                "Authorization": f"Bearer {telnyx_api_key}",
                                "Content-Type": "application/json",
                                "Accept": "application/json"
                            },
                            json={
                                "to": livekit_sip_uri,
                                "from": called_number or os.getenv("TELNYX_CALLER_NUMBER", "+14153765236")
                            }
                        )
                        if transfer_response.status_code in [200, 201]:
                            logger.info(f"✅ Call transferred to LiveKit successfully")
                            logger.info(f"   ⏳ LiveKit should receive the call via inbound trunk")
                            logger.info(f"   ⏳ LiveKit should route to room: {room_name}")
                            logger.info(f"   ✅ No Telnyx outbound calls used!")
                        else:
                            logger.error(f"❌ Failed to transfer call: {transfer_response.status_code} - {transfer_response.text}")
                            logger.error(f"   Response: {transfer_response.text}")
                            
                except Exception as e:
                    logger.error(f"❌ Error transferring call: {e}")
                    import traceback
                    logger.error(traceback.format_exc())
                
                # Return success response to webhook
                return {"status": "ok"}
                
            except Exception as e:
                logger.error(f"❌ Error creating LiveKit room/participant: {e}")
                # Still try to answer the call
                telnyx_api_key = os.getenv("TELNYX_API_KEY")
                if telnyx_api_key and call_control_id:
                    try:
                        import httpx
                        async with httpx.AsyncClient(timeout=10.0) as client:
                            response = await client.post(
                                f"https://api.telnyx.com/v2/calls/{call_control_id}/actions/answer",
                                headers={
                                    "Authorization": f"Bearer {telnyx_api_key}",
                                    "Content-Type": "application/json",
                                    "Accept": "application/json"
                                }
                            )
                            if response.status_code in [200, 201]:
                                logger.info(f"✅ Call answered despite error")
                            else:
                                logger.error(f"❌ Failed to answer call: {response.status_code}")
                    except Exception as e2:
                        logger.error(f"❌ Error answering call: {e2}")
                
                return {"status": "ok"}
        
        elif event_type == "call.answered":
            # Call was answered - log it and check if it's our outbound call to LiveKit
            data = payload.get("data") or payload.get("payload") or payload
            payload_data = data.get("payload") or data
            call_control_id = payload_data.get("call_control_id") or payload_data.get("id") or data.get("call_control_id")
            direction = payload_data.get("direction") or data.get("direction")
            called_number = payload_data.get("to") or data.get("to")
            
            logger.info(f"📞 Call answered: {call_control_id[:20]}...")
            
            # Check if this is our outbound call to LiveKit being answered
            livekit_sip_domain = os.getenv("LIVEKIT_SIP_DOMAIN", "")
            if direction in ["outbound", "outgoing"] and called_number and livekit_sip_domain and livekit_sip_domain in str(called_number):
                logger.info(f"✅ Outbound call to LiveKit was answered! SIP connection should be established now.")
            
            # For inbound calls, dialing already happened in call.initiated
            # For outbound calls (our dial-out to LiveKit), the bridge_on_answer parameter will automatically bridge
            
            return {"status": "ok"}
        
        elif event_type == "call.hangup":
            # Call ended - cleanup
            # Use same payload parsing logic
            data = payload.get("data") or payload.get("payload") or payload
            # Extract call_control_id from payload (not from event id)
            payload_data = data.get("payload") or data
            call_control_id = payload_data.get("call_control_id") or payload_data.get("id") or data.get("call_control_id")
            logger.info(f"📞 Call ended: {call_control_id}")
            
            # Find and cleanup session
            session_to_delete = []
            for room_name_key, session in sessions.items():
                if session.get("call_control_id") == call_control_id:
                    # Get the actual room name from session data
                    actual_room_name = session.get("room_name", room_name_key)
                    
                    # Cleanup room (try both the actual room name and the key)
                    try:
                        async with api.LiveKitAPI(
                            os.getenv("LIVEKIT_URL"),
                            os.getenv("LIVEKIT_API_KEY"),
                            os.getenv("LIVEKIT_API_SECRET")
                        ) as lk_api:
                            # Try to delete the actual room name first
                            try:
                                await lk_api.room.delete_room(api.DeleteRoomRequest(room=actual_room_name))
                                logger.info(f"✅ Cleaned up room: {actual_room_name}")
                            except Exception as e1:
                                # Check if it's a "not found" error - that's fine, room already deleted
                                if "not_found" in str(e1).lower() or "does not exist" in str(e1).lower():
                                    logger.debug(f"ℹ️ Room {actual_room_name} already deleted (not found)")
                                else:
                                    # If that fails for other reasons, try the key name
                                    try:
                                        await lk_api.room.delete_room(api.DeleteRoomRequest(room=room_name_key))
                                        logger.info(f"✅ Cleaned up room: {room_name_key}")
                                    except Exception as e2:
                                        # Check if it's a "not found" error - that's fine
                                        if "not_found" in str(e2).lower() or "does not exist" in str(e2).lower():
                                            logger.debug(f"ℹ️ Room {room_name_key} already deleted (not found)")
                                        else:
                                            raise e2
                    except Exception as e:
                        # Only log as error if it's not a "not found" error
                        if "not_found" in str(e).lower() or "does not exist" in str(e).lower():
                            logger.debug(f"ℹ️ Room already deleted (not found): {e}")
                        else:
                            logger.error(f"❌ Error cleaning up room: {e}")
                    
                    # Mark session for deletion
                    session_to_delete.append(room_name_key)
            
            # Delete all sessions with this call_control_id (handles both correct and double-prefixed names)
            for room_name_key in session_to_delete:
                if room_name_key in sessions:
                    del sessions[room_name_key]
                    logger.info(f"✅ Removed session: {room_name_key}")
            
            return {"status": "ok"}
        
        else:
            # Other events - just acknowledge
            logger.info(f"📞 Telnyx event: {event_type} - acknowledged")
            return {"status": "ok"}
    
    except Exception as e:
        logger.error(f"❌ Error handling Telnyx webhook: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/telnyx/transfer")
async def telnyx_transfer_call(request: TelnyxTransferRequest):
    """Transfer an active Telnyx call to another number
    
    This transfers the call from the AI agent to a human agent or call center.
    If transferring to the call center number, automatically selects option 1 (speak with representative).
    """
    try:
        logger.info(f"📞 Transferring call {request.call_control_id} to {request.transfer_to}")
        
        # Get Telnyx API key
        telnyx_api_key = os.getenv("TELNYX_API_KEY")
        if not telnyx_api_key:
            raise HTTPException(status_code=500, detail="TELNYX_API_KEY not configured")
        
        # Get the called number from session (the number that received the call)
        session = sessions.get(request.room_name, {})
        called_number = session.get("called_number") or os.getenv("TELNYX_CALLER_NUMBER", "+14153765236")
        
        # Check if this is the call center number that requires IVR selection
        # call_center_number = os.getenv("TELNYX_CALL_CENTER_NUMBER", "+18555518858")
        # needs_ivr_selection = (request.transfer_to == call_center_number)
        
        # Transfer call using Telnyx Call Control API
        # Note: The agent will speak acknowledgment BEFORE calling this endpoint
        # So we don't need a delay here - the agent handles the timing
        
        async with httpx.AsyncClient(timeout=15.0) as client:
            response = await client.post(
                f"https://api.telnyx.com/v2/calls/{request.call_control_id}/actions/transfer",
                headers={
                    "Authorization": f"Bearer {telnyx_api_key}",
                    "Content-Type": "application/json"
                },
                json={
                    "to": request.transfer_to,
                    "from": called_number  # Use the number that received the call
                }
            )
            
            if response.status_code in [200, 201]:
                logger.info(f"✅ Call transferred successfully to {request.transfer_to}")
                
                # TODO: Uncomment below to enable automatic IVR selection for call center
                # If transferring to call center, send DTMF to select option 1
                # if needs_ivr_selection:
                #     logger.info(f"📞 Call center IVR detected - will send DTMF '1' to select 'speak with representative'")
                #     
                #     # Wait for IVR to start (usually 2-3 seconds)
                #     await asyncio.sleep(2.5)
                #     
                #     # Send DTMF tone "1" to select first option
                #     try:
                #         dtmf_response = await client.post(
                #             f"https://api.telnyx.com/v2/calls/{request.call_control_id}/actions/send_dtmf",
                #             headers={
                #                 "Authorization": f"Bearer {telnyx_api_key}",
                #                 "Content-Type": "application/json"
                #             },
                #             json={
                #                 "digits": "1"
                #             }
                #         )
                #         
                #         if dtmf_response.status_code in [200, 201]:
                #             logger.info(f"✅ Sent DTMF '1' to select 'speak with representative'")
                #         else:
                #             logger.warning(f"⚠️ Failed to send DTMF: {dtmf_response.status_code} - {dtmf_response.text}")
                #     except Exception as e:
                #         logger.warning(f"⚠️ Error sending DTMF (call may still work): {e}")
                
                # Mark session as transferred
                if request.room_name in sessions:
                    sessions[request.room_name]["transferred"] = True
                    sessions[request.room_name]["transferred_to"] = request.transfer_to
                    sessions[request.room_name]["transferred_at"] = datetime.now().isoformat()
                
                # Close LiveKit room after successful transfer to stop the agent session
                # The call is now with the human agent, so the AI agent should stop listening
                # Add delay to allow agent's acknowledgment message to be spoken before closing room
                await asyncio.sleep(3.0)  # Give agent time to speak "I'm connecting you with a representative now..."
                
                try:
                    livekit_url = os.getenv("LIVEKIT_URL")
                    livekit_api_key = os.getenv("LIVEKIT_API_KEY")
                    livekit_api_secret = os.getenv("LIVEKIT_API_SECRET")
                    
                    if all([livekit_url, livekit_api_key, livekit_api_secret]):
                        async with api.LiveKitAPI(livekit_url, livekit_api_key, livekit_api_secret) as lk_api:
                            try:
                                await lk_api.room.delete_room(api.DeleteRoomRequest(room=request.room_name))
                                logger.info(f"✅ Closed LiveKit room after transfer: {request.room_name}")
                            except Exception as e:
                                # Check if it's a "not found" error - that's fine, room already closed
                                if "not_found" in str(e).lower() or "does not exist" in str(e).lower():
                                    logger.debug(f"ℹ️ Room {request.room_name} already closed (not found)")
                                else:
                                    logger.warning(f"⚠️ Could not close LiveKit room after transfer: {e}")
                    else:
                        logger.warning("⚠️ LiveKit credentials not configured - cannot close room after transfer")
                except Exception as e:
                    logger.warning(f"⚠️ Error closing LiveKit room after transfer: {e}")
                    # Don't fail the transfer if room cleanup fails
                
        return {
            "status": "success",
            "message": "Call transferred",
            "transfer_to": request.transfer_to
        }
    except Exception as e:
        logger.error(f"❌ Error transferring call: {e}")
        raise HTTPException(status_code=500, detail=str(e)) from e

@app.get("/api/telnyx/sip/trunk/setup")
async def telnyx_sip_setup_info():
    """Get information about setting up Telnyx SIP trunk with LiveKit"""
    livekit_sip_domain = os.getenv("LIVEKIT_SIP_DOMAIN")
    
    return {
        "livekit_sip_domain": livekit_sip_domain,
        "setup_instructions": {
            "1": "Create an inbound SIP trunk in LiveKit Cloud dashboard",
            "2": "Configure Telnyx SIP Connection to forward calls to:",
            "3": f"   SIP URI: sip:{livekit_sip_domain}",
            "4": "Configure Telnyx webhook URL:",
            "5": "   https://your-backend-url.com/api/telnyx/webhook",
            "6": "Set LIVEKIT_SIP_TRUNK_ID in .env after creating trunk"
        }
    }



if __name__ == "__main__":
    import sys
    import os
    # Ensure we can import the app module
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=8000,
        reload=False,
        log_level="info"
    )
