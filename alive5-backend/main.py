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
import base64
import hashlib
from typing import Dict, Any, Optional
from datetime import datetime
from contextlib import asynccontextmanager

import httpx
import uvicorn
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, Response
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
    first_name: Optional[str] = None
    last_name: Optional[str] = None
    email: Optional[str] = None
    phone: Optional[str] = None
    account_id: Optional[str] = None
    company: Optional[str] = None
    company_title: Optional[str] = None
    notes: Optional[str] = None

class TelnyxWebhookRequest(BaseModel):
    """Telnyx webhook payload structure"""
    data: Dict[str, Any]

class TelnyxTransferRequest(BaseModel):
    """Request to transfer a Telnyx call"""
    room_name: str
    call_control_id: str
    transfer_to: str


# Session storage - use AgentCore Memory if enabled, otherwise fallback to in-memory
# Import AgentCore Memory
try:
    from agentcore.memory import AgentCoreMemory
    memory_client = AgentCoreMemory()
    USE_AGENTCORE_MEMORY = memory_client.is_enabled()
except ImportError:
    memory_client = None
    USE_AGENTCORE_MEMORY = False

# Fallback in-memory storage (used if AgentCore Memory is disabled)
sessions: Dict[str, Dict[str, Any]] = {}
rooms: Dict[str, Dict[str, Any]] = {}

# Helper functions for session management
async def get_session_data(room_name: str) -> Optional[Dict[str, Any]]:
    """Get session data from Memory or fallback dict"""
    if USE_AGENTCORE_MEMORY and memory_client:
        session = await memory_client.get_session(room_name)
        if session:
            return session
    
    # Fallback to in-memory dict
    return sessions.get(room_name)

async def store_session_data(room_name: str, session_data: Dict[str, Any]):
    """Store session data in Memory or fallback dict"""
    if USE_AGENTCORE_MEMORY and memory_client:
        await memory_client.store_session(room_name, session_data)
    else:
        sessions[room_name] = session_data

async def update_session_data(room_name: str, updates: Dict[str, Any]):
    """Update session data in Memory or fallback dict"""
    session = await get_session_data(room_name)
    if not session:
        session = {}
    
    # Deep update
    def deep_update(base, updates):
        for key, value in updates.items():
            if isinstance(value, dict) and key in base and isinstance(base[key], dict):
                deep_update(base[key], value)
            else:
                base[key] = value
    
    deep_update(session, updates)
    await store_session_data(room_name, session)

async def delete_session_data(room_name: str):
    """Delete session data (Memory doesn't support delete, so we store empty dict)"""
    if USE_AGENTCORE_MEMORY and memory_client:
        # Memory doesn't have explicit delete, but we can store empty dict
        await memory_client.store_session(room_name, {})
    else:
        if room_name in sessions:
            del sessions[room_name]

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


def _voice_preview_text(voice_display_name: str) -> str:
    """Generate the preview text the character will speak."""
    # Voice names are like: "Calypso - ASMR Lady" → character is "Calypso"
    character = (voice_display_name or "").split(" - ", 1)[0].strip() or "this character"
    return f"Hi! I'm {character}. Pick me as your voice."


def _voice_preview_cache_path(voice_id: str, voice_display_name: str, model_id: str) -> Path:
    """
    Cache key includes voice_id + model_id + preview text.
    This keeps cache stable even if we later tweak the text/model.
    """
    text = _voice_preview_text(voice_display_name)
    key = f"{voice_id}|{model_id}|{text}".encode("utf-8")
    digest = hashlib.sha256(key).hexdigest()[:24]
    cache_dir = current_dir / "voice_previews"
    cache_dir.mkdir(parents=True, exist_ok=True)
    return cache_dir / f"{voice_id}-{digest}.wav"


@app.get("/api/voice_preview")
async def voice_preview(voice_id: str):
    """
    Return a short TTS audio preview for the given voice_id.
    The audio is cached on disk to avoid repeated TTS generation.
    """
    if not voice_id:
        raise HTTPException(status_code=400, detail="voice_id is required")

    voice_display_name = AVAILABLE_VOICES.get(voice_id)
    if not voice_display_name:
        raise HTTPException(status_code=404, detail="Unknown voice_id")

    cartesia_api_key = os.getenv("CARTESIA_API_KEY")
    if not cartesia_api_key:
        raise HTTPException(status_code=500, detail="CARTESIA_API_KEY not configured on server")

    model_id = os.getenv("CARTESIA_TTS_MODEL", "sonic-2")
    cache_path = _voice_preview_cache_path(voice_id, voice_display_name, model_id)

    if cache_path.exists() and cache_path.stat().st_size > 0:
        return FileResponse(str(cache_path), media_type="audio/wav")

    preview_text = _voice_preview_text(voice_display_name)

    cartesia_version = os.getenv("CARTESIA_VERSION", "2025-04-16")

    # Cartesia "Bytes" endpoint docs show `audio/wav` responses.
    # Keep output_format explicit; otherwise the API may interpret missing fields as zeros.
    payload = {
        "model_id": model_id,
        "transcript": preview_text,
        "voice": {"mode": "id", "id": voice_id},
        "output_format": {
            "container": "wav",
            "encoding": "pcm_s16le",
            "sample_rate": 16000,
        },
    }

    async with httpx.AsyncClient(timeout=30.0) as client:
        resp = await client.post(
            "https://api.cartesia.ai/tts/bytes",
            headers={
                "X-API-Key": cartesia_api_key,
                "Content-Type": "application/json",
                "Cartesia-Version": cartesia_version,
            },
            json=payload,
        )

    if resp.status_code != 200:
        detail = resp.text[:500] if resp.text else f"Cartesia TTS error (HTTP {resp.status_code})"
        raise HTTPException(status_code=502, detail=detail)

    content_type = (resp.headers.get("content-type") or "").lower()
    audio_bytes: bytes
    if "application/json" in content_type:
        data = resp.json()
        b64 = (
            data.get("audio")
            or data.get("data")
            or (data.get("result") or {}).get("audio")
            or (data.get("result") or {}).get("data")
        )
        if not b64 or not isinstance(b64, str):
            raise HTTPException(status_code=502, detail="TTS provider returned JSON without audio data")
        audio_bytes = base64.b64decode(b64)
    else:
        audio_bytes = resp.content

    if not audio_bytes:
        raise HTTPException(status_code=502, detail="Empty audio returned from TTS provider")

    # Write cache atomically
    tmp_path = cache_path.with_suffix(".tmp")
    tmp_path.write_bytes(audio_bytes)
    tmp_path.replace(cache_path)

    return Response(content=audio_bytes, media_type="audio/wav")

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
        
        # Convert ws:// to http:// for REST API calls (frontend uses ws:// for WebSocket)
        # Use localhost for API calls since backend is on same server as LiveKit
        livekit_api_url = livekit_url.replace("ws://", "http://").replace("wss://", "https://")
        # Replace public IP with localhost for API calls (backend is on same server)
        if "18.210.238.67" in livekit_api_url:
            livekit_api_url = livekit_api_url.replace("18.210.238.67", "localhost")
        logger.info(f"🔗 Connecting to LiveKit API at: {livekit_api_url}")
        
        # CRITICAL: Create the LiveKit room BEFORE returning connection details
        # This ensures the room exists and the worker can be dispatched when the frontend connects
        # This is the same approach used for phone calls (line 593)
        try:
            # Add timeout to prevent hanging (10 seconds)
            async def create_room_with_timeout():
                async with api.LiveKitAPI(livekit_api_url, livekit_api_key, livekit_api_secret) as lk_api:
                    logger.info(f"✅ Connected to LiveKit API, creating room: {room_name}")
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
                        logger.warning(f"⚠️ Room {room_name} might already exist: {room_error}")
                        # Try to get existing room info
                        try:
                            rooms = await lk_api.room.list_rooms(api.ListRoomsRequest(names=[room_name]))
                            if rooms.rooms:
                                logger.info(f"✅ Room {room_name} already exists")
                            else:
                                logger.warning(f"⚠️ Could not create or find room {room_name}, but continuing...")
                        except Exception as list_error:
                            logger.warning(f"⚠️ Could not verify room existence: {list_error}, but continuing...")
            
            await asyncio.wait_for(create_room_with_timeout(), timeout=10.0)
        except asyncio.TimeoutError:
            logger.error(f"❌ Timeout connecting to LiveKit API at {livekit_api_url} (10s timeout)")
            logger.warning("⚠️ Continuing without pre-creating room - frontend connection will create it")
        except Exception as api_error:
            logger.error(f"❌ Failed to connect to LiveKit API at {livekit_api_url}: {api_error}")
            import traceback
            logger.error(f"   Traceback: {traceback.format_exc()}")
            # Continue anyway - room creation might work when frontend connects
            logger.warning("⚠️ Continuing without pre-creating room - frontend connection will create it")
        
        # Store session data
        session_data = {
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
        
        # Store using helper function
        await store_session_data(room_name, session_data)
        
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
        
        # Convert ws:// to wss:// for frontend (frontend is on HTTPS, needs secure WebSocket)
        # Use nginx proxy for WSS (nginx handles TLS termination and proxies to LiveKit WS on port 7880)
        # LiveKit SDK will append /rtc/v1 to the base URL, so we return just the domain
        frontend_url = livekit_url
        if frontend_url.startswith("ws://"):
            # Replace ws://IP:7880 with wss://domain (nginx-proxied WSS connection)
            # SDK will append /rtc/v1, nginx will proxy /rtc/* to LiveKit
            if "18.210.238.67" in frontend_url:
                frontend_url = "wss://18.210.238.67.nip.io"
            else:
                # Generic conversion: extract host and use nginx proxy
                host = frontend_url.replace("ws://", "").replace(":7880", "")
                frontend_url = f"wss://{host}"
            logger.info(f"🔒 Converted WebSocket URL for HTTPS frontend: {livekit_url} → {frontend_url} (nginx-proxied WSS)")
        
        return {
            "url": frontend_url,
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
    
    # Try helper function first (handles Memory and fallback)
    session = await get_session_data(room_name_clean)
    if session:
        return session
    
    # Try decoded name
    session = await get_session_data(room_name_decoded)
    if session:
        return session
    
    # Try original
    session = await get_session_data(room_name)
    if session:
        return session
    
    # Fallback: Try to find a matching session (in case of encoding issues)
    # Only check in-memory dict for this (Memory lookup already tried above)
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
    # Get existing session
    session = await get_session_data(request.room_name)
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")
    
    # Update session data
    await update_session_data(request.room_name, {
        "user_data": request.user_data,
        "updated_at": datetime.now().isoformat()
    })
    
    return {"status": "updated"}

@app.post("/api/change_voice")
async def change_voice(request: VoiceChangeRequest):
    """Change voice for a session"""
    logger.info(f"Voice change request: room={request.room_name}, voice={request.voice_id}")
    
    # Get session
    session = None
    if USE_AGENTCORE_MEMORY and memory_client:
        session = await memory_client.get_session(request.room_name)
    
    if not session:
        if request.room_name not in sessions:
            raise HTTPException(status_code=404, detail="Session not found")
        session = sessions[request.room_name]
    
    # Validate voice_id exists
    if request.voice_id not in AVAILABLE_VOICES:
        raise HTTPException(status_code=400, detail="Invalid voice ID")
    
    # Update voice
    session["user_data"]["selected_voice"] = request.voice_id
    session["selected_voice"] = request.voice_id
    session["voice_id"] = request.voice_id
    session["updated_at"] = datetime.now().isoformat()
    
    # Store back
    if USE_AGENTCORE_MEMORY and memory_client:
        await memory_client.store_session(request.room_name, session)
    else:
        sessions[request.room_name] = session
    
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
                # Convert ws:// to http:// for REST API calls
                livekit_api_url = livekit_url.replace("ws://", "http://").replace("wss://", "https://")
                # Replace public IP with localhost for API calls (backend is on same server)
                if "18.210.238.67" in livekit_api_url:
                    livekit_api_url = livekit_api_url.replace("18.210.238.67", "localhost")
                async with api.LiveKitAPI(livekit_api_url, livekit_api_key, livekit_api_secret) as lk_api:
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
        await delete_session_data(room_name)
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
        await update_session_data(room_name, {
            "thread_id": final_thread_id,
            "crm_id": final_crm_id,
            "api_key": api_key,
            "channel_id": channel_id,
            "widget_id": widget_id
        })
        
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
        logger.info(f"   Name: {request.full_name or ((request.first_name or '') + ' ' + (request.last_name or '')).strip()}")
        logger.info(f"   Email: {request.email}")
        logger.info(f"   Notes: {request.notes}")
        
        # Get session data (includes api_key, thread_id, crm_id from init)
        session = await get_session_data(request.room_name) or {}
        api_key = session.get("api_key") or os.getenv("A5_API_KEY")
        thread_id = session.get("thread_id")
        crm_id = session.get("crm_id")
        channel_id = session.get("channel_id")
        
        if not api_key:
            raise Exception("A5_API_KEY not configured - cannot submit CRM data")
        if not crm_id:
            raise Exception("Session not initialized - call init_livechat first")
        
        # Store CRM data in session (worker will send via frontend socket)
        crm_data = session.get("crm_data", {})
        crm_data.update({
            "full_name": request.full_name,
            "first_name": request.first_name,
            "last_name": request.last_name,
            "email": request.email,
            "phone": request.phone,
            "account_id": request.account_id,
            "company": request.company,
            "company_title": request.company_title,
            "notes": request.notes,
            "submitted_at": datetime.now().isoformat()
        })
        
        await update_session_data(request.room_name, {"crm_data": crm_data})
        
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
        
        session = await get_session_data(room_name) or {}
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
            
            # IMPORTANT (Self-hosted LiveKit SIP) - Robust per-call rooms:
            # - Reusing the same LiveKit room across calls causes CRM/session leakage.
            # - Very long SIP usernames can trigger SIP flood protection.
            #
            # Solution:
            # - Keep ONE stable dispatch rule using CALLEE routing with a room_prefix.
            # - Transfer to a SHORT callee derived from call_control_id.
            # - LiveKit SIP will create/join room: {room_prefix}{callee}.
            import hashlib
            safe_call_control_id = call_control_id.replace(":", "-")
            room_prefix = os.getenv("TELNYX_ROOM_PREFIX", "telnyx_call_")
            short_callee = hashlib.sha1(safe_call_control_id.encode("utf-8")).hexdigest()[:12]
            telnyx_room_name = f"{room_prefix}{short_callee}"
            
            # Get LiveKit credentials
            livekit_url = os.getenv("LIVEKIT_URL")
            livekit_api_key = os.getenv("LIVEKIT_API_KEY")
            livekit_api_secret = os.getenv("LIVEKIT_API_SECRET")
            livekit_sip_domain = os.getenv("LIVEKIT_SIP_DOMAIN")
            
            if not all([livekit_url, livekit_api_key, livekit_api_secret, livekit_sip_domain]):
                raise HTTPException(status_code=500, detail="LiveKit SIP configuration missing")
            
            # Convert ws:// to http:// and use localhost for API calls
            livekit_api_url = livekit_url.replace("ws://", "http://").replace("wss://", "https://")
            if "18.210.238.67" in livekit_api_url:
                livekit_api_url = livekit_api_url.replace("18.210.238.67", "localhost")
            
            # Create LiveKit room (idempotent: OK if it already exists)
            try:
                async with api.LiveKitAPI(livekit_api_url, livekit_api_key, livekit_api_secret) as lk_api:
                    try:
                        await lk_api.room.create_room(
                            api.CreateRoomRequest(
                                name=telnyx_room_name,
                                empty_timeout=300,  # 5 minutes
                                max_participants=2,
                            )
                        )
                        logger.info(f"✅ Created LiveKit room: {telnyx_room_name}")
                    except Exception as room_err:
                        # Room likely already exists; don't fail the call for that.
                        logger.warning(f"ℹ️ Could not create room (may already exist): {room_err}")

                    # SELF-HOSTED SIP (livekit/sip):
                    # We must ensure there's an inbound SIP trunk + a dispatch rule that routes the next
                    # SIP INVITE into *this* room. Otherwise Telnyx will connect to the SIP endpoint,
                    # but no SIP participant will ever join the room.
                    sip_trunk_id: str | None = None
                    sip_dispatch_rule_id: str | None = None
                    try:
                        effective_called_number = (
                            called_number
                            or os.getenv("TELNYX_CALLER_NUMBER", "")
                            or ""
                        ).strip()

                        # 1) Ensure inbound trunk exists (create if missing)
                        trunks = await lk_api.sip.list_inbound_trunk(api.ListSIPInboundTrunkRequest())
                        for t in (trunks.items or []):
                            if effective_called_number and effective_called_number in list(t.numbers):
                                sip_trunk_id = t.sip_trunk_id
                                break

                        if not sip_trunk_id:
                            # NOTE: For security, you should restrict allowed_addresses to Telnyx IP ranges,
                            # but 0.0.0.0/0 is fine for initial bring-up/testing.
                            trunk = api.SIPInboundTrunkInfo(
                                name="telnyx-inbound",
                                numbers=[effective_called_number] if effective_called_number else [],
                                allowed_addresses=["0.0.0.0/0"],
                            )
                            created_trunk = await lk_api.sip.create_inbound_trunk(
                                api.CreateSIPInboundTrunkRequest(trunk=trunk)
                            )
                            sip_trunk_id = created_trunk.sip_trunk_id
                            logger.info(f"✅ Created SIP inbound trunk: {sip_trunk_id}")

                        # 2) Ensure a SINGLE dispatch rule exists for this trunk.
                        # Use CALLEE routing so we can create per-call rooms by changing the SIP URI user part.
                        existing_rule = None
                        rules = await lk_api.sip.list_dispatch_rule(api.ListSIPDispatchRuleRequest())
                        # First, try to find by stored rule ID (if we have one from env or previous calls)
                        stored_rule_id = os.getenv("LIVEKIT_SIP_DISPATCH_RULE_NAME")
                        for r in (rules.items or []):
                            # Check by stored rule ID first (most reliable - fixes orphaned rules)
                            if stored_rule_id and r.sip_dispatch_rule_id == stored_rule_id:
                                existing_rule = r
                                logger.info(f"🔍 Found dispatch rule by stored ID: {stored_rule_id}")
                                break
                        # If not found by ID, check by trunk_id
                        if not existing_rule:
                            for r in (rules.items or []):
                                if sip_trunk_id and sip_trunk_id in list(r.trunk_ids):
                                    # Some users have a manually-created "catch-all" rule with inbound_numbers=[]
                                    # (matches any DID on that trunk). That will conflict with creating a more specific
                                    # rule, so if we see it, we update it instead of creating a new one.
                                    inbound_numbers = list(r.inbound_numbers)
                                    if not inbound_numbers:
                                        existing_rule = r
                                        logger.info(f"🔍 Found dispatch rule by trunk_id (catch-all): {r.sip_dispatch_rule_id}")
                                        break
                                    if effective_called_number and effective_called_number in inbound_numbers:
                                        existing_rule = r
                                        logger.info(f"🔍 Found dispatch rule by trunk_id (specific DID): {r.sip_dispatch_rule_id}")
                                        break
                        # If still not found, check by name (fallback for manually created rules)
                        if not existing_rule:
                            for r in (rules.items or []):
                                if r.name and ("telnyx" in r.name.lower() or "dispatch" in r.name.lower()):
                                    existing_rule = r
                                    logger.info(f"🔍 Found dispatch rule by name: {r.name} ({r.sip_dispatch_rule_id})")
                                    break

                        desired_rule = api.SIPDispatchRule(
                            dispatch_rule_callee=api.SIPDispatchRuleCallee(
                                room_prefix=room_prefix,
                                randomize=False,  # Keep False - room name is already unique via SHA1 hash
                            )
                        )

                        if existing_rule:
                            sip_dispatch_rule_id = existing_rule.sip_dispatch_rule_id
                            # Update rule if it doesn't match what we want.
                            try:
                                # NOTE: livekit python SDK expects (rule_id, SIPDispatchRuleInfo) here (not a request object).
                                # CRITICAL: Always set trunk_ids even if the rule had None before (fixes orphaned rules)
                                desired_info = api.SIPDispatchRuleInfo(
                                    sip_dispatch_rule_id=sip_dispatch_rule_id,
                                    name=existing_rule.name or "telnyx-dispatch",
                                    trunk_ids=[sip_trunk_id] if sip_trunk_id else [],
                                    inbound_numbers=list(existing_rule.inbound_numbers) if existing_rule.inbound_numbers else [],
                                    hide_phone_number=True,
                                    rule=desired_rule,
                                )
                                await lk_api.sip.update_dispatch_rule(sip_dispatch_rule_id, desired_info)
                                logger.info(f"✅ Updated SIP dispatch rule: {sip_dispatch_rule_id} (room_prefix={room_prefix}, trunk={sip_trunk_id})")
                            except Exception as upd_err:
                                logger.error(f"❌ Could not update SIP dispatch rule {sip_dispatch_rule_id}: {upd_err}", exc_info=True)
                        else:
                            created_rule = await lk_api.sip.create_dispatch_rule(
                                api.CreateSIPDispatchRuleRequest(
                                    rule=desired_rule,
                                    trunk_ids=[sip_trunk_id] if sip_trunk_id else [],
                                    inbound_numbers=[effective_called_number] if effective_called_number else [],
                                    name="telnyx-dispatch",
                                    hide_phone_number=True,
                                )
                            )
                            sip_dispatch_rule_id = created_rule.sip_dispatch_rule_id
                            logger.info(f"✅ Created SIP dispatch rule: {sip_dispatch_rule_id}")

                    except Exception as sip_err:
                        logger.error(f"❌ Failed to ensure SIP trunk/dispatch rule: {sip_err}", exc_info=True)
                    
                    logger.info("✅ Room created - SIP dispatch rule prepared for Telnyx transfer")
                
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
                    "room_name": telnyx_room_name,
                    "user_name": f"Caller_{caller_number}",
                    "call_control_id": call_control_id,
                    "caller_number": caller_number,
                    "called_number": called_number,
                    "sip_trunk_id": sip_trunk_id,
                    "sip_dispatch_rule_id": sip_dispatch_rule_id,
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
                await store_session_data(telnyx_room_name, session_data)
                
                # Also store with double-prefixed name (workaround for dispatch rule issue)
                # This allows the session to be found even if LiveKit creates room with wrong name
                double_prefixed_name = f"telnyx_call__{telnyx_room_name}"
                await store_session_data(double_prefixed_name, session_data)
                logger.info(f"📝 Stored session under both names: {telnyx_room_name} and {double_prefixed_name}")

                # IMPORTANT: Phone calls do NOT have a frontend to call /api/init_livechat.
                # The worker polls /api/sessions/{room} for thread_id/crm_id/channel_id; without these
                # it will keep retrying and the call flow won't fully start.
                #
                # So we initialize the Alive5 session here in the backend (non-blocking), and write the
                # resulting IDs into BOTH session keys (normal + double-prefixed) so the worker can find them.
                async def _init_phone_alive5_session():
                    try:
                        init = await init_livechat(
                            room_name=telnyx_room_name,
                            org_name=phone_org_name,
                            botchain_name=phone_botchain,
                        )
                        # Mirror the same IDs to the double-prefixed key (avoid generating a second thread_id)
                        try:
                            await update_session_data(double_prefixed_name, {
                                "thread_id": init.get("thread_id"),
                                "crm_id": init.get("crm_id"),
                                "api_key": init.get("api_key"),
                                "channel_id": init.get("channel_id"),
                                "widget_id": os.getenv("A5_WIDGET_ID"),
                            })
                        except Exception as e2:
                            logger.warning(f"⚠️ Could not mirror phone session data to {double_prefixed_name}: {e2}")
                        logger.info(f"✅ Phone Alive5 session initialized for {telnyx_room_name} (worker can now proceed)")
                    except Exception as e:
                        logger.error(f"❌ Failed to init Alive5 session for phone call room {telnyx_room_name}: {e}", exc_info=True)

                try:
                    asyncio.create_task(_init_phone_alive5_session())
                except Exception as e:
                    logger.warning(f"⚠️ Could not schedule phone Alive5 session init task: {e}")
                
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
                
                # Transfer to a SHORT callee (not the DID), so the callee dispatch rule can route to:
                #   room = {room_prefix}{short_callee}
                livekit_sip_uri = f"sip:{short_callee}@{livekit_sip_domain}:5060"
                
                logger.info(f"📞 Answering call and transferring to LiveKit")
                logger.info(f"   Room name (dispatch target): {telnyx_room_name}")
                logger.info(f"   SIP URI: {livekit_sip_uri}")
                logger.info(f"   ✅ Using transfer (not dial) - avoids Telnyx outbound call limits")
                logger.info(f"   ✅ SIP dispatch rule will route the callee to room via room_prefix")
                
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
                            logger.info(f"   ⏳ LiveKit should route to room: {telnyx_room_name}")
                            logger.info(f"   ⏳ SIP URI sent to Telnyx: {livekit_sip_uri}")
                            logger.info(f"   ⏳ Expected callee in SIP INVITE: {short_callee}")
                            logger.info(f"   ✅ No Telnyx outbound calls used!")
                        else:
                            logger.error(f"❌ Failed to transfer call: {transfer_response.status_code} - {transfer_response.text}")
                            logger.error(f"   Response: {transfer_response.text}")
                            logger.error(f"   SIP URI attempted: {livekit_sip_uri}")
                            
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
                    sip_dispatch_rule_id = session.get("sip_dispatch_rule_id")
                    
                    # Cleanup room (try both the actual room name and the key)
                    try:
                        livekit_url_for_api = (os.getenv("LIVEKIT_URL") or "").replace("ws://", "http://").replace("wss://", "https://")
                        # Ensure backend talks to local LiveKit API on the same server
                        if "18.210.238.67" in livekit_url_for_api:
                            livekit_url_for_api = livekit_url_for_api.replace("18.210.238.67", "localhost")
                        async with api.LiveKitAPI(
                            livekit_url_for_api,
                            os.getenv("LIVEKIT_API_KEY"),
                            os.getenv("LIVEKIT_API_SECRET")
                        ) as lk_api:
                            # NOTE:
                            # We keep the SIP dispatch rule around (it's a stable rule for the trunk + DID),
                            # because LiveKit does not allow creating a new per-call rule for the same trunk/inbound number.
                            # Deleting it here would break the next call (or worse, remove a manually configured rule).
                            if sip_dispatch_rule_id:
                                logger.info(f"ℹ️ Keeping SIP dispatch rule (not deleting): {sip_dispatch_rule_id}")

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
                await delete_session_data(room_name_key)
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
                await update_session_data(request.room_name, {
                    "transferred": True,
                    "transferred_to": request.transfer_to,
                    "transferred_at": datetime.now().isoformat()
                })
                
                # Close LiveKit room after successful transfer to stop the agent session
                # The call is now with the human agent, so the AI agent should stop listening
                # Add delay to allow agent's acknowledgment message to be spoken before closing room
                await asyncio.sleep(3.0)  # Give agent time to speak "I'm connecting you with a representative now..."
                
                try:
                    livekit_url = os.getenv("LIVEKIT_URL")
                    livekit_api_key = os.getenv("LIVEKIT_API_KEY")
                    livekit_api_secret = os.getenv("LIVEKIT_API_SECRET")
                    
                    if all([livekit_url, livekit_api_key, livekit_api_secret]):
                        # Convert ws:// to http:// and use localhost for API calls
                        livekit_api_url = livekit_url.replace("ws://", "http://").replace("wss://", "https://")
                        if "18.210.238.67" in livekit_api_url:
                            livekit_api_url = livekit_api_url.replace("18.210.238.67", "localhost")
                        async with api.LiveKitAPI(livekit_api_url, livekit_api_key, livekit_api_secret) as lk_api:
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
