"""
Alive5 Simple Voice Agent Backend
Simplified backend for the simple-agent worker and frontend
"""

import os
import json
import logging
from pathlib import Path
from typing import Dict, Any, Optional
import socketio
from datetime import datetime

import httpx
import uvicorn
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from livekit import api

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
        print(f"✅ Loaded .env from: {env_path}")
        env_loaded = True
        break
if not env_loaded:
    load_dotenv()

# Setup logging
logging.basicConfig(level=logging.INFO, format="%(message)s", force=True)
logger = logging.getLogger("simple-backend")

# FastAPI app
app = FastAPI(title="Alive5 Simple Voice Agent Backend", version="1.0")

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
        
        # Use provided values or defaults
        selected_voice_id = request.selected_voice or list(AVAILABLE_VOICES.keys())[0]
        selected_voice_name = AVAILABLE_VOICES.get(selected_voice_id, "Unknown Voice")
        faq_bot_id = request.faq_bot_id or "faq_b9952a56-fc7b-41c9-b0a0-5c662ddb039e"
        org_name = request.org_name or "alive5stage0"
        
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
        
        # Get LiveKit connection details
        livekit_url = os.getenv("LIVEKIT_URL")
        livekit_api_key = os.getenv("LIVEKIT_API_KEY")
        livekit_api_secret = os.getenv("LIVEKIT_API_SECRET")
        
        if not all([livekit_url, livekit_api_key, livekit_api_secret]):
            raise HTTPException(status_code=500, detail="LiveKit configuration missing")
        
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
    if room_name not in sessions:
        raise HTTPException(status_code=404, detail="Session not found")
    
    return sessions[room_name]

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
    """Delete room and clean up session"""
    if room_name in sessions:
        del sessions[room_name]
    if room_name in rooms:
        del rooms[room_name]
    
    return {"status": "deleted", "room_name": room_name}

@app.post("/api/init_livechat")
async def init_livechat(room_name: str, org_name: str, botchain_name: str):
    """Initialize livechat session with Alive5 Socket.io
    
    Step 1: Get widget data and auth token
    Step 2: Connect to socket.io with auth
    Step 3: Emit init-livechat-bot to create thread/CRM
    Step 4: Store thread_id and crm_id in session
    """
    try:
        logger.info(f"🚀 Initializing livechat for room: {room_name}")
        
        # Step 1: Get widget data (includes auth token)
        # TODO: Replace with actual getWidgetData API call
        widget_api_url = "https://api-v2.alive5.com/get-widget-data"  # Replace with actual endpoint
        
        async with httpx.AsyncClient() as client:
            # This should return: { authToken: "...", channel_id: "...", widget_id: "..." }
            response = await client.get(
                widget_api_url,
                params={"org_name": org_name, "botchain": botchain_name}
            )
            
            if response.status_code != 200:
                raise Exception(f"Failed to get widget data: {response.status_code}")
            
            widget_data = response.json()
            auth_token = widget_data.get("authToken")
            channel_id = widget_data.get("channel_id")
            widget_id = widget_data.get("widget_id")
        
        # Step 2: Create socket.io client and connect
        sio = socketio.AsyncClient()
        
        await sio.connect(
            'https://api-v2.alive5.com',
            transports=['websocket', 'polling'],
            auth={
                'authToken': auth_token,
                'thread_id': "",
                'crm_id': "",
                'channel_id': channel_id
            },
            socketio_path='/socket.io'
        )
        
        # Step 3: Emit init-livechat-bot event
        init_data = {
            "channel_id": channel_id,
            "org_name": org_name,
            "botchain_name": botchain_name,
            "message_type": "voicechat",
            "Widget_id": widget_id
        }
        
        # Wait for response with thread_id and crm_id
        response_data = {}
        
        @sio.on('livechat-initialized')
        async def on_init(data):
            nonlocal response_data
            response_data = data
        
        await sio.emit('init-livechat-bot', init_data)
        
        # Wait for response (timeout after 5 seconds)
        import asyncio
        await asyncio.sleep(2)  # Give time for response
        
        # Step 4: Store in session
        if room_name in sessions:
            sessions[room_name]["thread_id"] = response_data.get("thread_id")
            sessions[room_name]["crm_id"] = response_data.get("crm_id")
            sessions[room_name]["auth_token"] = auth_token
            sessions[room_name]["channel_id"] = channel_id
            sessions[room_name]["widget_id"] = widget_id
            sessions[room_name]["socket_connected"] = True
        
        await sio.disconnect()
        
        logger.info(f"✅ Livechat initialized - Thread: {response_data.get('thread_id')}, CRM: {response_data.get('crm_id')}")
        
        return {
            "status": "success",
            "thread_id": response_data.get("thread_id"),
            "crm_id": response_data.get("crm_id")
        }
        
    except Exception as e:
        logger.error(f"❌ Error initializing livechat: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/submit_crm")
async def submit_crm(request: CRMSubmissionRequest):
    """Submit collected CRM data via Socket.io livechat-message event
    
    Uses existing thread_id and crm_id from init_livechat
    """
    try:
        logger.info(f"📝 CRM submission received for room: {request.room_name}")
        logger.info(f"   Name: {request.full_name}")
        logger.info(f"   Email: {request.email}")
        logger.info(f"   Notes: {request.notes}")
        
        # Get session data (includes auth_token, thread_id, crm_id from init)
        session = sessions.get(request.room_name, {})
        auth_token = session.get("auth_token")
        thread_id = session.get("thread_id")
        crm_id = session.get("crm_id")
        channel_id = session.get("channel_id")
        widget_id = session.get("widget_id")
        
        if not auth_token:
            raise Exception("Session not initialized - call init_livechat first")
        
        # Store CRM data in session
        if "crm_data" not in sessions[request.room_name]:
            sessions[request.room_name]["crm_data"] = {}
        
        sessions[request.room_name]["crm_data"].update({
            "full_name": request.full_name,
            "email": request.email,
            "phone": request.phone,
            "notes": request.notes,
            "submitted_at": datetime.now().isoformat()
        })
        
        # Connect to Socket.io
        sio = socketio.AsyncClient()
        
        await sio.connect(
            'https://api-v2.alive5.com',
            transports=['websocket', 'polling'],
            auth={
                'authToken': auth_token,
                'thread_id': thread_id or "",
                'crm_id': crm_id or "",
                'channel_id': channel_id
            },
            socketio_path='/socket.io'
        )
        
        # Prepare message data as per client specification
        message_data = {
            "channel_id": channel_id,
            "Message_content": f"Name: {request.full_name}\nEmail: {request.email}\nPhone: {request.phone or 'N/A'}\nNotes: {request.notes}",
            "message_type": "voicechat",
            "org_name": request.org_name,
            "thread_id": thread_id,
            "crm_id": crm_id,
            "newThread": False,  # Thread already exists from init
            "attach_botchain": request.botchain_name,
            "webpage_title": "Voice Agent",
            "webpage_url": "",
            "assignedTo": "",
            "user_interacted": "additional_action",
            "Widget_id": widget_id
        }
        
        # Emit livechat-message event
        await sio.emit('livechat-message', message_data)
        logger.info(f"✅ CRM data sent via livechat-message")
        
        await sio.disconnect()
        
        return {
            "status": "success",
            "message": "CRM data sent to Alive5",
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
        auth_token = session.get("auth_token")
        thread_id = session.get("thread_id")
        crm_id = session.get("crm_id")
        channel_id = session.get("channel_id")
        
        if not auth_token:
            logger.warning("No active session to end")
            return {"status": "no_session"}
        
        # Connect and emit visitor-chat-end
        sio = socketio.AsyncClient()
        
        await sio.connect(
            'https://api-v2.alive5.com',
            transports=['websocket', 'polling'],
            auth={
                'authToken': auth_token,
                'thread_id': thread_id or "",
                'crm_id': crm_id or "",
                'channel_id': channel_id
            },
            socketio_path='/socket.io'
        )
        
        end_data = {
            "thread_id": thread_id,
            "crm_id": crm_id,
            "channel_id": channel_id
        }
        
        await sio.emit('visitor-chat-end', end_data)
        logger.info(f"✅ Livechat session ended")
        
        await sio.disconnect()
        
        # Clean up session
        if room_name in sessions:
            sessions[room_name]["socket_connected"] = False
        
        return {"status": "success", "message": "Session ended"}
        
    except Exception as e:
        logger.error(f"❌ Error ending livechat: {e}")
        raise HTTPException(status_code=500, detail=str(e))



if __name__ == "__main__":
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=False,
        log_level="info"
    )
