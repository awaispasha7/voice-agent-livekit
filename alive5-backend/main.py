"""
Alive5 Simple Voice Agent Backend
Simplified backend for the simple-agent worker and frontend
"""

import os
import json
import logging
from pathlib import Path
from typing import Dict, Any, Optional
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

class SessionUpdateRequest(BaseModel):
    room_name: str
    user_data: Dict[str, Any] = {}

class VoiceChangeRequest(BaseModel):
    room_name: str
    voice_id: str


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
        
        # Store session data
        sessions[room_name] = {
            "room_name": room_name,
            "user_name": request.user_name,
            "user_data": {
                "botchain_name": request.botchain_name,
                "org_name": request.org_name,
                "faq_isVoice": request.faq_isVoice,
                "selected_voice": list(AVAILABLE_VOICES.keys())[0]  # Default voice
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



if __name__ == "__main__":
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=False,
        log_level="info"
    )
