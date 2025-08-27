import os
from datetime import timedelta
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from livekit import api
from dotenv import load_dotenv
import random
import time
import uuid
import uvicorn
from pydantic import BaseModel
from typing import Optional, Dict, Any
import json
import logging

# Load environment variables
load_dotenv(dotenv_path="../.env")

app = FastAPI()
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

print(f"Loaded credentials:")
print(f"API_KEY: {LIVEKIT_API_KEY}")
print(f"API_SECRET: {LIVEKIT_API_SECRET[:10] if LIVEKIT_API_SECRET else 'None'}...")
print(f"URL: {LIVEKIT_URL}")

# Session tracking for analytics
active_sessions: Dict[str, Dict[str, Any]] = {}

# Request models
class ConnectionRequest(BaseModel):
    participant_name: str
    room_name: Optional[str] = None
    intent: Optional[str] = None  # sales, support, billing
    user_data: Optional[Dict[str, Any]] = None

class SessionUpdateRequest(BaseModel):
    room_name: str
    intent: Optional[str] = None
    user_data: Optional[Dict[str, Any]] = None
    status: Optional[str] = None

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
        "features": ["Intent-based routing", "Session tracking", "Dynamic agent assignment"],
        "version": "2.0"
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
        
        if room_name not in active_sessions:
            raise HTTPException(status_code=404, detail="Session not found")
        
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

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)