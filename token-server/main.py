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
from typing import Optional

# Load environment variables
load_dotenv(dotenv_path="../.env")

app = FastAPI()

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

# Request model for POST endpoint
class ConnectionRequest(BaseModel):
    participant_name: str
    room_name: Optional[str] = None

def generate_truly_unique_room_name(participant_name: str = None) -> str:
    """Generate a truly unique room name using UUID"""
    unique_id = str(uuid.uuid4())
    timestamp = int(time.time())
    
    if participant_name:
        # Sanitize participant name (remove special characters)
        clean_name = ''.join(c for c in participant_name if c.isalnum()).lower()[:8]
        return f"alive5_{clean_name}_{timestamp}_{unique_id[:8]}"
    else:
        return f"alive5_user_{timestamp}_{unique_id[:8]}"

@app.get("/")
def read_root():
    return {"message": "Alive5 Token Server is running"}

@app.get("/health")
def health_check():
    return {"status": "ok"}

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
        
        print(f"Generating token for {participant_name} in room {room_name}")
        
        # Create token with shorter TTL to ensure cleanup
        token = api.AccessToken(LIVEKIT_API_KEY, LIVEKIT_API_SECRET)
        token.with_identity(participant_name)
        token.with_grants(api.VideoGrants(
            room_join=True,
            room=room_name,
            can_publish=True,
            can_subscribe=True,
        ))
        token.with_ttl(timedelta(minutes=30))  # Shorter TTL for better cleanup
        
        jwt_token = token.to_jwt()
        
        return {
            "serverUrl": LIVEKIT_URL,
            "roomName": room_name,
            "participantToken": jwt_token,
            "participantName": participant_name
        }
        
    except Exception as e:
        print(f"Error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/connection_details")
def create_connection_with_custom_room(request: ConnectionRequest):
    """POST endpoint - allows frontend to specify participant name and room"""
    if not all([LIVEKIT_API_KEY, LIVEKIT_API_SECRET, LIVEKIT_URL]):
        raise HTTPException(
            status_code=500,
            detail="Missing LiveKit credentials"
        )
    
    try:
        participant_name = request.participant_name
        
        # Always generate a truly unique room name
        room_name = generate_truly_unique_room_name(participant_name)
        
        print(f"Generating token for {participant_name} in unique room {room_name}")
        
        # Create token with shorter TTL
        token = api.AccessToken(LIVEKIT_API_KEY, LIVEKIT_API_SECRET)
        token.with_identity(participant_name)
        token.with_grants(api.VideoGrants(
            room_join=True,
            room=room_name,
            can_publish=True,
            can_subscribe=True,
        ))
        token.with_ttl(timedelta(minutes=30))  # Shorter TTL for cleanup
        
        jwt_token = token.to_jwt()
        
        return {
            "serverUrl": LIVEKIT_URL,
            "roomName": room_name,
            "participantToken": jwt_token,
            "participantName": participant_name
        }
        
    except Exception as e:
        print(f"Error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.delete("/api/rooms/{room_name}")
def cleanup_room(room_name: str):
    """Endpoint to cleanup/end a room session"""
    try:
        print(f"Room cleanup requested for: {room_name}")
        # Note: Actual room cleanup would require LiveKit API calls
        # For now, we'll log the cleanup request
        return {"message": f"Cleanup requested for room {room_name}"}
    except Exception as e:
        print(f"Cleanup error: {e}")
        return {"error": str(e)}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)