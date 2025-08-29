import logging
import os
import uuid
import asyncio
from datetime import datetime
from dotenv import load_dotenv
from typing import Dict, Any, Optional
import json

from livekit.agents import (
    AgentSession,
    Agent,
    JobContext,
    WorkerOptions,
    cli,
    RoomInputOptions,
    RoomOutputOptions,
    AutoSubscribe
)
from livekit.plugins import (
    deepgram,
    cartesia,
    openai,
    silero,
    noise_cancellation
)
from livekit.plugins.turn_detector.multilingual import MultilingualModel
from livekit import rtc

# Load environment variables
load_dotenv(dotenv_path=".env")

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(name)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger("dynamic-voice-agent")
logger.setLevel(logging.INFO)

# Verify environment variables
required_vars = ["OPENAI_API_KEY", "DEEPGRAM_API_KEY", "CARTESIA_API_KEY", "LIVEKIT_URL", "LIVEKIT_API_KEY", "LIVEKIT_API_SECRET"]
for var in required_vars:
    if not os.getenv(var):
        logger.error(f"Missing required environment variable: {var}")
    else:
        value = os.getenv(var)
        masked_value = value[:5] + "*" * (len(value) - 5) if len(value) > 5 else "*****"
        logger.info(f"Loaded {var}: {masked_value}")

# Global session tracking
active_sessions = {}

class SimplifiedAssistant(Agent):
    def __init__(self, session_id: str) -> None:
        self.session_id = session_id
        self.conversation_stage = "greeting"
        self.transcripts: list[str] = []
        self.full_transcript: str = ""
        
        super().__init__(instructions=self._get_instructions())

    def _get_instructions(self) -> str:
        """Generate instructions for the voice assistant"""
        return f"""
You are Scott, the AI voice assistant for Alive5 Support (Session: {self.session_id}). 

Your goal is to have natural conversations and help users with:
- SALES: Questions about pricing, plans, demos, or team licenses
- SUPPORT: Technical issues, troubleshooting, or how-to questions  
- BILLING: Invoices, payments, account management, or subscription changes

CONVERSATION FLOW:
1. Start with a warm greeting
2. Listen and respond naturally to user needs
3. Provide helpful information and solutions
4. Collect necessary information when appropriate
5. Offer escalation to human agents when needed

TONE: Professional, helpful, empathetic, and conversational
VOICE: Clear, warm, engaging
LANGUAGE: Simple, jargon-free, concise responses
AVOID: Technical jargon, complex explanations, lengthy responses, and unnecessary characters like '*' or '&'.

IMPORTANT RULES:
- Always confirm you're listening when asked
- Be honest about limitations and offer human escalation when needed
- Keep responses focused and conversational
- End calls gracefully when resolution is achieved
- Never make promises about pricing or technical capabilities you're unsure about

Current conversation stage: {self.conversation_stage}
"""

    async def conversation_item_added(self, event):
        """Handle conversation item added events for transcript logging"""
        try:
            # Check if this is a user message (speech-to-text result)
            if event.item.role == "user" and event.item.content:
                user_message = event.item.content
                logger.info(f"TRANSCRIPT: User said: '{user_message}'")
                
                # Add to transcript history for context
                self.transcripts.append(user_message)
                if self.full_transcript:
                    self.full_transcript += f"\nUser: {user_message}"
                else:
                    self.full_transcript = f"User: {user_message}"
                
                # Note: Intent detection is now handled by the backend via frontend calls
                logger.info(f"SESSION_UPDATE: Message logged for session {self.session_id}")
                
        except Exception as e:
            logger.error(f"Error processing conversation item: {e}")

    def update_conversation_context(self, stage: str):
        """Update conversation stage"""
        self.conversation_stage = stage
        logger.info(f"CONVERSATION_STAGE: Session {self.session_id} moved to stage: {stage}")

def prewarm(proc):
    """Preload models for better performance"""
    proc.userdata["vad"] = silero.VAD.load()

async def entrypoint(ctx: JobContext):
    """Main entry point for the dynamic voice agent"""
    session_id = str(uuid.uuid4())[:8]
    room_name = ctx.room.name
    
    if room_name in active_sessions:
        logger.warning(f"Room {room_name} already has an active session. Skipping.")
        return
        
    active_sessions[room_name] = session_id
    logger.info(f"Starting dynamic agent session {session_id} in room: {room_name}")
    
    agent_session = None
    assistant = SimplifiedAssistant(session_id)
    
    try:
        await ctx.connect(auto_subscribe=AutoSubscribe.AUDIO_ONLY)
        logger.info(f"Connected to room {room_name}")
        
        try:
            participant = await asyncio.wait_for(ctx.wait_for_participant(), timeout=600.0)
            logger.info(f"Participant {participant.identity} joined session {session_id}")
            
            await asyncio.sleep(2)
        except asyncio.TimeoutError:
            logger.warning(f"No participant joined session {session_id} within timeout")
            return

        # Create AgentSession with proper configuration
        agent_session = AgentSession(
            stt=deepgram.STT(
                model="nova-2",
                language="en-US",
                api_key=os.getenv("DEEPGRAM_API_KEY")
            ),
            llm=openai.LLM(
                model="gpt-4o",
                temperature=0.7,
                api_key=os.getenv("OPENAI_API_KEY")
            ),
            tts=cartesia.TTS(
                model="sonic-english",
                voice="a0e99841-438c-4a64-b679-ae501e7d6091",
                api_key=os.getenv("CARTESIA_API_KEY")
            ),
            vad=ctx.proc.userdata["vad"],
            turn_detection=MultilingualModel(),
            # Enable transcription forwarding to frontend (enabled by default)
            use_tts_aligned_transcript=True,  # Better transcript synchronization
        )
        
        # Start the session with proper room options
        await agent_session.start(
            room=ctx.room,
            agent=assistant,
            room_input_options=RoomInputOptions(
                noise_cancellation=noise_cancellation.BVC(),
                text_enabled=True,  # Enable text input
            ),
            room_output_options=RoomOutputOptions(
                transcription_enabled=True,  # Enable transcript output to frontend
                sync_transcription=True,  # Synchronize transcripts with speech
            ),
        )
        
        logger.info(f"Dynamic agent session started for {session_id}")
        
        # Initial greeting
        greeting_message = "Hello! I'm Scott from Alive5. How can I help you today? Are you looking for sales information, technical support, or have questions about billing?"
        
        await agent_session.generate_reply(
            instructions=f"Say exactly: '{greeting_message}'"
        )
        
        logger.info(f"Initial greeting sent for session {session_id}")
        
        # Keep session alive while participants are connected
        while ctx.room.remote_participants:
            await asyncio.sleep(1)
            logger.debug(f"Session {session_id} active - Participants connected")
        
    except Exception as e:
        logger.error(f"Error in session {session_id}: {str(e)}")
        raise
    finally:
        # Cleanup
        if room_name in active_sessions:
            del active_sessions[room_name]
        
        logger.info(f"Session {session_id} summary - Stage: {assistant.conversation_stage}")
        
        if agent_session:
            try:
                await agent_session.aclose()
                logger.info(f"Agent session {session_id} closed properly")
            except Exception as e:
                logger.error(f"Error closing session {session_id}: {e}")
        
        if ctx.room and ctx.room.connection_state == "connected":
            await ctx.room.disconnect()
            
        logger.info(f"Dynamic session {session_id} ended and cleaned up")

if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1 and sys.argv[1] == "test":
        # Test the assistant
        assistant = SimplifiedAssistant("test-session")
        print(f"Assistant created for session: {assistant.session_id}")
        print("Worker is now simplified - intent detection handled by backend")
    else:
        cli.run_app(
            WorkerOptions(
                entrypoint_fnc=entrypoint,
                prewarm_fnc=prewarm,
            ),
        )