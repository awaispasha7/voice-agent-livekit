import logging
import os
import uuid
import asyncio
import httpx
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
from livekit.agents import llm
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
logger = logging.getLogger("flow-based-voice-agent")
logger.setLevel(logging.INFO)

# Reduce noise from some verbose loggers
logging.getLogger("livekit.agents.utils.aio.duplex_unix").setLevel(logging.WARNING)
logging.getLogger("livekit.agents.cli.watcher").setLevel(logging.WARNING)
logging.getLogger("livekit.agents.ipc.channel").setLevel(logging.WARNING)

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

class FlowBasedLLM(llm.LLM):
    """Custom LLM that intercepts responses and routes through flow system"""
    
    def __init__(self, backend_url: str, api_key: str):
        super().__init__()
        self.backend_url = backend_url
        self.api_key = api_key
        self.room_name = None
        
    def set_room_name(self, room_name: str):
        """Set the room name for this LLM instance"""
        self.room_name = room_name
        
    async def generate_response(self, chat_context: llm.ChatContext) -> llm.LLMStream:
        """Main method to override - intercepts all LLM responses"""
        try:
            # Get the latest user message
            if not chat_context.messages:
                yield llm.LLMResponse(text="Hello! I'm Scott from Alive5. How can I help you today?")
                return
                
            last_message = chat_context.messages[-1].text
            logger.info(f"🎤 CUSTOM LLM: Processing message: '{last_message}'")
            
            # Convert chat context to conversation history
            conversation_history = []
            for msg in chat_context.messages:
                conversation_history.append({
                    "role": msg.role,
                    "content": msg.text,
                    "timestamp": datetime.now().isoformat()
                })
            
            # Process through backend flow system
            response_text = await self.process_through_backend(last_message, conversation_history)
            
            # Yield the response
            yield llm.LLMResponse(text=response_text)
            
        except Exception as e:
            logger.error(f"Error in custom LLM: {e}")
            yield llm.LLMResponse(text="I apologize, but I'm having trouble processing your request. Let me connect you to a human agent.")
    
    async def process_through_backend(self, user_message: str, conversation_history: list) -> str:
        """Process user message through backend flow system"""
        try:
            if not self.room_name:
                return "I'm experiencing technical difficulties. Please try again."
                
            # Check backend health
            backend_healthy = await check_backend_health()
            if not backend_healthy:
                return "I'm experiencing some technical difficulties. Let me connect you to a human agent who can help you right away."
            
            # Send to backend flow processor
            logger.info(f"🔄 BACKEND REQUEST: Room={self.room_name}, Message='{user_message}'")
            
            async with httpx.AsyncClient(timeout=BACKEND_TIMEOUT) as client:
                payload = {
                    "room_name": self.room_name,
                    "user_message": user_message,
                    "conversation_history": conversation_history[-10:]  # Last 10 messages
                }
                
                response = await client.post(
                    f"{self.backend_url}/api/process_flow_message",
                    headers={"Content-Type": "application/json"},
                    json=payload
                )
                
                if response.status_code == 200:
                    flow_result = response.json()
                    logger.info(f"✅ FLOW RESULT: {flow_result}")
                    
                    # Extract response text from flow result
                    flow_data = flow_result.get("flow_result", {})
                    response_text = flow_data.get("response", "")
                    
                    if not response_text or response_text.strip() == "":
                        return "I apologize, but I'm having trouble processing your request. Let me connect you to one of our specialists who can assist you better."
                    
                    return response_text
                else:
                    logger.error(f"❌ Backend error: {response.status_code} - {response.text}")
                    return "I apologize, but I'm having trouble processing your request. Let me connect you to one of our specialists who can assist you better."
                    
        except Exception as e:
            logger.error(f"❌ Flow processing error: {e}")
            return "I apologize, but I'm having trouble processing your request. Let me connect you to one of our specialists who can assist you better."
    
    async def chat(self, chat_context: llm.ChatContext) -> llm.LLMStream:
        """Required method for LLM class - delegates to generate_response"""
        return self.generate_response(chat_context)


# Backend configuration
BACKEND_URL = "https://voice-agent-livekit-backend-9f8ec30b9fba.herokuapp.com"
BACKEND_TIMEOUT = 15  # Increased timeout
BACKEND_RETRY_INTERVAL = 30
BACKEND_MAX_RETRIES = 10

async def check_backend_health() -> bool:
    """Check if the backend is accessible"""
    try:
        async with httpx.AsyncClient(timeout=BACKEND_TIMEOUT) as client:
            response = await client.get(f"{BACKEND_URL}/health")
            if response.status_code == 200:
                logger.info("✅ Backend health check passed")
                return True
            else:
                logger.warning(f"⚠️ Backend health check failed: {response.status_code}")
                return False
    except Exception as e:
        logger.error(f"❌ Backend health check failed: {str(e)}")
        return False

class BackendRetryManager:
    """Manages automatic retry attempts for backend connection"""
    
    def __init__(self, assistant):
        self.assistant = assistant
        self.retry_count = 0
        self.is_retrying = False
        self.retry_task = None
    
    async def start_retry_loop(self):
        """Start the retry loop in the background"""
        if self.is_retrying:
            return
        
        self.is_retrying = True
        self.retry_task = asyncio.create_task(self._retry_loop())
        logger.info("🔄 Backend retry loop started")
    
    async def stop_retry_loop(self):
        """Stop the retry loop"""
        self.is_retrying = False
        if self.retry_task:
            self.retry_task.cancel()
            try:
                await self.retry_task
            except asyncio.CancelledError:
                pass
        logger.info("🛑 Backend retry loop stopped")
    
    async def _retry_loop(self):
        """Main retry loop"""
        while self.is_retrying and self.retry_count < BACKEND_MAX_RETRIES:
            try:
                await asyncio.sleep(BACKEND_RETRY_INTERVAL)
                
                if not self.is_retrying:
                    break
                
                logger.info(f"🔄 Retry attempt {self.retry_count + 1}/{BACKEND_MAX_RETRIES}")
                
                backend_healthy = await check_backend_health()
                if backend_healthy:
                    logger.info("✅ Backend is back online!")
                    await self.assistant.signal_worker_status("reconnected", "Backend connection restored")
                    self.retry_count = 0
                    self.is_retrying = False
                    break
                else:
                    self.retry_count += 1
                    logger.warning(f"⚠️ Backend still down, retry {self.retry_count}/{BACKEND_MAX_RETRIES}")
                    await self.assistant.signal_worker_status("retrying", f"Retrying backend connection ({self.retry_count}/{BACKEND_MAX_RETRIES})")
                    
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in retry loop: {e}")
                self.retry_count += 1
        
        if self.retry_count >= BACKEND_MAX_RETRIES:
            logger.error("❌ Maximum retry attempts reached. Backend appears to be permanently down.")
            await self.assistant.signal_worker_status("failed", "Backend connection failed after maximum retries")
            self.is_retrying = False

class FlowBasedAssistant(Agent):
    def __init__(self, session_id: str, custom_llm: FlowBasedLLM) -> None:
        self.session_id = session_id
        self.room = None
        self.retry_manager = None
        self.backend_healthy = None
        
        # Use the custom LLM instead of default
        super().__init__(
            instructions="Flow-based voice assistant for Alive5 Support",
            llm=custom_llm
        )



    async def send_disconnection_signal(self):
        """Send disconnection signal to frontend"""
        try:
            if self.room:
                await self.room.local_participant.publish_data(
                    json.dumps({
                        'type': 'conversation_end',
                        'message': 'Conversation ended gracefully',
                        'timestamp': datetime.now().isoformat(),
                        'reason': 'user_requested'
                    }).encode(),
                    topic="lk.conversation.control"
                )
                logger.info("📡 DISCONNECT SIGNAL SENT")
        except Exception as e:
            logger.error(f"Error sending disconnect signal: {e}")

    async def send_agent_transcript(self, message: str):
        """Send agent transcript to frontend for display"""
        try:
            if self.room:
                await self.room.local_participant.publish_data(
                    json.dumps({
                        'type': 'agent_transcript',
                        'message': message,
                        'speaker': 'Scott_AI_Agent',
                        'timestamp': datetime.now().isoformat()
                    }).encode(),
                    topic="lk.agent.transcript"
                )
                logger.info(f"📋 AGENT TRANSCRIPT SENT: '{message[:50]}...'")
        except Exception as e:
            logger.error(f"Error sending agent transcript: {e}")

    async def signal_worker_status(self, status: str, message: str = ""):
        """Signal worker status to frontend"""
        try:
            if self.room:
                await self.room.local_participant.set_metadata(json.dumps({
                    "worker_status": status,
                    "message": message,
                    "timestamp": datetime.now().isoformat(),
                    "session_id": self.session_id
                }))
                logger.info(f"🚦 STATUS SIGNAL: {status} - {message}")
        except Exception as e:
            logger.error(f"Error signaling status: {e}")

def prewarm(proc):
    """Preload models for better performance"""
    proc.userdata["vad"] = silero.VAD.load()

async def entrypoint(ctx: JobContext):
    """Main entry point for the flow-based voice agent"""
    session_id = str(uuid.uuid4())[:8]
    room_name = ctx.room.name
    
    if room_name in active_sessions:
        logger.warning(f"⚠️ Room {room_name} already has active session")
        return
    
    # Create custom LLM
    custom_llm = FlowBasedLLM(BACKEND_URL, "dummy_key")
    custom_llm.set_room_name(room_name)
    
    # Create assistant with custom LLM
    assistant = FlowBasedAssistant(session_id, custom_llm)
    active_sessions[room_name] = session_id
    logger.info(f"🚀 STARTING SESSION {session_id} in room {room_name}")
    
    # Check backend health
    logger.info("🔍 Checking backend health...")
    backend_healthy = await check_backend_health()
    assistant.backend_healthy = backend_healthy
    
    if not backend_healthy:
        logger.error("❌ Backend health check failed")
    else:
        logger.info("✅ Backend health check passed")
    
    agent_session = None
    
    try:
        # Connect to room
        await ctx.connect(auto_subscribe=AutoSubscribe.AUDIO_ONLY)
        logger.info(f"🔗 Connected to room {room_name}")
        
        # Set up assistant references
        assistant.room = ctx.room
        assistant.retry_manager = BackendRetryManager(assistant)
        
        # Send initial status
        if backend_healthy:
            await assistant.signal_worker_status("ready", "Worker ready, backend accessible")
        else:
            await assistant.signal_worker_status("backend_down", "Backend not accessible")
        
        # Wait for participant
        try:
            participant = await asyncio.wait_for(ctx.wait_for_participant(), timeout=600.0)
            logger.info(f"👤 Participant {participant.identity} joined")
            await asyncio.sleep(2)
        except asyncio.TimeoutError:
            logger.warning("⏰ No participant joined within timeout")
            return

        # Create agent session with custom LLM
        agent_session = AgentSession(
            stt=deepgram.STT(
                model="nova-2",
                language="en-US",
                api_key=os.getenv("DEEPGRAM_API_KEY")
            ),
            llm=custom_llm,  # Use our custom LLM
            tts=cartesia.TTS(
                model="sonic-english",
                voice="a0e99841-438c-4a64-b679-ae501e7d6091", 
                api_key=os.getenv("CARTESIA_API_KEY")
            ),
            vad=ctx.proc.userdata["vad"],
            turn_detection=MultilingualModel(),
        )
        
        # Start session
        await agent_session.start(
            room=ctx.room,
            agent=assistant,
            room_input_options=RoomInputOptions(
                noise_cancellation=noise_cancellation.BVC(),
                text_enabled=True,
            ),
            room_output_options=RoomOutputOptions(
                transcription_enabled=True,
                sync_transcription=False,
            ),
        )
        
        assistant.agent_session = agent_session
        logger.info(f"🎙️ Agent session started for {session_id}")
        
        # The custom LLM will handle the initial greeting automatically
        logger.info(f"👋 Custom LLM will handle initial greeting for {session_id}")
        
        # Keep session alive
        try:
            while ctx.room.remote_participants:
                await asyncio.sleep(1)
        except Exception as e:
            logger.warning(f"⚠️ Session monitoring interrupted: {e}")
        
    except Exception as e:
        logger.error(f"❌ Session error {session_id}: {e}", exc_info=True)
    finally:
        # Cleanup
        logger.info(f"🧹 Cleaning up session {session_id}")
        
        # Remove from active sessions
        if room_name in active_sessions:
            del active_sessions[room_name]
            
        # Stop retry manager
        if assistant.retry_manager:
            try:
                await assistant.retry_manager.stop_retry_loop()
            except Exception as e:
                logger.error(f"Error stopping retry manager: {e}")
        
        # Close agent session
        if agent_session:
            try:
                await asyncio.wait_for(agent_session.aclose(), timeout=10.0)
                logger.info(f"✅ Agent session closed: {session_id}")
            except Exception as e:
                logger.error(f"Error closing agent session: {e}")
        
        # Disconnect room
        try:
            if ctx.room and ctx.room.connection_state == "connected":
                await asyncio.wait_for(ctx.room.disconnect(), timeout=5.0)
        except Exception as e:
            logger.error(f"Error disconnecting room: {e}")
            
        logger.info(f"🏁 Session {session_id} cleanup completed")

if __name__ == "__main__":
    cli.run_app(
        WorkerOptions(
            entrypoint_fnc=entrypoint,
            prewarm_fnc=prewarm,
        ),
    )