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
    def __init__(self, session_id: str) -> None:
        self.session_id = session_id
        self.conversation_stage = "greeting"
        self.transcripts: list[str] = []
        self.full_transcript: str = ""
        self.room = None
        self.current_flow_state = None
        self.retry_manager = None
        self.backend_healthy = None
        self.greeting_sent = False  # Track if greeting has been sent
        self.conversation_history = []  # Track full conversation
        
        super().__init__(instructions=self._get_instructions())

    def _get_instructions(self) -> str:
        """Generate instructions for the flow-based voice assistant"""
        return f"""
You are Scott, the AI voice assistant for Alive5 Support (Session: {self.session_id}). 

CRITICAL FLOW-BASED OPERATION:
- You are operating in FLOW-CONTROLLED mode
- Do NOT generate automatic responses to user messages
- ONLY respond when explicitly instructed by the flow system
- Wait for flow processing to provide exact response text

RESPONSE PROTOCOL:
1. User speaks → Transcription captured
2. Backend processes through flow system
3. Flow system returns exact response text
4. You speak ONLY that exact text

PERSONALITY when instructed to respond:
- Professional, helpful, empathetic
- Clear, warm, engaging voice
- Simple, jargon-free language
- Concise but complete responses

IMPORTANT: Never add extra content or modify flow responses.

Current stage: {self.conversation_stage}
"""

    async def conversation_item_added(self, event):
        """Handle conversation items with strict flow control"""
        try:
            if event.item.role == "user" and event.item.content:
                user_message = event.item.content.strip()
                if not user_message:
                    return
                    
                logger.info(f"🎤 USER SPEECH: '{user_message}'")
                
                # Add to conversation history
                self.conversation_history.append({
                    "role": "user", 
                    "content": user_message,
                    "timestamp": datetime.now().isoformat()
                })
                
                # Check for conversation end
                if self.should_end_conversation(user_message):
                    logger.info(f"👋 FAREWELL DETECTED: '{user_message}'")
                    await self.handle_conversation_end(user_message)
                    return
                
                # Process through flow system ONLY
                await self.process_through_flow_system(user_message)
                
            elif event.item.role == "assistant" and event.item.content:
                # Track assistant responses
                assistant_message = event.item.content.strip()
                logger.info(f"🤖 ASSISTANT SPEECH: '{assistant_message}'")
                
                # Add to conversation history
                self.conversation_history.append({
                    "role": "assistant",
                    "content": assistant_message, 
                    "timestamp": datetime.now().isoformat()
                })
                
                # Send to frontend for display
                await self.send_agent_transcript(assistant_message)
                
        except Exception as e:
            logger.error(f"Error in conversation_item_added: {e}")

    async def process_through_flow_system(self, user_message: str):
        """Process user message strictly through backend flow system"""
        try:
            # Check backend health
            backend_healthy = await check_backend_health()
            if not backend_healthy:
                logger.warning("❌ Backend not accessible")
                await self.signal_worker_status("backend_down", "Backend server is not accessible")
                
                if self.retry_manager and not self.retry_manager.is_retrying:
                    await self.retry_manager.start_retry_loop()
                
                await self.handle_backend_down_response()
                return
            
            room_name = self.room.name if self.room else "unknown"
            
            # Send to backend flow processor with conversation history
            logger.info(f"🔄 BACKEND REQUEST: Room={room_name}, Message='{user_message}'")
            
            async with httpx.AsyncClient(timeout=BACKEND_TIMEOUT) as client:
                payload = {
                    "room_name": room_name,
                    "user_message": user_message,
                    "conversation_history": self.conversation_history[-10:]  # Last 10 messages
                }
                
                response = await client.post(
                    f"{BACKEND_URL}/api/process_flow_message",
                    headers={"Content-Type": "application/json"},
                    json=payload
                )
                
                if response.status_code == 200:
                    flow_result = response.json()
                    logger.info(f"✅ FLOW RESULT: {flow_result}")
                    await self.handle_flow_response(flow_result)
                else:
                    logger.error(f"❌ Backend error: {response.status_code} - {response.text}")
                    await self.handle_backend_error_response()
                    
        except Exception as e:
            logger.error(f"❌ Flow processing error: {e}")
            await self.handle_backend_error_response()

    async def handle_flow_response(self, flow_result: Dict[str, Any]):
        """Handle response from flow system and generate appropriate speech"""
        try:
            flow_data = flow_result.get("flow_result", {})
            flow_type = flow_data.get("type", "unknown")
            response_text = flow_data.get("response", "")
            
            logger.info(f"🎯 PROCESSING FLOW TYPE: {flow_type}")
            logger.info(f"📝 RESPONSE TEXT: '{response_text}'")
            
            if not response_text or response_text.strip() == "":
                logger.warning("⚠️ Empty response from flow system")
                return
            
            # Handle different flow types
            if flow_type == "flow_started":
                flow_name = flow_data.get("flow_name", "Unknown")
                logger.info(f"🚀 FLOW STARTED: {flow_name}")
                self.conversation_stage = f"flow_{flow_name.lower()}"
                
            elif flow_type in ["agent_transfer", "agent"]:
                logger.info("👥 AGENT TRANSFER REQUESTED")
                # Generate response and then end conversation
                await self.generate_speech_response(response_text)
                await asyncio.sleep(3)
                await self.handle_conversation_end("Agent transfer completed")
                return
                
            elif flow_type == "faq_response":
                logger.info("❓ FAQ RESPONSE")
                self.conversation_stage = "faq_interaction"
                
            elif flow_type == "error":
                logger.error("❌ FLOW ERROR")
                
            # Generate speech for any response with text
            if response_text.strip():
                await self.generate_speech_response(response_text)
                
        except Exception as e:
            logger.error(f"Error handling flow response: {e}")

    async def generate_speech_response(self, response_text: str):
        """Generate speech response using the provided text"""
        try:
            logger.info(f"🗣️ GENERATING SPEECH: '{response_text}'")
            
            # Create specific instructions for the LLM to say exactly what the flow dictated
            flow_instructions = f"""
You must respond with exactly this message from the flow system:

"{response_text}"

Say this naturally as Scott from Alive5, but do not add any additional content, questions, or modifications. 
Speak exactly what is provided above.
"""
            
            # Use the agent session to generate speech
            if hasattr(self, 'agent_session') and self.agent_session:
                await self.agent_session.generate_reply(flow_instructions)
            else:
                logger.error("No agent session available for speech generation")
            
        except Exception as e:
            logger.error(f"Error generating speech response: {e}")

    async def handle_backend_down_response(self):
        """Handle response when backend is down"""
        fallback_message = "I'm experiencing some technical difficulties. Let me connect you to a human agent who can help you right away."
        await self.generate_speech_response(fallback_message)

    async def handle_backend_error_response(self):
        """Handle response when backend returns an error"""
        error_message = "I apologize, but I'm having trouble processing your request. Let me connect you to one of our specialists who can assist you better."
        await self.generate_speech_response(error_message)

    def should_end_conversation(self, user_message: str) -> bool:
        """Enhanced farewell detection"""
        message_lower = user_message.lower().strip()
        
        farewell_patterns = [
            # Direct endings
            "bye", "goodbye", "good bye", "see you", "farewell", "later",
            "that's all", "thats all", "that is all", "i'm done", "im done",
            
            # Completion signals  
            "thank you bye", "thanks bye", "perfect thanks", "great thanks",
            "that helps thanks", "got it thanks", "ok bye", "okay bye",
            
            # Explicit commands
            "end call", "hang up", "disconnect", "finish", "close",
            "gotta go", "need to go", "have to go", "talk later"
        ]
        
        # Check for patterns
        for pattern in farewell_patterns:
            if pattern in message_lower:
                logger.info(f"🔍 FAREWELL MATCH: '{pattern}' in '{message_lower}'")
                return True
        
        # Short thank you messages likely indicate ending
        if ("thank" in message_lower or "thanks" in message_lower) and len(message_lower.split()) <= 4:
            return True
            
        return False

    async def handle_conversation_end(self, user_message: str):
        """Handle conversation ending gracefully"""
        try:
            logger.info(f"👋 ENDING CONVERSATION: {user_message}")
            
            # Generate appropriate farewell
            farewell_responses = [
                "Thank you for contacting Alive5! Have a great day and feel free to reach out anytime.",
                "Perfect! Thanks for using Alive5 support. Take care!",
                "You're all set! Thanks for calling Alive5 and have a wonderful day."
            ]
            
            # Pick appropriate farewell based on context
            farewell = farewell_responses[0]  # Default
            if "thank" in user_message.lower():
                farewell = farewell_responses[1]
            elif any(word in user_message.lower() for word in ["help", "good", "great", "perfect"]):
                farewell = farewell_responses[2]
            
            await self.generate_speech_response(farewell)
            
            # Wait for farewell to complete
            await asyncio.sleep(3)
            
            # Signal disconnection
            await self.send_disconnection_signal()
            
        except Exception as e:
            logger.error(f"Error handling conversation end: {e}")

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
    
    # Create assistant
    assistant = FlowBasedAssistant(session_id)
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

        # Create agent session
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
        
        # Send initial greeting through flow system (simulate user joining)
        await asyncio.sleep(1)
        initial_greeting = "Hello! I'm Scott from Alive5. How can I help you today?"
        
        # Send greeting directly and add to conversation history
        assistant.conversation_history.append({
            "role": "assistant",
            "content": initial_greeting,
            "timestamp": datetime.now().isoformat()
        })
        
        await assistant.send_agent_transcript(initial_greeting)
        await assistant.generate_speech_response(initial_greeting)
        
        logger.info(f"👋 Initial greeting sent for {session_id}")
        
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