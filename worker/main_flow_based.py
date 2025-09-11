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
BACKEND_TIMEOUT = 10  # seconds
BACKEND_RETRY_INTERVAL = 30  # seconds
BACKEND_MAX_RETRIES = 10  # maximum retry attempts

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
        self.room = None  # Will be set by entrypoint
        self.current_flow_state = None
        self.retry_manager = None  # Will be set after room is available
        self.backend_healthy = None  # Will be set during initialization
        
        super().__init__(instructions=self._get_instructions())

    def _get_instructions(self) -> str:
        """Generate instructions for the flow-based voice assistant"""
        return f"""
You are Scott, the AI voice assistant for Alive5 Support (Session: {self.session_id}). 

CRITICAL: You are now using a FLOW-BASED system. DO NOT respond to user messages automatically.

RESPONSE PROTOCOL:
- ONLY respond when you receive explicit instructions starting with "You must say exactly this"
- DO NOT generate your own responses to user questions
- WAIT for the flow system to provide the exact response
- When given explicit instructions, say exactly what you're told

FLOW SYSTEM:
- The backend processes user messages and determines the correct response
- You will receive specific instructions for each response
- Follow these instructions exactly without adding your own content

TONE: Professional, helpful, empathetic, and conversational when instructed
VOICE: Clear, warm, engaging
LANGUAGE: Simple, jargon-free, concise responses
AVOID: Automatic responses, adding extra content, technical jargon

IMPORTANT: Do not respond to user questions on your own. Wait for explicit instructions.

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
                
                # Check if user wants to end the conversation
                if self.should_end_conversation(user_message):
                    logger.info(f"END_CONVERSATION: User indicated they want to end the call: '{user_message}'")
                    await self.handle_conversation_end(user_message)
                    return
                
                # Process through flow system
                await self.process_user_message_through_flow(user_message)
                
                logger.info(f"FLOW_PROCESSING: Message processed for session {self.session_id}")
            
            # Check if this is an assistant response
            elif event.item.role == "assistant" and event.item.content:
                assistant_message = event.item.content
                logger.info(f"TRANSCRIPT: Assistant said: '{assistant_message}'")
                
                # Send assistant response as a separate transcription message
                await self.send_agent_transcript(assistant_message)
                
        except Exception as e:
            logger.error(f"Error processing conversation item: {e}")

    async def process_user_message_through_flow(self, user_message: str):
        """Process user message through the flow system"""
        try:
            # Check backend health first
            backend_healthy = await check_backend_health()
            if not backend_healthy:
                logger.warning("Backend not accessible, using fallback response")
                await self.signal_worker_status("backend_down", "Backend server is not accessible")
                
                # Start retry loop if not already running
                if self.retry_manager and not self.retry_manager.is_retrying:
                    await self.retry_manager.start_retry_loop()
                
                await self.handle_fallback_response(user_message)
                return
            
            # Get room name from the room object
            room_name = self.room.name if self.room else "unknown"
            
            # Call the backend flow processing endpoint
            async with httpx.AsyncClient(timeout=BACKEND_TIMEOUT) as client:
                response = await client.post(
                    f"{BACKEND_URL}/api/process_flow_message",
                    headers={
                        "Content-Type": "application/json",
                    },
                    json={
                        "room_name": room_name,
                        "user_message": user_message
                    }
                )
                
                if response.status_code == 200:
                    flow_result = response.json()
                    logger.info(f"FLOW_RESULT: {flow_result}")
                    
                    # Process the flow result
                    await self.handle_flow_result(flow_result)
                else:
                    logger.error(f"Flow processing failed: {response.status_code} - {response.text}")
                    # Fallback to simple response
                    await self.handle_fallback_response(user_message)
                    
        except Exception as e:
            logger.error(f"Error processing flow message: {e}")
            # Fallback to simple response
            await self.handle_fallback_response(user_message)

    async def handle_flow_result(self, flow_result: Dict[str, Any]):
        """Handle the result from the flow processing - fully dynamic"""
        try:
            flow_data = flow_result.get("flow_result", {})
            flow_type = flow_data.get("type", "unknown")
            response_text = flow_data.get("response", "")
            
            logger.info(f"FLOW_HANDLER: Processing flow type: {flow_type}")
            logger.info(f"FLOW_HANDLER: Response text: '{response_text}'")
            logger.info(f"FLOW_HANDLER: Full flow result: {flow_result}")
            
            # Dynamic flow type handling - works with any flow type
            if flow_type == "flow_started":
                # A new flow has started
                flow_name = flow_data.get("flow_name", "Unknown")
                logger.info(f"FLOW_STARTED: {flow_name}")
                self.conversation_stage = f"flow_{flow_name.lower()}"
                
            elif flow_type == "agent_transfer" or flow_type == "agent":
                # Transfer to human agent
                logger.info("FLOW_AGENT_TRANSFER: Transferring to human agent")
                if response_text:
                    await self.generate_flow_response(response_text)
                await asyncio.sleep(3)
                await self.handle_conversation_end("Agent transfer requested")
                return
                
            elif flow_type == "error":
                # Error occurred
                logger.error("FLOW_ERROR: Error in flow processing")
                
            else:
                # Handle any other flow type dynamically
                logger.info(f"FLOW_DYNAMIC: Processing {flow_type} step")
            
            # Generate response for any flow type that has text
            if response_text:
                logger.info(f"FLOW_HANDLER: About to call generate_flow_response with: '{response_text}'")
                await self.generate_flow_response(response_text)
                logger.info(f"FLOW_HANDLER: generate_flow_response completed")
            else:
                logger.warning(f"FLOW_HANDLER: No response text for flow type: {flow_type}")
                    
        except Exception as e:
            logger.error(f"Error handling flow result: {e}")

    async def generate_flow_response(self, response_text: str):
        """Generate a response using the flow text"""
        try:
            if hasattr(self, 'agent_session') and self.agent_session:
                logger.info(f"FLOW_RESPONSE: About to generate response: '{response_text}'")
                # Use a more direct instruction to ensure the exact text is spoken
                await self.agent_session.generate_reply(
                    instructions=f"You must say exactly this and nothing else: '{response_text}'. Do not add any additional context or explanations."
                )
                logger.info(f"FLOW_RESPONSE: Generated response: '{response_text}'")
            else:
                logger.warning("No agent session available for response generation")
        except Exception as e:
            logger.error(f"Error generating flow response: {e}")

    async def handle_fallback_response(self, user_message: str):
        """Handle fallback response when flow processing fails"""
        try:
            fallback_response = "I'm sorry, I'm having trouble processing your request. Let me connect you to a human agent who can help you better."
            
            if hasattr(self, 'agent_session') and self.agent_session:
                await self.agent_session.generate_reply(
                    instructions=f"Say exactly: '{fallback_response}'"
                )
                logger.info(f"FALLBACK_RESPONSE: Generated fallback response")
        except Exception as e:
            logger.error(f"Error generating fallback response: {e}")
    
    async def signal_worker_status(self, status: str, message: str = ""):
        """Signal worker status to frontend via room data"""
        try:
            if self.room:
                # Send status as room metadata
                await self.room.local_participant.set_metadata(json.dumps({
                    "worker_status": status,
                    "message": message,
                    "timestamp": datetime.now().isoformat(),
                    "session_id": self.session_id
                }))
                logger.info(f"WORKER_STATUS_SIGNAL: {status} - {message}")
        except Exception as e:
            logger.error(f"Error signaling worker status: {e}")

    def should_end_conversation(self, user_message: str) -> bool:
        """Check if user message indicates they want to end the conversation"""
        message_lower = user_message.lower().strip()
        logger.info(f"FAREWELL_CHECK: Analyzing message: '{user_message}' -> '{message_lower}'")
        
        # Common farewell phrases
        farewell_phrases = [
            # Direct goodbyes
            "bye", "goodbye", "good bye", "see you", "see ya", "later", "farewell",
            "that's all", "thats all", "that is all", "that'll be all", "that will be all",
            
            # Completion phrases
            "i'm done", "im done", "i am done", "we're done", "were done", "we are done",
            "that's it", "thats it", "that is it", "that's everything", "thats everything",
            "nothing else", "no more questions", "i'm good", "im good", "i am good",
            
            # Thank you + ending
            "thank you, bye", "thanks, bye", "thank you goodbye", "thanks goodbye",
            "thank you that's all", "thanks thats all", "appreciate it bye", "thanks for your help bye",
            
            # Explicit endings
            "end call", "end the call", "hang up", "disconnect", "finish", "close",
            "i have to go", "gotta go", "need to go", "have to run", "talk later",
            
            # Satisfied responses
            "perfect thank you", "great thanks", "awesome thanks", "that helps thanks",
            "got it thanks", "understood thanks", "ok bye", "okay bye", "alright bye"
        ]
        
        # Check for exact matches or if the message starts with these phrases
        for phrase in farewell_phrases:
            if phrase in message_lower or message_lower.startswith(phrase):
                logger.info(f"FAREWELL_DETECTED: Matched phrase: '{phrase}' in message: '{message_lower}'")
                return True
        
        # Check for patterns like "thanks" + short response (likely ending)
        if ("thank" in message_lower or "thanks" in message_lower) and len(message_lower.split()) <= 3:
            logger.info(f"FAREWELL_DETECTED: Short thanks message: '{message_lower}'")
            return True
            
        logger.info(f"FAREWELL_CHECK: No farewell detected in: '{message_lower}'")
        return False

    async def handle_conversation_end(self, user_message: str):
        """Handle the end of conversation gracefully"""
        try:
            # Generate a polite farewell response
            farewell_instructions = f"""
The user has indicated they want to end the conversation by saying: "{user_message}"

Please provide a brief, warm farewell response that:
1. Acknowledges their request to end the call
2. Thanks them for using Alive5
3. Offers future assistance if needed
4. Keeps it short and natural (1-2 sentences max)

Examples:
- "Thank you for contacting Alive5! Have a great day and feel free to reach out anytime."
- "Perfect! Thanks for using Alive5 support. Take care!"
- "You're all set! Thanks for calling Alive5 and have a wonderful day."
"""
            
            # Send farewell message through the agent session
            if hasattr(self, 'agent_session') and self.agent_session:
                await self.agent_session.generate_reply(instructions=farewell_instructions)
                logger.info(f"FAREWELL: Sent goodbye message for session {self.session_id}")
                
                # Wait a moment for the farewell to be delivered
                await asyncio.sleep(2)
            
            # Signal for disconnection via data message
            await self.send_disconnection_signal()
            
        except Exception as e:
            logger.error(f"Error handling conversation end: {e}")

    async def send_disconnection_signal(self):
        """Send signal to frontend to disconnect"""
        try:
            if hasattr(self, 'room') and self.room:
                await self.room.local_participant.publish_data(
                    data=json.dumps({
                        'type': 'conversation_end',
                        'message': 'The conversation has ended. Disconnecting...',
                        'timestamp': datetime.now().isoformat(),
                        'reason': 'user_requested'
                    }).encode(),
                    topic="lk.conversation.control"
                )
                logger.info(f"DISCONNECT_SIGNAL: Sent disconnection signal for session {self.session_id}")
        except Exception as e:
            logger.error(f"Error sending disconnection signal: {e}")

    async def generate_reply(self, instructions: str = None):
        """Override generate_reply to capture agent responses"""
        try:
            # Call the parent generate_reply method
            result = await super().generate_reply(instructions)
            
            # If we have a response, send it as agent transcript
            if result and hasattr(result, 'content'):
                await self.send_agent_transcript(result.content)
            
            return result
        except Exception as e:
            logger.error(f"Error in generate_reply: {e}")
            return None

    async def send_agent_transcript(self, message: str):
        """Send agent transcript as a separate message to distinguish from user speech"""
        try:
            # Send agent transcript through the transcription stream but with agent identity
            if hasattr(self, 'room') and self.room:
                # Send as data message for reliable delivery
                await self.room.local_participant.publish_data(
                    data=json.dumps({
                        'type': 'agent_transcript',
                        'message': message,
                        'speaker': 'Scott_AI_Agent',
                        'timestamp': datetime.now().isoformat()
                    }).encode(),
                    topic="lk.agent.transcript"
                )
                logger.info(f"AGENT_TRANSCRIPT: Sent agent message via data: '{message}'")
        except Exception as e:
            logger.error(f"Error sending agent transcript: {e}")

    def update_conversation_context(self, stage: str):
        """Update conversation stage"""
        self.conversation_stage = stage
        logger.info(f"CONVERSATION_STAGE: Session {self.session_id} moved to stage: {stage}")

def prewarm(proc):
    """Preload models for better performance"""
    proc.userdata["vad"] = silero.VAD.load()

async def entrypoint(ctx: JobContext):
    """Main entry point for the flow-based voice agent"""
    session_id = str(uuid.uuid4())[:8]
    room_name = ctx.room.name
    
    if room_name in active_sessions:
        logger.warning(f"Room {room_name} already has an active session. Skipping.")
        return
    
    # Create assistant first
    assistant = FlowBasedAssistant(session_id)
    active_sessions[room_name] = session_id
    logger.info(f"Starting flow-based agent session {session_id} in room: {room_name}")
    
    # Check backend health before starting
    logger.info("🔍 Checking backend health before starting worker...")
    backend_healthy = await check_backend_health()
    if not backend_healthy:
        logger.error("❌ Backend is not accessible. Worker will start but may not function properly.")
        logger.error("💡 Please ensure the backend is running and accessible.")
        # Signal backend down status (will be sent after room connection)
        assistant.backend_healthy = False
    else:
        logger.info("✅ Backend health check passed. Worker ready to start.")
        # Signal worker ready status (will be sent after room connection)
        assistant.backend_healthy = True
    
    agent_session = None
    
    try:
        await ctx.connect(auto_subscribe=AutoSubscribe.AUDIO_ONLY)
        logger.info(f"Connected to room {room_name}")
        
        # Store room reference in assistant for transcript handling
        assistant.room = ctx.room
        assistant.agent_session = None  # Will be set after session starts
        assistant.retry_manager = BackendRetryManager(assistant)
        
        # Send initial status signal now that room is connected
        if assistant.backend_healthy:
            await assistant.signal_worker_status("ready", "Worker is ready and backend is accessible")
        else:
            await assistant.signal_worker_status("backend_down", "Backend server is not accessible")
        
        try:
            participant = await asyncio.wait_for(ctx.wait_for_participant(), timeout=600.0)
            logger.info(f"Participant {participant.identity} joined session {session_id}")
            
            await asyncio.sleep(2)
        except asyncio.TimeoutError:
            logger.warning(f"No participant joined session {session_id} within timeout")
            return
        except Exception as e:
            logger.error(f"Error waiting for participant in session {session_id}: {e}")
            return

        # Create AgentSession with proper configuration for distinct transcription
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
        
        # Start the session with proper room options for distinct transcription
        await agent_session.start(
            room=ctx.room,
            agent=assistant,
            room_input_options=RoomInputOptions(
                noise_cancellation=noise_cancellation.BVC(),
                text_enabled=True,  # Enable text input
            ),
            room_output_options=RoomOutputOptions(
                transcription_enabled=True,  # Enable transcript output to frontend
                sync_transcription=False,  # Disable sync to prevent mixing user/agent transcripts
            ),
        )
        
        # Store agent session reference in assistant for farewell handling
        assistant.agent_session = agent_session
        
        logger.info(f"Flow-based agent session started for {session_id}")
        
        # Initial greeting
        greeting_message = "Hello! I'm Scott from Alive5. How can I help you today?"
        
        await agent_session.generate_reply(
            instructions=f"Say exactly: '{greeting_message}'"
        )
        
        logger.info(f"Initial greeting sent for session {session_id}")
        
        # Keep session alive while participants are connected
        try:
            while ctx.room.remote_participants:
                await asyncio.sleep(1)
                logger.debug(f"Session {session_id} active - Participants connected")
        except Exception as e:
            logger.warning(f"Session monitoring interrupted for {session_id}: {e}")
        
    except Exception as e:
        logger.error(f"Error in session {session_id}: {str(e)}", exc_info=True)
    finally:
        # Robust cleanup
        logger.info(f"Starting cleanup for session {session_id}")
        
        # Clean up session tracking
        if room_name in active_sessions:
            del active_sessions[room_name]
            logger.info(f"Removed session {session_id} from active sessions")
        
        # Clean up retry manager
        if assistant.retry_manager:
            try:
                await assistant.retry_manager.stop_retry_loop()
                logger.info(f"Retry manager stopped for session {session_id}")
            except Exception as e:
                logger.error(f"Error stopping retry manager for session {session_id}: {e}")
        
        # Clean up agent session
        if agent_session:
            try:
                await asyncio.wait_for(agent_session.aclose(), timeout=10.0)
                logger.info(f"Agent session {session_id} closed properly")
            except asyncio.TimeoutError:
                logger.warning(f"Agent session {session_id} close timed out")
            except Exception as e:
                logger.error(f"Error closing agent session {session_id}: {e}")
        
        # Clean up room connection
        try:
            if ctx.room and ctx.room.connection_state == "connected":
                await asyncio.wait_for(ctx.room.disconnect(), timeout=5.0)
                logger.info(f"Room disconnected for session {session_id}")
        except asyncio.TimeoutError:
            logger.warning(f"Room disconnect timed out for session {session_id}")
        except Exception as e:
            logger.error(f"Error disconnecting room for session {session_id}: {e}")
            
        logger.info(f"Flow-based session {session_id} cleanup completed")

if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1 and sys.argv[1] == "test":
        # Test the assistant
        assistant = FlowBasedAssistant("test-session")
        print(f"Flow-based assistant created for session: {assistant.session_id}")
        print("Worker now uses flow-based processing")
    else:
        cli.run_app(
            WorkerOptions(
                entrypoint_fnc=entrypoint,
                prewarm_fnc=prewarm,
            ),
        )
