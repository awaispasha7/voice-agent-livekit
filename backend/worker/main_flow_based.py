import logging
import os
import uuid
import asyncio
import httpx
from datetime import datetime
from dotenv import load_dotenv
from typing import Dict, Any, Optional, Union, Literal, AsyncIterable
import json
import re
import os, psutil, threading, time

def monitor_memory():
    process = psutil.Process(os.getpid())
    while True:
        mem = process.memory_info().rss / (1024 * 1024)  # Resident Set Size in MB
        cpu = process.cpu_percent(interval=None)
        # print(f"[WORKER-MEM] {mem:.1f} MB | CPU: {cpu:.1f}%")
        time.sleep(5)  # print every 5 seconds

threading.Thread(target=monitor_memory, daemon=True).start()


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
from livekit.rtc import Participant, DataPacketKind
from livekit.agents.voice.agent import ModelSettings
from livekit.plugins import (
    deepgram,
    cartesia,
    openai,
    silero,
    noise_cancellation
)
from livekit.agents import llm
from livekit.agents.types import NOT_GIVEN, NotGivenOr, DEFAULT_API_CONNECT_OPTIONS, APIConnectOptions
from livekit import rtc

def preprocess_text_for_tts(text: str) -> str:
    """Preprocess text to improve TTS pronunciation"""
    if not text:
        return text
    
    # Fix common acronym pronunciation issues
    text = text.replace("(SSO)", "S-S-O")  # Pronounce SSO as letters
    text = text.replace("SSO", "S-S-O")    # Handle SSO without parentheses
    text = text.replace("CRM", "C-R-M")    # Pronounce CRM as letters
    text = text.replace("API", "A-P-I")    # Pronounce API as letters
    text = text.replace("URL", "U-R-L")    # Pronounce URL as letters
    
    return text

# Load environment variables
# Try multiple possible paths for .env file
import os
from pathlib import Path

# Get the current file's directory
current_dir = Path(__file__).parent
# Try different possible .env locations
env_paths = [
    current_dir / "../../.env",  # Relative to worker directory
    current_dir / "../../../.env",  # Relative to project root
    Path("/home/ubuntu/alive5-voice-agent/.env"),  # Absolute production path
    Path(".env"),  # Current working directory
]

env_loaded = False
for env_path in env_paths:
    if env_path.exists():
        load_dotenv(dotenv_path=str(env_path))
        # print(f"✅ Loaded .env from: {env_path}")  # Commented out to reduce log clutter
        env_loaded = True
        break

if not env_loaded:
    # print("⚠️ No .env file found in any expected location")  # Commented out to reduce log clutter
    # print(f"   Searched paths: {[str(p) for p in env_paths]}")
    # Fallback to default behavior
    load_dotenv()

# Configure logging with clean format (no systemd prefixes)
logging.basicConfig(
    level=logging.INFO,
    format='%(message)s',  # Clean format without timestamps/prefixes
    force=True  # Override any existing configuration
)
logger = logging.getLogger("flow-based-voice-agent")
logger.setLevel(logging.INFO)

# Reduce noise from some verbose loggers
logging.getLogger("livekit.agents.utils.aio.duplex_unix").setLevel(logging.WARNING)
logging.getLogger("livekit.agents.cli.watcher").setLevel(logging.WARNING)
logging.getLogger("livekit.agents.ipc.channel").setLevel(logging.WARNING)

# Debug: Show current working directory and file locations (commented out to reduce log clutter)
# print(f"🔍 DEBUG: Current working directory: {os.getcwd()}")
# print(f"🔍 DEBUG: Worker file location: {__file__}")
# print(f"🔍 DEBUG: Environment variables loaded: {env_loaded}")

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
        self.has_sent_initial_greeting = False
        
    def set_room_name(self, room_name: str):
        """Set the room name for this LLM instance"""
        self.room_name = room_name
        
    def chat(
        self,
        *,
        chat_ctx: llm.ChatContext,
        tools: list[llm.FunctionTool] | None = None,
        conn_options: APIConnectOptions = DEFAULT_API_CONNECT_OPTIONS,
        parallel_tool_calls: NotGivenOr[bool] = NOT_GIVEN,
        tool_choice: NotGivenOr[llm.ToolChoice] = NOT_GIVEN,
        extra_kwargs: NotGivenOr[dict[str, Any]] = NOT_GIVEN,
    ) -> llm.LLMStream:
        """Main method to override - intercepts all LLM responses"""
        # Since we're using hooks for the main flow, just return empty response
        # to prevent the default LLM from generating responses
        logger.info(f"🎤 CUSTOM LLM: chat() called - returning empty response (using hooks instead)")
        return self._create_response_stream(chat_ctx, "")
    
    def _create_response_stream(self, chat_ctx: llm.ChatContext, response_text: str) -> llm.LLMStream:
        """Create a proper LLMStream with the response text"""
        # Create a simple async generator that yields ChatChunks
        async def _response_generator():
            # Always yield a chunk, but with empty content if needed
            chunk = llm.ChatChunk(
                id=str(uuid.uuid4()),
                delta=llm.ChoiceDelta(
                    content=response_text,
                    role="assistant"
                )
            )
            yield chunk
        
        # Create a proper LLMStream instance
        # Based on the documentation, LLMStream is an abstract base class
        # We need to create a concrete implementation
        class SimpleLLMStream(llm.LLMStream):
            def __init__(self, generator):
                self._generator = generator
                self._chat_ctx = chat_ctx
                
            async def _run(self):
                """Required abstract method implementation"""
                async for chunk in self._generator:
                    yield chunk
                
            async def __anext__(self):
                return await self._generator.__anext__()
                
            def __aiter__(self):
                return self
                
            @property
            def chat_ctx(self):
                return self._chat_ctx
                
            @property
            def fnc_ctx(self):
                return None
                
            @property
            def function_calls(self):
                return []
                
            def execute_functions(self):
                return []
                
            async def aclose(self):
                pass
        
        return SimpleLLMStream(_response_generator())
    
    
    async def _call_backend_async(self, user_message: str, conversation_history: list) -> dict:
        """Async backend call to process flow messages"""
        try:
            import httpx
            
            # Get botchain information from session
            botchain_name = None
            org_name = None
            try:
                session_endpoint = f"{self.backend_url}/api/sessions/{self.room_name}"
                async with httpx.AsyncClient(timeout=httpx.Timeout(5.0)) as client:
                    session_response = await client.get(session_endpoint)
                    if session_response.status_code == 200:
                        session_data = session_response.json()
                        user_data = session_data.get("user_data", {})
                        botchain_name = user_data.get("botchain_name")
                        org_name = user_data.get("org_name")
            except Exception as e:
                logger.warning(f"🔧 Could not get session info for botchain: {e}")
            
            logger.info(f"🔧 Making HTTP request to: {self.backend_url}/api/process_flow_message")
            logger.info(f"🔧 Request payload: room_name={self.room_name}, user_message='{user_message}', botchain_name={botchain_name}")
            
            # Prepare request payload
            payload = {
                "room_name": self.room_name,
                "user_message": user_message,
                "conversation_history": conversation_history
            }
            
            # Add botchain information if available
            if botchain_name:
                payload["botchain_name"] = botchain_name
            if org_name:
                payload["org_name"] = org_name
            
            async with httpx.AsyncClient() as client:
                response = await client.post(
                    f"{self.backend_url}/api/process_flow_message",
                    json=payload,
                    headers={"Authorization": f"Bearer {self.api_key}"},
                    timeout=BACKEND_TIMEOUT
                )
                
                logger.info(f"🔧 Backend response status: {response.status_code}")
                logger.info(f"🔧 Backend response text: {response.text}")
                
                if response.status_code == 200:
                    data = response.json()
                    logger.info(f"🔧 Backend response data: {data}")
                    
                    # Check for both "success" and "processed" status
                    if data.get("status") in ["success", "processed"] and "flow_result" in data:
                        flow_result = data["flow_result"]
                        logger.info(f"🔧 Flow result: {flow_result}")
                        
                        # Normalize fields
                        ftype = (flow_result or {}).get("type")
                        response_text = (flow_result or {}).get("response") or ""
                        next_step = (flow_result or {}).get("next_step")
                        
                        # If flow just started OR response is boilerplate, try to speak the next_step's text
                        if ftype == "flow_started" and (not response_text or response_text.strip().upper() == "N/A" or response_text.strip().lower().startswith("i understand you want to know")):
                            if isinstance(next_step, dict):
                                candidate = next_step.get("text")
                                if candidate and candidate.strip():
                                    logger.info(f"🔧 Using next_step text due to empty/N/A response: '{candidate}'")
                                    return {"text": candidate, "type": "question" if candidate.strip().endswith("?") else ftype}
                            # If next_step is missing or has no text, gently ask the user to clarify for this flow
                            if (not next_step) or (isinstance(next_step, dict) and not (next_step.get("text") or "").strip()):
                                fallback_prompt = "Could you please clarify so I can proceed?"
                                logger.info("🔧 Missing next_step text; using gentle clarification prompt")
                                return {"text": fallback_prompt, "type": "clarify"}
                        
                        # Handle all flow result types
                        if ftype in ["flow_response", "question", "message", "faq_response", "faq", "flow_started", "conversation_end", "transfer_initiated"]:
                            if response_text and response_text.strip():
                                logger.info(f"🔧 Flow response ({ftype}): '{response_text}'")
                                # Bubble up flow_name for intent updates
                                out: Dict[str, Any] = {"text": response_text, "type": ftype}
                                if ftype == "flow_started" and isinstance(flow_result, dict) and flow_result.get("flow_name"):
                                    out["flow_name"] = flow_result.get("flow_name")
                                return out
                        
                        if ftype == "error":
                            response_text = response_text or "I'm here to help!"
                            logger.info(f"🔧 Error response: '{response_text}'")
                            return {"text": response_text, "type": "error"}
                        
                        if ftype == "agent_handoff":
                            response_text = response_text or "I'm connecting you with a human agent. Please hold on."
                            logger.info(f"🔧 Agent handoff response: '{response_text}'")
                            return {"text": response_text, "type": "agent_handoff"}
                    
                    # Fallback to generic response
                    logger.warning("🔧 No valid flow result found, using fallback")
                    return {"text": "I'm here to help! How can I assist you today?", "type": "fallback"}
                else:
                    logger.error(f"❌ Backend error: {response.status_code} - {response.text}")
                    return {"text": "I'm here to help! How can I assist you today?", "type": "fallback"}
                    
        except Exception as e:
            logger.error(f"❌ Error calling backend: {e}", exc_info=True)
            return {"text": "I'm here to help! How can I assist you today?", "type": "fallback"}
    
    async def process_through_backend(self, user_message: str, conversation_history: list) -> str:
        """Process user message through backend flow system"""
        try:
            if not self.room_name:
                return "I'm experiencing technical difficulties. Please try again."
                
            # Check backend health
            backend_healthy = await check_backend_health()
            if not backend_healthy:
                return "I'm experiencing some technical difficulties. Let me connect you to a human agent who can help you right away."
            
            # Get botchain information from session
            botchain_name = None
            org_name = None
            try:
                session_endpoint = f"{self.backend_url}/api/sessions/{self.room_name}"
                async with httpx.AsyncClient(timeout=httpx.Timeout(5.0)) as client:
                    session_response = await client.get(session_endpoint)
                    if session_response.status_code == 200:
                        session_data = session_response.json()
                        user_data = session_data.get("user_data", {})
                        botchain_name = user_data.get("botchain_name")
                        org_name = user_data.get("org_name")
            except Exception as e:
                logger.warning(f"🔄 Could not get session info for botchain: {e}")
            
            # Send to backend flow processor
            logger.info(f"🔄 BACKEND REQUEST: Room={self.room_name}, Message='{user_message}', Botchain={botchain_name}")
            
            async with httpx.AsyncClient(timeout=BACKEND_TIMEOUT) as client:
                payload = {
                    "room_name": self.room_name,
                    "user_message": user_message,
                    "conversation_history": conversation_history[-10:]  # Last 10 messages
                }
                
                # Add botchain information if available
                if botchain_name:
                    payload["botchain_name"] = botchain_name
                if org_name:
                    payload["org_name"] = org_name
                
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



# Backend configuration
BACKEND_URL = os.getenv("BACKEND_URL", "http://localhost:8000")
BACKEND_TIMEOUT = 25  # Increased timeout to reduce read timeouts
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
        self.custom_llm = custom_llm
        self._greeted = False
        self._speech_lock = asyncio.Lock()
        self._ending = False
        self.selected_voice = "a167e0f3-df7e-4d52-a9c3-f949145efdab"  # Default voice (Customer Support Man)
        # Aggregate multiple short user turns before backend call
        self._aggregate_buffer: str = ""
        self._aggregate_task: Optional[asyncio.Task] = None
        self._aggregate_window_s: float = 1.2
        self._last_chat_ctx: Optional[llm.ChatContext] = None
        self._backend_call_lock = asyncio.Lock()
        
        # Use the custom LLM instead of default
        super().__init__(
            instructions="Flow-based voice assistant for Alive5 Support",
            llm=custom_llm
        )
    
    async def on_enter(self) -> None:
        """Called when the agent enters the room"""
        logger.info(f"🎤 AGENT ENTERED ROOM: {self.session_id}")
        
        # Check for greeting bot in template first, then fall back to hardcoded greeting
        try:
            if not self._greeted and self.session:
                # Try to get greeting from backend template
                greeting_response = await self._get_greeting_from_backend()
                
                if greeting_response:
                    # Use greeting from template
                    async with self._speech_lock:
                        await self.session.say(preprocess_text_for_tts(greeting_response))
                    self._greeted = True
                    try:
                        await self.send_agent_transcript(greeting_response)
                    except Exception:
                        pass
                    logger.info(f"👋 Template greeting sent for {self.session_id}: {greeting_response}")
                    
                    # Initialize greeting bot flow in backend
                    await self._initialize_greeting_flow_in_backend(greeting_response)
                else:
                    # No greeting found in template, but don't send hardcoded greeting
                    # The backend will handle the greeting flow initialization
                    logger.info(f"👋 No greeting found in template for {self.session_id}")
                    self._greeted = True
        except Exception as e:
            logger.error(f"❌ Failed to send initial greeting: {e}", exc_info=True)
    
    async def _get_greeting_from_backend(self) -> Optional[str]:
        """Get greeting from backend template if available"""
        try:
            # First, get session info to check for custom botchain
            backend_url = os.getenv("BACKEND_URL", "http://localhost:8000")
            session_endpoint = f"{backend_url}/api/sessions/{self.room_name}"
            
            botchain_name = None
            org_name = None
            
            # Try to get session info to check for custom botchain
            try:
                async with httpx.AsyncClient(timeout=httpx.Timeout(5.0)) as client:
                    session_response = await client.get(session_endpoint)
                    if session_response.status_code == 200:
                        session_data = session_response.json()
                        user_data = session_data.get("user_data", {})
                        botchain_name = user_data.get("botchain_name")
                        org_name = user_data.get("org_name")
                        selected_voice = user_data.get("selected_voice", "a167e0f3-df7e-4d52-a9c3-f949145efdab")
                        
                        if botchain_name:
                            logger.info(f"🎯 GREETING BOT: Found custom botchain in session: {botchain_name}/{org_name}")
                        else:
                            logger.info("🎯 GREETING BOT: No custom botchain found, using default template")
                        
                        # Store the selected voice
                        self.selected_voice = selected_voice
                        logger.info(f"🎤 VOICE: Using voice {selected_voice}")
            except Exception as e:
                logger.warning(f"🎯 GREETING BOT: Could not get session info: {e}")
            
            # If we have a custom botchain, load it first
            if botchain_name:
                try:
                    logger.info(f"🎯 GREETING BOT: Loading custom template for botchain: {botchain_name}")
                    # Call the template refresh endpoint with custom botchain
                    refresh_endpoint = f"{backend_url}/api/refresh_template"
                    async with httpx.AsyncClient(timeout=httpx.Timeout(10.0)) as client:
                        refresh_response = await client.post(
                            refresh_endpoint,
                            json={
                                "botchain_name": botchain_name,
                                "org_name": org_name or "alive5stage0"
                            }
                        )
                        if refresh_response.status_code == 200:
                            logger.info(f"🎯 GREETING BOT: Successfully loaded custom template for {botchain_name}")
                        else:
                            logger.warning(f"🎯 GREETING BOT: Failed to load custom template: {refresh_response.status_code}")
                except Exception as e:
                    logger.warning(f"🎯 GREETING BOT: Error loading custom template: {e}")
            
            # Now get the greeting from the (potentially updated) template
            greeting_endpoint = f"{backend_url}/api/get_greeting"
            
            async with httpx.AsyncClient(timeout=httpx.Timeout(5.0)) as client:
                response = await client.get(greeting_endpoint)
                if response.status_code == 200:
                    result = response.json()
                    if result.get("greeting_available"):
                        greeting_text = result.get("greeting_text", "")
                        logger.info(f"🎯 GREETING BOT: Found greeting in template: {greeting_text}")
                        return greeting_text
                    else:
                        logger.info("🎯 GREETING BOT: No greeting bot found in template")
                        return None
                else:
                    logger.warning(f"🎯 GREETING BOT: Backend call failed with status {response.status_code}")
                    return None
        except Exception as e:
            logger.warning(f"🎯 GREETING BOT: Failed to get greeting from backend: {e}")
            return None
    
    async def _initialize_greeting_flow_in_backend(self, greeting_text: str):
        """Initialize greeting bot flow in backend so it knows about the flow state"""
        try:
            # Call backend to initialize greeting flow
            backend_url = os.getenv("BACKEND_URL", "http://localhost:8000")
            init_endpoint = f"{backend_url}/api/initialize_greeting_flow"
            
            # Get room name from stored attribute
            room_name = getattr(self, 'room_name', 'unknown')
            
            payload = {
                "room_name": room_name,
                "greeting_text": greeting_text
            }
            
            async with httpx.AsyncClient(timeout=httpx.Timeout(5.0)) as client:
                response = await client.post(init_endpoint, json=payload)
                if response.status_code == 200:
                    result = response.json()
                    if result.get("success"):
                        logger.info(f"🎯 GREETING FLOW: Initialized greeting flow in backend for room {room_name}")
                    else:
                        logger.warning(f"🎯 GREETING FLOW: Failed to initialize greeting flow: {result.get('error')}")
                else:
                    logger.warning(f"🎯 GREETING FLOW: Backend call failed with status {response.status_code}")
        except Exception as e:
            logger.warning(f"🎯 GREETING FLOW: Failed to initialize greeting flow in backend: {e}")
    
    async def on_user_turn_completed(self, turn_ctx: llm.ChatContext, new_message: llm.ChatMessage) -> None:
        """Called when user finishes speaking; aggregate multiple turns before processing"""
        logger.info(f"🎤 USER TURN COMPLETED: '{new_message.text_content}'")
        self._last_chat_ctx = turn_ctx
        text = (new_message.text_content or "").strip()
        if not text:
            return
        # Append to buffer; if previous ends without punctuation, add a space
        joiner = " " if (self._aggregate_buffer and not self._aggregate_buffer.endswith((" ", ",", ".", "?", "!"))) else ""
        self._aggregate_buffer = f"{self._aggregate_buffer}{joiner}{text}".strip()
        # Reset timer
        if self._aggregate_task and not self._aggregate_task.done():
            self._aggregate_task.cancel()
            try:
                await self._aggregate_task
            except Exception:
                pass
        self._aggregate_task = asyncio.create_task(self._flush_aggregate_after_delay())

    async def _flush_aggregate_after_delay(self) -> None:
        try:
            await asyncio.sleep(self._aggregate_window_s)
            text = self._aggregate_buffer.strip()
            self._aggregate_buffer = ""
            if not text:
                return
            await self._process_aggregated_text(text)
        except asyncio.CancelledError:
            return
        except Exception as e:
            logger.error(f"Aggregate flush error: {e}")

    async def on_data_received(self, data: bytes, participant: Participant, kind: DataPacketKind, topic: str | None) -> None:
        """Handle data messages from the room"""
        try:
            if not data:
                return
                
            message = json.loads(data.decode('utf-8'))
            message_type = message.get("type")
            
            if message_type == "voice_change":
                voice_id = message.get("voice_id")
                if voice_id:
                    logger.info(f"🎤 VOICE_CHANGE: Received voice change signal for voice {voice_id}")
                    await self._update_voice(voice_id)
                    
        except Exception as e:
            logger.error(f"Error handling data message: {e}")

    async def _update_voice(self, voice_id: str) -> None:
        """Update the TTS voice for the agent session"""
        try:
            if not hasattr(self, 'agent_session') or not self.agent_session:
                logger.warning("🎤 VOICE_CHANGE: No agent session available to update voice")
                return
                
            # Update the selected voice
            self.selected_voice = voice_id
            logger.info(f"🎤 VOICE_CHANGE: Updated selected_voice to {voice_id}")
            
            # Create new TTS with the new voice
            new_tts = cartesia.TTS(
                model="sonic-2024-10-19",
                voice=voice_id,
                api_key=os.getenv("CARTESIA_API_KEY")
            )
            
            # Update the agent session's TTS
            self.agent_session.tts = new_tts
            logger.info(f"🎤 VOICE_CHANGE: Successfully updated TTS voice to {voice_id}")
            
        except Exception as e:
            logger.error(f"🎤 VOICE_CHANGE: Failed to update voice: {e}")

    async def _process_aggregated_text(self, user_text: str) -> None:
        """Send aggregated text to backend and speak response"""
        try:
            # Smalltalk blending
            lower_text = user_text.lower()
            polite_reply: Optional[str] = None
            greetings = ["hi", "hello", "hey", "good morning", "good evening"]
            byes = ["bye", "goodbye", "see you", "talk to you later", "that\'s all", "thanks, bye"]
            affirmations = ["okay", "ok", "sounds good", "that sounds great", "great", "thanks", "thank you"]
            
            # Don't add polite replies for simple greetings - let the backend handle them naturally
            if any(lower_text.startswith(g) for g in greetings):
                polite_reply = None  # Let the backend handle greetings naturally
            elif any(b in lower_text for b in byes):
                polite_reply = "Thanks for calling Alive5. Have a great day!"
            elif any(a in lower_text for a in affirmations):
                polite_reply = "Okay."

            # Initialize clarification trackers
            if not hasattr(self, "_last_question_text"):
                self._last_question_text = None
                self._clarify_count = 0

            # Build conversation history from last chat ctx
            conversation_history = []
            if self._last_chat_ctx:
                for msg in self._last_chat_ctx.items:
                    if hasattr(msg, 'text_content') and msg.text_content:
                        conversation_history.append({
                            "role": msg.role,
                            "content": msg.text_content,
                            "timestamp": datetime.now().isoformat()
                        })
            conversation_history.append({
                "role": "user",
                "content": user_text,
                "timestamp": datetime.now().isoformat()
            })

            logger.info(f"🎤 Aggregated user message: '{user_text}'")
            async with self._backend_call_lock:
                backend_out = await self.custom_llm._call_backend_async(user_text, conversation_history)
            response_text = backend_out.get("text", "")
            rtype = backend_out.get("type", "")
            logger.info(f"🎤 Backend returned: type={rtype} text='{response_text}'")

            # Intent update broadcast
            try:
                if rtype == "flow_started" and backend_out.get("flow_name") and self.room:
                    await self.room.local_participant.publish_data(
                        json.dumps({
                            'type': 'intent_update',
                            'intent': backend_out.get("flow_name"),
                            'source': 'Flow System',
                            'timestamp': datetime.now().isoformat()
                        }).encode(),
                        topic="lk.intent.update"
                    )
            except Exception as _e:
                logger.warning(f"Failed to publish intent update: {_e}")

            # Clarify when not understood - but don't intercept simple greetings
            if rtype in ("error", "fallback") and self._last_question_text:
                # Check if this is a simple greeting that should be allowed through
                user_text_lower = user_text.lower().strip()
                simple_greetings = ["hi", "hello", "hi there", "hey", "good morning", "good afternoon", "good evening"]
                
                if not any(greeting in user_text_lower for greeting in simple_greetings):
                    if self._clarify_count < 2:
                        response_text = f"Sorry, I didn't catch that. {self._last_question_text}"
                        self._clarify_count += 1
                else:
                    # For simple greetings, let the backend response through without modification
                    logger.info(f"🎤 Allowing simple greeting through: '{user_text}'")
            else:
                self._clarify_count = 0

            # Sanitize odd fallbacks and avoid duplicated greetings
            rlow = response_text.lower().strip()
            if "i'm not scott" in rlow:
                response_text = response_text.replace("I'm not Scott", "I'm Scott").replace("i'm not scott", "I'm Scott")
                rlow = response_text.lower().strip()
            if rtype != "conversation_end" and polite_reply and not response_text.strip().endswith("?"):
                # Don't add polite replies for simple greetings when backend returns error/fallback
                if not (rlow.startswith("hello") or "how can i help" in rlow):
                    # Check if this is a simple greeting that should be handled naturally
                    user_text_lower = user_text.lower().strip()
                    simple_greetings = ["hi", "hello", "hi there", "hey", "good morning", "good afternoon", "good evening"]
                    
                    if not any(greeting in user_text_lower for greeting in simple_greetings):
                        response_text = f"{polite_reply} {response_text}".strip()

            # Track last question
            if rtype == "question" or response_text.strip().endswith("?"):
                self._last_question_text = response_text
            elif rtype in ("message", "faq", "faq_response", "flow_started", "conversation_end"):
                self._last_question_text = response_text if response_text.strip().endswith("?") else None

            # Deduplicate repeated sentence fragments in the response
            def _dedupe_sentences(text: str) -> str:
                parts = [p.strip() for p in re.split(r"(?<=[.!?])\s+", text) if p.strip()]
                seen = set()
                out = []
                for p in parts:
                    key = p.lower()
                    if key not in seen:
                        seen.add(key)
                        out.append(p)
                return " ".join(out)
            response_text = _dedupe_sentences(response_text)

            async with self._speech_lock:
                await self.session.say(preprocess_text_for_tts(response_text))

            try:
                await self.send_agent_transcript(response_text)
            except Exception as _e:
                logger.warning(f"Failed to publish agent transcript: {_e}")

            # Graceful end: if backend indicates conversation_end, signal frontend and disconnect
            if rtype == "conversation_end" and not self._ending:
                self._ending = True
                try:
                    await self.send_disconnection_signal()
                except Exception as _e:
                    logger.warning(f"Failed to send conversation end signal: {_e}")
                try:
                    # small delay so TTS finishes
                    await asyncio.sleep(1.0)
                    if self.room and self.room.connection_state == "connected":
                        await self.room.disconnect()
                except Exception as _e:
                    logger.warning(f"Failed to disconnect room after conversation end: {_e}")
        except Exception as e:
            logger.error(f"❌ Error processing aggregated user message: {e}", exc_info=True)

    async def llm_node(
        self,
        chat_ctx: llm.ChatContext,
        tools: list[llm.FunctionTool],
        model_settings: ModelSettings
    ) -> AsyncIterable[llm.ChatChunk]:
        """Override LLM node to use our custom flow processing"""
        logger.info(f"🎤 LLM NODE: Processing chat context with {len(chat_ctx.items)} items")
        
        # Check if the last message is from the assistant (our flow response)
        if chat_ctx.items and chat_ctx.items[-1].role == "assistant":
            # The response was already added by on_user_turn_completed
            # Just yield it as a ChatChunk
            last_message = chat_ctx.items[-1]
            chunk = llm.ChatChunk(
                id=str(uuid.uuid4()),
                delta=llm.ChoiceDelta(
                    content=last_message.text_content or "",
                    role="assistant"
                )
            )
            yield chunk
        else:
            # Fallback to default LLM behavior if no assistant message
            logger.info(f"🎤 LLM NODE: No assistant message found, using default behavior")
            # Call superclass implementation (Agent.llm_node)
            async for chunk in super().llm_node(chat_ctx, tools, model_settings):
                yield chunk

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
    logger.info(f"🔧 Creating custom LLM with backend URL: {BACKEND_URL}")
    custom_llm = FlowBasedLLM(BACKEND_URL, "dummy_key")
    custom_llm.set_room_name(room_name)
    logger.info(f"🔧 Custom LLM created with room_name: {room_name}")
    
    # Create assistant with custom LLM
    assistant = FlowBasedAssistant(session_id, custom_llm)
    assistant.room_name = room_name  # Store room name for greeting flow initialization
    logger.info(f"🔧 FlowBasedAssistant created for session: {session_id}")
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
                model="sonic-2024-10-19",
                voice=assistant.selected_voice, 
                api_key=os.getenv("CARTESIA_API_KEY")
            ),
            vad=ctx.proc.userdata["vad"],
            turn_detection=None,  # Disable turn detection to avoid compatibility issues
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
        # on_enter handles initial greeting via TTS
        
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