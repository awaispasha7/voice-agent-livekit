"""
Alive5 Simple Voice Agent - Single LLM with Function Calling
"""

import asyncio
import logging
import os
import random
import sys
from pathlib import Path
from typing import Dict, Any, Optional

# Set environment variables to control LiveKit logging BEFORE any imports
os.environ["LIVEKIT_LOG_LEVEL"] = "WARN"
os.environ["RUST_LOG"] = "warn"

# Simple logging configuration - just remove timestamps
logging.basicConfig(level=logging.INFO, format="%(message)s", force=True)

from dotenv import load_dotenv
from livekit.agents import (
    JobContext, WorkerOptions, cli, Agent, AgentSession,
    function_tool, RunContext, RoomInputOptions, RoomOutputOptions, AutoSubscribe
)
from livekit.plugins import openai, deepgram, cartesia, silero, noise_cancellation

from system_prompt import get_system_prompt
from functions import handle_load_bot_flows, handle_faq_bot_request, handle_bedrock_knowledge_base_request

# Load environment variables
load_dotenv(Path(__file__).parent / "../../.env")

# Create our logger
logger = logging.getLogger("simple-agent")

# Reduce LiveKit agent logging verbosity
logging.getLogger("livekit.agents").setLevel(logging.WARNING)
logging.getLogger("livekit").setLevel(logging.WARNING)
logging.getLogger("livekit.plugins").setLevel(logging.WARNING)
logging.getLogger("livekit.plugins.cartesia").setLevel(logging.WARNING)
logging.getLogger("livekit.plugins.deepgram").setLevel(logging.WARNING)
logging.getLogger("livekit.plugins.openai").setLevel(logging.WARNING)

# Reduce OpenAI and HTTP client logging verbosity
logging.getLogger("openai").setLevel(logging.WARNING)
logging.getLogger("openai._base_client").setLevel(logging.WARNING)
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("httpcore").setLevel(logging.WARNING)
logging.getLogger("httpcore.connection").setLevel(logging.WARNING)
logging.getLogger("httpcore.http11").setLevel(logging.WARNING)

# Reduce asyncio logging
logging.getLogger("asyncio").setLevel(logging.WARNING)

# Suppress all LiveKit internal logging with timestamps
import warnings
warnings.filterwarnings("ignore", category=UserWarning, module="livekit")

# Force clean logging format after LiveKit imports
def force_clean_logging():
    """Force all loggers to use clean format without timestamps"""
    clean_formatter = logging.Formatter("%(levelname)s %(name)s - %(message)s")
    
    # Completely replace root logger handlers
    root_logger = logging.getLogger()
    root_logger.handlers.clear()
    root_logger.addHandler(logging.StreamHandler(sys.stdout))
    for handler in root_logger.handlers:
        handler.setFormatter(clean_formatter)
    
    # Completely replace our specific loggers
    for logger_name in ["simple-agent", "functions"]:
        logger = logging.getLogger(logger_name)
        logger.handlers.clear()
        logger.addHandler(logging.StreamHandler(sys.stdout))
        for handler in logger.handlers:
            handler.setFormatter(clean_formatter)
        # Prevent propagation to root logger to avoid duplication
        logger.propagate = False

# Apply clean formatting
force_clean_logging()

# -----------------------------------------------------------------------------
# Simple Voice Agent
# -----------------------------------------------------------------------------
class SimpleVoiceAgent(Agent):
    """Single-LLM voice agent with function calling"""
    
    def __init__(self, room_name: str, botchain_name: str = "voice-1", org_name: str = "alive5stage0", special_instructions: str = ""):
        self.room_name = room_name
        self.botchain_name = botchain_name
        self.org_name = org_name
        self.room = None
        self.selected_voice = "e90c6678-f0d3-4767-9883-5d0ecf5894a8"  # Default voice
        self._turn_detection = None  # Required by Agent class
        
        # Flow management
        self.bot_template = None
        self.flow_states = {}  # Track current step of each flow
        
        # CRM data collection
        self.collected_data = {
            "full_name": None,
            "email": None,
            "phone": None,
            "notes_entry": []
        }
        
        # Get model from env or use default
        model_name = os.getenv("OPENAI_MODEL", "gpt-4o")

        import httpx
        llm_timeout = httpx.Timeout(connect=30.0, read=60.0, write=30.0, pool=30.0)
        
        try:
            llm_instance = openai.LLM(model=model_name, temperature=0.7, timeout=llm_timeout)
            self._inference_model_id = None  # Not using LiveKit Inference
        except Exception as e:
            logger.error(f"❌ Failed to initialize OpenAI plugin LLM with {model_name}: {e}")
            logger.warning(f"🔄 Falling back to gpt-4o-mini")
            try:
                llm_instance = openai.LLM(model="gpt-4o-mini", temperature=0.7, timeout=llm_timeout)
                logger.info(f"✅ Fallback LLM initialized: gpt-4o-mini")
            except Exception as e2:
                logger.error(f"❌ Fallback also failed: {e2}")
                raise Exception(f"Could not initialize any LLM. Original error: {e}, Fallback error: {e2}")
            self._inference_model_id = None
        
        # Initialize the base Agent class with special instructions
        system_prompt = get_system_prompt(botchain_name, org_name, special_instructions)
        super().__init__(instructions=system_prompt, llm=llm_instance)
        
    
    @function_tool()
    async def load_bot_flows(self, context: RunContext, botchain_name: str, org_name: str) -> Dict[str, Any]:
        """Load Alive5 bot flow definitions dynamically. MUST be called on startup before first user interaction.
        
        Args:
            botchain_name: The botchain name (e.g., 'voice-1')
            org_name: The organization name (default: 'alive5stage0')
        """
        # logger.info(f"🔧 Loading bot flows for {botchain_name}")
        return await handle_load_bot_flows(botchain_name, org_name)
    
    
    @function_tool()
    async def transfer_call_to_human(self, context: RunContext, transfer_number: Optional[str] = None) -> Dict[str, Any]:
        """Transfer the current phone call to a human agent or phone number.
        
        Args:
            transfer_number: Phone number to transfer to (e.g., "+18555518858"). 
                           Optional - if not provided, uses default call center number from environment.
                           If no transfer number is configured, returns helpful message.
        
        Returns:
            Success status and message
        """
        try:
            # Get transfer number or use default
            if not transfer_number:
                transfer_number = os.getenv("TELNYX_CALL_CENTER_NUMBER")
                
                # If no transfer number configured, inform user
                if not transfer_number:
                    logger.warning("⚠️ No TELNYX_CALL_CENTER_NUMBER configured - transfer not available")
                    return {
                        "success": False,
                        "message": "I'm sorry, call transfers are not currently configured. Is there anything else I can help you with?"
                    }
            
            # Get call control ID from session
            import httpx
            from urllib.parse import quote, unquote
            
            backend_url = os.getenv("BACKEND_URL", "http://18.210.238.67")
            
            # Ensure room name doesn't have double prefix (clean it up)
            room_name_clean = self.room_name
            if room_name_clean.startswith("telnyx_call__telnyx_call_"):
                # Remove double prefix
                room_name_clean = room_name_clean.replace("telnyx_call__telnyx_call_", "telnyx_call_", 1)
                logger.warning(f"⚠️ Fixed double-prefixed room name: {self.room_name} -> {room_name_clean}")
            
            # URL-encode the room name to handle special characters (colons, etc.)
            encoded_room_name = quote(room_name_clean, safe='')
            
            logger.info(f"📞 Attempting transfer - Room: {room_name_clean}, Encoded: {encoded_room_name}")
            
            async with httpx.AsyncClient(timeout=5.0) as client:
                response = await client.get(f"{backend_url}/api/sessions/{encoded_room_name}")
                if response.status_code == 200:
                    session_data = response.json()
                    call_control_id = session_data.get("call_control_id")
                    
                    if call_control_id:
                        # Transfer call via backend
                        transfer_response = await client.post(
                            f"{backend_url}/api/telnyx/transfer",
                            json={
                                "room_name": room_name_clean,  # Use cleaned room name
                                "call_control_id": call_control_id,
                                "transfer_to": transfer_number
                            }
                        )
                        
                        if transfer_response.status_code == 200:
                            logger.info(f"✅ Call transferred to {transfer_number}")
                            return {
                                "success": True,
                                "message": f"Connecting you to a representative now..."
                            }
                        else:
                            logger.error(f"❌ Transfer failed: {transfer_response.status_code} - {transfer_response.text}")
                            return {
                                "success": False,
                                "message": "I'm having trouble transferring you. Please hold..."
                            }
                    else:
                        logger.warning("⚠️ No call_control_id found - not a phone call")
                        return {
                            "success": False,
                            "message": "Transfer is only available for phone calls."
                        }
                else:
                    logger.error(f"❌ Could not get session data: {response.status_code} - {response.text}")
                    logger.error(f"   Original room name: {self.room_name}")
                    logger.error(f"   Cleaned room name: {room_name_clean}")
                    logger.error(f"   Encoded room name: {encoded_room_name}")
                    return {
                        "success": False,
                        "message": "Unable to process transfer request."
                    }
        except Exception as e:
            logger.error(f"❌ Error transferring call: {e}")
            return {
                "success": False,
                "message": "I'm having trouble transferring you right now."
            }
    
    @function_tool()
    async def submit_crm_data(self, context: RunContext) -> Dict[str, Any]:
        """Submit collected customer data to CRM at the end of conversation.
        Call this when you have collected the customer's information (name, email, notes) and the conversation is ending.
        """
        try:
            # Prepare data for submission
            crm_data = {
                "room_name": self.room_name,
                "botchain_name": self.botchain_name,
                "org_name": self.org_name,
                "full_name": self.collected_data.get("full_name"),
                "email": self.collected_data.get("email"),
                "phone": self.collected_data.get("phone"),
                "notes": " | ".join(self.collected_data.get("notes_entry", [])) if self.collected_data.get("notes_entry") else None
            }
            
            # Submit to backend API
            import httpx
            backend_url = os.getenv("BACKEND_URL", "http://18.210.238.67")
            async with httpx.AsyncClient(timeout=10.0) as client:
                response = await client.post(
                    f"{backend_url}/api/submit_crm",
                    json=crm_data
                )
                
                if response.status_code == 200:
                    logger.info(f"✅ CRM data submitted successfully")
                    return {
                        "success": True,
                        "message": "Customer information has been saved and forwarded to the team."
                    }
                else:
                    logger.error(f"❌ CRM submission failed: {response.status_code}")
                    return {
                        "success": False,
                        "message": "There was an issue saving the information, but I've noted your details."
                    }
        except Exception as e:
            logger.error(f"❌ Error submitting CRM data: {e}")
            return {
                "success": False,
                "message": "Information noted, though there was a technical issue with submission."
            }
    
    @function_tool()
    async def faq_bot_request(self, context: RunContext, faq_question: str, bot_id: str = None, isVoice: bool = None) -> Dict[str, Any]:
        """Call the Alive5 FAQ bot API to get answers about Alive5 services, pricing, features, or company information.
        
        Args:
            faq_question: The user's question about Alive5
            bot_id: The FAQ bot ID (if None, uses session data)
            isVoice: Whether this is a voice interaction (if None, uses agent's faq_isVoice setting)
        """
        # logger.info(f"🔧 FAQ bot request: {faq_question}")
        
        # Use dynamic FAQ bot ID from session data if not provided
        if bot_id is None:
            bot_id = await self._get_faq_bot_id()
        
        # Use agent's faq_isVoice if isVoice not specified
        if isVoice is None:
            isVoice = getattr(self, 'faq_isVoice', True)
        
        # Provide immediate feedback to user
        if hasattr(self, "agent_session") and self.agent_session:
            await self.agent_session.say("Let me check that for you...")
        
        # Call FAQ with waiting callback
        async def waiting_callback(message):
            if hasattr(self, "agent_session") and self.agent_session:
                await self.agent_session.say(message)
        
        # Try Bedrock Knowledge Base first (faster), fallback to Alive5 API if needed
        return await handle_bedrock_knowledge_base_request(faq_question, max_results=5, waiting_callback=waiting_callback)
    
    async def _get_current_voice(self):
        """Get current voice from session data (like working implementation)"""
        try:
            if not self.room_name:
                return "f114a467-c40a-4db8-964d-aaba89cd08fa"  # Miles - Yogi (same as working system)
            
            import httpx
            backend_url = os.getenv("BACKEND_URL", "http://18.210.238.67")
            async with httpx.AsyncClient(timeout=5.0) as client:
                response = await client.get(f"{backend_url}/api/sessions/{self.room_name}")
                if response.status_code == 200:
                    data = response.json()
                    voice_id = data.get("user_data", {}).get("selected_voice")
                    voice_name = data.get("user_data", {}).get("selected_voice_name", "Unknown")
                    if voice_id:
                        logger.info(f"🎤 Using voice: {voice_name} ({voice_id})")
                        return voice_id
        except Exception as e:
            logger.error(f"Failed to get voice: {e}")
        return "f114a467-c40a-4db8-964d-aaba89cd08fa"  # Miles - Yogi (same as working system)
    
    async def _get_faq_bot_id(self):
        """Get FAQ bot ID from session data"""
        try:
            if not self.room_name:
                return "faq_b9952a56-fc7b-41c9-b0a0-5c662ddb039e"  # Default FAQ bot
            
            import httpx
            backend_url = os.getenv("BACKEND_URL", "http://18.210.238.67")
            async with httpx.AsyncClient(timeout=5.0) as client:
                response = await client.get(f"{backend_url}/api/sessions/{self.room_name}")
                if response.status_code == 200:
                    data = response.json()
                    faq_bot_id = data.get("user_data", {}).get("faq_bot_id")
                    if faq_bot_id:
                        logger.info(f"🤖 Using FAQ bot: {faq_bot_id}")
                        return faq_bot_id
        except Exception as e:
            logger.error(f"Failed to get FAQ bot ID: {e}")
        return "faq_b9952a56-fc7b-41c9-b0a0-5c662ddb039e"  # Default FAQ bot
    
    async def on_room_enter(self, room):
        """Called when agent enters the room - start with greeting"""
        # Start the conversation with greeting
        await self._start_conversation()
    
    async def on_session_end(self):
        """Called when session is ending - cleanup livechat (skip for phone calls)"""
        # Skip livechat cleanup for phone calls
        if self.room_name.startswith("telnyx_call_"):
            logger.info(f"📞 Phone call ending - skipping livechat cleanup")
            return
        
        try:
            import httpx
            backend_url = os.getenv("BACKEND_URL", "http://18.210.238.67")
            async with httpx.AsyncClient(timeout=5.0) as client:
                response = await client.post(
                    f"{backend_url}/api/end_livechat",
                    params={"room_name": self.room_name}
                )
                if response.status_code == 200:
                    logger.info(f"✅ Livechat session ended successfully")
                else:
                    logger.warning(f"⚠️ Livechat end failed: {response.status_code}")
        except Exception as e:
            logger.warning(f"⚠️ Could not end livechat: {e}")
    
    async def _start_conversation(self):
        """Start the conversation - let LLM handle flow loading and greeting"""
        try:
            # Use generate_reply to make the agent speak first
            if hasattr(self, "agent_session") and self.agent_session:
                await self.agent_session.generate_reply()
            else:
                logger.warning("⚠️ Agent session not available")
            
        except Exception as e:
            logger.error(f"❌ Error starting conversation: {e}", exc_info=True)
            # Informative fallback message
            if hasattr(self, "agent_session") and self.agent_session:
                await self.agent_session.say("Failed to load the bot flows. But you can still speak with me naturally.")
    
    async def _publish_to_frontend(self, data_type: str, message: str = None, **kwargs):
        """Publish data to frontend for display"""
        try:
            if not hasattr(self, 'room') or not self.room:
                return
            
            import json
            from datetime import datetime
            
            data = {
                "type": data_type,
                "timestamp": datetime.now().isoformat(),
                **kwargs
            }
            
            if message:
                data["message"] = message
            
            # Determine topic based on data type
            topic = "lk.conversation.control"
            if data_type == "user_transcript":
                topic = "lk.user.transcript"
                data["speaker"] = "User"
            
            await self.room.local_participant.publish_data(
                json.dumps(data).encode('utf-8'),
                topic=topic
            )
            
        except Exception as e:
            logger.error(f"Failed to publish to frontend: {e}")
    
    async def on_user_turn_completed(self, turn_ctx, new_message):
        """Handle user input and publish to frontend"""
        # Get user's transcribed text
        user_text = new_message.text_content or ""
        if user_text.strip():
            # Publish user transcript to frontend
            await self._publish_to_frontend("user_transcript", user_text, speaker="User")
            logger.info(f"USER: {user_text}")
        
        # Call parent method to handle the input
        await super().on_user_turn_completed(turn_ctx, new_message)
    

# -----------------------------------------------------------------------------
# Entry Point
# -----------------------------------------------------------------------------
def prewarm(proc):
    """Preload VAD model - using same approach as working implementation"""
    proc.userdata["vad"] = silero.VAD.load()

async def entrypoint(ctx: JobContext):
    """Main entry point for the simple agent"""
    logger.info("=" * 80)
    logger.info(f"🚀 NEW VOICE SESSION STARTING - Room: {ctx.room.name}")
    logger.info("=" * 80)
    
    # Fetch session data from backend to get dynamic configuration
    backend_url = os.getenv("BACKEND_URL", "http://18.210.238.67")
    botchain_name = "voice-1"
    org_name = "alive5stage0"
    faq_isVoice = True  # Default to concise responses
    special_instructions = ""  # Default empty special instructions
    
    # Detect if this is a phone call (Telnyx) or web session
    # Clean up room name: handle URL-encoding and double prefix from dispatch rule
    from urllib.parse import unquote, quote
    room_name_clean = ctx.room.name
    
    # First, URL-decode if needed (LiveKit dispatch rule may not decode it)
    if '%' in room_name_clean:
        room_name_decoded = unquote(room_name_clean)
        logger.warning(f"⚠️ Room name is URL-encoded: {room_name_clean} -> {room_name_decoded}")
        
        # Check if a room with the decoded name already exists (prevent duplicate sessions)
        # This happens when LiveKit dispatch rule doesn't decode the room name from SIP URI
        try:
            import httpx
            backend_url = os.getenv("BACKEND_URL", "http://18.210.238.67")
            encoded_decoded_name = quote(room_name_decoded, safe='')  # quote already imported above
            async with httpx.AsyncClient(timeout=2.0) as client:
                response = await client.get(f"{backend_url}/api/sessions/{encoded_decoded_name}")
                if response.status_code == 200:
                    # Decoded room name exists - this is a duplicate session from URL-encoded room
                    logger.warning(f"⚠️ Duplicate session detected! Room {room_name_decoded} already exists.")
                    logger.warning(f"   This session ({room_name_clean}) is a duplicate - exiting to prevent conflicts.")
                    return  # Exit early to prevent duplicate agent sessions
        except Exception as e:
            logger.debug(f"Could not check for duplicate: {e}")
        
        room_name_clean = room_name_decoded
    
    # Remove double prefix if present
    if room_name_clean.startswith("telnyx_call__telnyx_call_"):
        # Remove double prefix
        room_name_clean = room_name_clean.replace("telnyx_call__telnyx_call_", "telnyx_call_", 1)
        logger.warning(f"⚠️ Fixed double-prefixed room name: {ctx.room.name} -> {room_name_clean}")
    
    is_phone_call = room_name_clean.startswith("telnyx_call_")
    user_data = {}
    
    try:
        import httpx
        from urllib.parse import quote
        # URL-encode room name for API call
        encoded_room_name = quote(room_name_clean, safe='')
        async with httpx.AsyncClient() as client:
            response = await client.get(f"{backend_url}/api/sessions/{encoded_room_name}")
            if response.status_code == 200:
                session_data = response.json()
                user_data = session_data.get("user_data", {})
                # Check source field to confirm it's a phone call
                if user_data.get("source") == "telnyx_phone":
                    is_phone_call = True
            else:
                logger.warning(f"⚠️ Could not fetch session data (status {response.status_code}), using defaults")
    except Exception as e:
        logger.warning(f"⚠️ Could not fetch session data: {e}, using defaults")
    
    # Get configuration from session data or use defaults
    botchain_name = user_data.get("botchain_name", "voice-1")
    org_name = user_data.get("org_name", "alive5stage0")
    faq_isVoice = user_data.get("faq_isVoice", True)  # Default to concise responses
    special_instructions = user_data.get("special_instructions", "")  # Load special instructions
    
    # Connect to the room first - using same approach as working implementation
    await ctx.connect(auto_subscribe=AutoSubscribe.SUBSCRIBE_ALL)
    
    # Initialize livechat session with Socket.io (skip for phone calls)
    if not is_phone_call:
        try:
            import httpx
            async with httpx.AsyncClient(timeout=10.0) as client:
                response = await client.post(
                    f"{backend_url}/api/init_livechat",
                    params={
                        "room_name": room_name_clean,  # Use cleaned room name
                        "org_name": org_name,
                        "botchain_name": botchain_name
                    }
                )
                if response.status_code == 200:
                    init_result = response.json()
                    logger.info(f"✅ Livechat initialized - Thread: {init_result.get('thread_id')}, CRM: {init_result.get('crm_id')}")
                else:
                    logger.warning(f"⚠️ Livechat init failed: {response.status_code}")
        except Exception as e:
            logger.warning(f"⚠️ Could not initialize livechat: {e}")
    else:
        logger.info(f"📞 Phone call detected - skipping livechat initialization")
    
    # Create and start the agent (use cleaned room name)
    agent = SimpleVoiceAgent(room_name_clean, botchain_name, org_name, special_instructions)
    agent.faq_isVoice = faq_isVoice
    
    # Get VAD - with environment variable control for testing
    vad = None
    use_vad = os.getenv("USE_VAD", "true").lower() == "true"
    
    if use_vad:
        if "vad" in ctx.proc.userdata:
            vad = ctx.proc.userdata["vad"]
            logger.info("✅ VAD loaded from prewarm")
        else:
            try:
                vad = silero.VAD.load()
                logger.info("✅ VAD loaded successfully")
            except Exception as e:
                logger.warning(f"⚠️ Failed to load VAD: {e}, continuing without VAD")
    else:
        logger.info("🚫 VAD disabled via USE_VAD environment variable")
    
    # Set room on agent for frontend communication
    agent.room = ctx.room
    
    # Get current voice from session data (like working implementation)
    current_voice = await agent._get_current_voice()
    agent.selected_voice = current_voice
    logger.info(f"🎤 Initializing TTS with voice: {current_voice}")
    
    # Create TTS with fallback handling
    try:
        tts = cartesia.TTS(model="sonic-2", voice=current_voice, api_key=os.getenv("CARTESIA_API_KEY"))
        logger.info("✅ Cartesia TTS initialized successfully")
    except Exception as e:
        logger.error(f"❌ Cartesia TTS failed: {e}")
        logger.info("🔄 Falling back to basic Cartesia TTS...")
        try:
            tts = cartesia.TTS(model="sonic-2")
            logger.info("✅ Basic Cartesia TTS initialized")
        except Exception as e2:
            logger.error(f"❌ Basic Cartesia TTS also failed: {e2}")
            raise Exception("TTS initialization failed completely")
    
    # Get model name for logging
    model_name_for_log = None
    if hasattr(agent.llm, '_opts') and hasattr(agent.llm._opts, 'model'):
        model_name_for_log = agent.llm._opts.model
    elif hasattr(agent.llm, 'model'):
        model_name_for_log = agent.llm.model
    
    session = AgentSession(
        stt=deepgram.STT(model="nova-2", language="en-US", api_key=os.getenv("DEEPGRAM_API_KEY")),
        llm=agent.llm,  # OpenAI plugin LLM instance (supports GPT-5 models directly)
        tts=tts,
        vad=vad,
        turn_detection=None
    )
    
    
    
    # Set the agent's room and session
    agent.room = ctx.room
    agent.agent_session = session
    
    # Start the session - with environment variable control for testing
    use_noise_cancellation = os.getenv("USE_NOISE_CANCELLATION", "true").lower() == "true"
    
    room_input_options = RoomInputOptions(text_enabled=True)
    if use_noise_cancellation:
        room_input_options.noise_cancellation = noise_cancellation.BVC()
        logger.info("🔇 Noise cancellation enabled")
    else:
        logger.info("🚫 Noise cancellation disabled via USE_NOISE_CANCELLATION environment variable")
    
    await session.start(
        room=ctx.room,
        agent=agent,
        room_input_options=room_input_options,
        room_output_options=RoomOutputOptions(
            transcription_enabled=True,
            sync_transcription=False
        )
    )
    
    # Start the conversation with greeting
    await agent.on_room_enter(ctx.room)
    
    logger.info("✅ Simple agent started successfully")
    logger.info("=" * 80)
    logger.info(f"🎯 SESSION READY - Room: {room_name_clean} | Botchain: {botchain_name} | Org: {org_name} | Model: {model_name_for_log}")
    logger.info("=" * 80)

if __name__ == "__main__":
    cli.run_app(WorkerOptions(entrypoint_fnc=entrypoint, prewarm_fnc=prewarm))