"""
Alive5 Simple Voice Agent - Single LLM with Function Calling
"""

import asyncio
import logging
import os
import random
import sys
from pathlib import Path
from typing import Dict, Any

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
from functions import handle_load_bot_flows, handle_faq_bot_request

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
    
    def __init__(self, room_name: str, botchain_name: str = "voice-1", org_name: str = "alive5stage0"):
        self.room_name = room_name
        self.botchain_name = botchain_name
        self.org_name = org_name
        self.room = None
        self.selected_voice = "e90c6678-f0d3-4767-9883-5d0ecf5894a8"  # Default voice
        self._turn_detection = None  # Required by Agent class
        
        # Flow management
        self.bot_template = None
        self.flow_states = {}  # Track current step of each flow
        
        # Create OpenAI LLM instance
        llm_instance = openai.LLM(model="gpt-4o", temperature=0.7)
        
        # Initialize the base Agent class (special instructions will be loaded later)
        system_prompt = get_system_prompt(botchain_name, org_name, "")
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
        
        return await handle_faq_bot_request(faq_question, bot_id, isVoice, waiting_callback)
    
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
    
    async def _get_special_instructions(self):
        """Get special instructions from session data"""
        try:
            if not self.room_name:
                return ""
            
            import httpx
            backend_url = os.getenv("BACKEND_URL", "http://18.210.238.67")
            async with httpx.AsyncClient(timeout=5.0) as client:
                response = await client.get(f"{backend_url}/api/sessions/{self.room_name}")
                if response.status_code == 200:
                    data = response.json()
                    instructions = data.get("user_data", {}).get("special_instructions", "")
                    if instructions:
                        logger.info(f"📝 Special instructions: {instructions[:100]}...")
                        return instructions
        except Exception as e:
            logger.error(f"Failed to get special instructions: {e}")
        return ""
    
    async def on_room_enter(self, room):
        """Called when agent enters the room - start with greeting"""
        # Update system prompt with special instructions
        special_instructions = await self._get_special_instructions()
        if special_instructions:
            updated_prompt = get_system_prompt(self.botchain_name, self.org_name, special_instructions)
            self.instructions = updated_prompt
        
        # Start the conversation with greeting
        await self._start_conversation()
    
    async def _start_conversation(self):
        """Start the conversation - let LLM handle flow loading and greeting"""
        try:
            # Use generate_reply to make the agent speak first
            if hasattr(self, "agent_session") and self.agent_session:
                await self.agent_session.generate_reply()
            else:
                logger.warning("⚠️ Agent session not available")
            
        except Exception as e:
            logger.error(f"❌ Error starting conversation: {e}")
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
    
    try:
        import httpx
        async with httpx.AsyncClient() as client:
            response = await client.get(f"{backend_url}/api/sessions/{ctx.room.name}")
            if response.status_code == 200:
                session_data = response.json()
                user_data = session_data.get("user_data", {})
                botchain_name = user_data.get("botchain_name", "voice-1")
                org_name = user_data.get("org_name", "alive5stage0")
                faq_isVoice = user_data.get("faq_isVoice", True)  # Default to concise responses
            else:
                logger.warning(f"⚠️ Could not fetch session data (status {response.status_code}), using defaults")
    except Exception as e:
        logger.warning(f"⚠️ Could not fetch session data: {e}, using defaults")
    
    # Connect to the room first - using same approach as working implementation
    await ctx.connect(auto_subscribe=AutoSubscribe.SUBSCRIBE_ALL)
    
    # Create and start the agent
    agent = SimpleVoiceAgent(ctx.room.name, botchain_name, org_name)
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
    
    # Start the agent session with proper STT/TTS configuration (like working implementation)
    session = AgentSession(
        stt=deepgram.STT(model="nova-2", language="en-US", api_key=os.getenv("DEEPGRAM_API_KEY")),
        llm=agent.llm,  # This should be the LLM with function tools
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
    logger.info(f"🎯 SESSION READY - Room: {ctx.room.name} | Botchain: {botchain_name} | Org: {org_name}")
    logger.info("=" * 80)

if __name__ == "__main__":
    cli.run_app(WorkerOptions(entrypoint_fnc=entrypoint, prewarm_fnc=prewarm))