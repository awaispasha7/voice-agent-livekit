"""
Voice Agent - Single LLM with Function Calling (Brand-Agnostic)
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
# Suppress transformers warnings about PyTorch/TensorFlow (turn detector uses ONNX)
os.environ["TRANSFORMERS_VERBOSITY"] = "error"
# Disable LiveKit telemetry to avoid opentelemetry dependency issues
os.environ["LIVEKIT_TELEMETRY_ENABLED"] = "false"

# Simple logging configuration - just remove timestamps
logging.basicConfig(level=logging.INFO, format="%(message)s", force=True)

from dotenv import load_dotenv
from livekit.agents import (
    JobContext, WorkerOptions, cli, Agent, AgentSession,
    function_tool, RunContext, RoomInputOptions, RoomOutputOptions, AutoSubscribe
)
from livekit import rtc
from livekit.plugins import openai, deepgram, cartesia, silero, noise_cancellation, aws
from livekit.plugins.turn_detector.english import EnglishModel
from livekit.plugins.turn_detector.multilingual import MultilingualModel

from system_prompt import get_system_prompt
from functions import handle_load_bot_flows, handle_faq_bot_request, handle_bedrock_knowledge_base_request

# Load environment variables
load_dotenv(Path(__file__).parent / "../../.env")

# Create our logger
logger = logging.getLogger("simple-agent")
logger.setLevel(logging.DEBUG)  # Enable debug logging for transcription events

# Reduce LiveKit agent logging verbosity (but keep worker connection logs)
logging.getLogger("livekit.agents.worker").setLevel(logging.INFO)  # Keep worker connection logs
logging.getLogger("livekit.agents").setLevel(logging.WARNING)
logging.getLogger("livekit").setLevel(logging.WARNING)
logging.getLogger("livekit.plugins").setLevel(logging.WARNING)
logging.getLogger("livekit.plugins.cartesia").setLevel(logging.WARNING)
logging.getLogger("livekit.plugins.deepgram").setLevel(logging.WARNING)
logging.getLogger("livekit.plugins.openai").setLevel(logging.WARNING)
logging.getLogger("livekit.plugins.aws").setLevel(logging.WARNING)

# Reduce OpenAI and HTTP client logging verbosity
logging.getLogger("openai").setLevel(logging.WARNING)
logging.getLogger("openai._base_client").setLevel(logging.WARNING)
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("httpcore").setLevel(logging.WARNING)

# Suppress transformers warnings (turn detector uses ONNX, not PyTorch/TensorFlow)
logging.getLogger("transformers").setLevel(logging.ERROR)
# Suppress the specific warning about PyTorch/TensorFlow/Flax not being found
import warnings
warnings.filterwarnings("ignore", message=".*PyTorch.*TensorFlow.*Flax.*")
warnings.filterwarnings("ignore", message=".*Models won't be available.*")
warnings.filterwarnings("ignore", category=UserWarning, module="transformers")
logging.getLogger("httpcore.connection").setLevel(logging.WARNING)
logging.getLogger("httpcore.http11").setLevel(logging.WARNING)

# Reduce boto3/botocore logging verbosity (AWS SDK)
logging.getLogger("boto3").setLevel(logging.WARNING)
logging.getLogger("botocore").setLevel(logging.WARNING)
logging.getLogger("botocore.hooks").setLevel(logging.WARNING)
logging.getLogger("botocore.loaders").setLevel(logging.WARNING)
logging.getLogger("botocore.auth").setLevel(logging.WARNING)
logging.getLogger("botocore.endpoint").setLevel(logging.WARNING)
logging.getLogger("botocore.httpsession").setLevel(logging.WARNING)
logging.getLogger("botocore.parsers").setLevel(logging.WARNING)
logging.getLogger("botocore.retryhandler").setLevel(logging.WARNING)
logging.getLogger("urllib3").setLevel(logging.WARNING)
logging.getLogger("urllib3.connectionpool").setLevel(logging.WARNING)

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
        # Initialize FAQ bot ID to None - will be set during entrypoint
        self.faq_bot_id = None
        # Flag to track if we're using Nova Sonic (speech-to-speech model)
        self._using_nova = False
        
        # Flow management
        self.bot_template = None
        self.flow_states = {}  # Track current step of each flow
        self._flows_loaded = False  # Track if flows have been loaded to prevent multiple calls
        
        # CRM data collection
        self.collected_data = {
            "full_name": None,
            "email": None,
            "phone": None,
            "notes_entry": []
        }
        
        # Alive5 livechat session data (will be set during entrypoint)
        self.alive5_thread_id = None
        self.alive5_crm_id = None
        self.alive5_channel_id = None
        # Note: API key is now read from environment, not stored per session
        self.alive5_widget_id = None
        self.alive5_org_name = org_name
        self.alive5_botchain = botchain_name
        self._alive5_message_count = 0  # Track message count for is_new flag
        self._alive5_message_queue = []  # Queue messages until session data is ready
        self._alive5_thread_created_at = None  # Store original thread creation timestamp
        
        # Get LLM provider from env (bedrock, openai, or nova)
        llm_provider = os.getenv("LLM_PROVIDER", "bedrock").lower()
        
        import httpx
        llm_timeout = httpx.Timeout(connect=30.0, read=60.0, write=30.0, pool=30.0)
        
        # Initialize LLM based on provider
        if llm_provider == "bedrock":
            # Use AWS Bedrock LLM (data stays in AWS, no training on your data)
            bedrock_model = os.getenv("BEDROCK_MODEL", "anthropic.claude-3-5-sonnet-20240620-v1:0")
            bedrock_region = os.getenv("BEDROCK_REGION", "us-east-1")
            
            try:
                llm_instance = aws.LLM(
                    model=bedrock_model,
                    region=bedrock_region,
                    temperature=0.3
                )
                logger.info(f"✅ AWS Bedrock LLM initialized: {bedrock_model}")
                self._inference_model_id = None
            except Exception as e:
                logger.error(f"❌ Failed to initialize Bedrock LLM with {bedrock_model}: {e}")
                logger.warning(f"🔄 Falling back to OpenAI gpt-4o-mini")
                try:
                    llm_instance = openai.LLM(model="gpt-4o-mini", temperature=0.3, timeout=llm_timeout)
                    logger.info(f"✅ Fallback LLM initialized: gpt-4o-mini (OpenAI)")
                except Exception as e2:
                    logger.error(f"❌ Fallback also failed: {e2}")
                    raise Exception(f"Could not initialize any LLM. Bedrock error: {e}, OpenAI fallback error: {e2}")
                self._inference_model_id = None
        elif llm_provider == "nova":
            # Use Amazon Nova Sonic (speech-to-speech, realtime model)
            # Nova Sonic is a complete speech-to-speech solution (no separate STT/TTS needed)
            nova_voice = os.getenv("NOVA_VOICE", None)  # Optional voice name
            nova_region = os.getenv("NOVA_REGION", "us-east-1")
            
            try:
                # Nova Sonic uses realtime.RealtimeModel() which is a speech-to-speech model
                llm_instance = aws.realtime.RealtimeModel(
                    voice=nova_voice if nova_voice and nova_voice != "default" else None,
                    region=nova_region
                )
                logger.info(f"✅ Amazon Nova Sonic initialized (region: {nova_region}, voice: {nova_voice or 'default'})")
                self._inference_model_id = None
                # Mark that we're using Nova Sonic (speech-to-speech, no separate STT/TTS)
                self._using_nova = True
            except Exception as e:
                logger.error(f"❌ Failed to initialize Nova Sonic: {e}")
                logger.warning(f"🔄 Falling back to OpenAI gpt-4o-mini")
                try:
                    llm_instance = openai.LLM(model="gpt-4o-mini", temperature=0.3, timeout=llm_timeout)
                    logger.info(f"✅ Fallback LLM initialized: gpt-4o-mini (OpenAI)")
                    self._using_nova = False
                except Exception as e2:
                    logger.error(f"❌ Fallback also failed: {e2}")
                    raise Exception(f"Could not initialize any LLM. Nova Sonic error: {e}, OpenAI fallback error: {e2}")
                self._inference_model_id = None
        else:
            # Use OpenAI LLM (original implementation)
            openai_model = os.getenv("OPENAI_MODEL", "gpt-4o")
            
            try:
                llm_instance = openai.LLM(model=openai_model, temperature=0.3, timeout=llm_timeout)
                logger.info(f"✅ OpenAI LLM initialized: {openai_model}")
                self._inference_model_id = None
            except Exception as e:
                logger.error(f"❌ Failed to initialize OpenAI plugin LLM with {openai_model}: {e}")
                logger.warning(f"🔄 Falling back to gpt-4o-mini")
                try:
                    llm_instance = openai.LLM(model="gpt-4o-mini", temperature=0.3, timeout=llm_timeout)
                    logger.info(f"✅ Fallback LLM initialized: gpt-4o-mini")
                except Exception as e2:
                    logger.error(f"❌ Fallback also failed: {e2}")
                    raise Exception(f"Could not initialize any LLM. Original error: {e}, Fallback error: {e2}")
                self._inference_model_id = None
        
        # Initialize the base Agent class with special instructions
        system_prompt = get_system_prompt(botchain_name, org_name, special_instructions)
        super().__init__(instructions=system_prompt, llm=llm_instance)
        
    
    
    @function_tool()
    async def transfer_call_to_human(self, context: RunContext, transfer_number: Optional[str] = None) -> Dict[str, Any]:
        """Transfer the current phone call to a human agent or phone number.
        
        **IMPORTANT: This function only works for phone calls, not web sessions.**
        For web sessions, it will return a message explaining that transfer is not available.
        
        Args:
            transfer_number: Phone number to transfer to (e.g., "+18555518858"). 
                           Optional - if not provided, uses default call center number from environment.
                           If no transfer number is configured, returns helpful message.
        
        Returns:
            Success status and message. If this is a web session (no call_control_id), 
            returns success=False with message explaining transfer is not available for web interface.
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
                        "is_web_session": False,
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
                    source = session_data.get("user_data", {}).get("source")
                    
                    # Check if this is a web session (no call_control_id or source is not telnyx_phone)
                    if not call_control_id or source != "telnyx_phone":
                        logger.info("ℹ️ Transfer requested for web session - not available")
                        return {
                            "success": False,
                            "is_web_session": True,
                            "message": "I'm sorry, call transfers are only available for phone calls, not through this web interface. Is there anything else I can help you with today?"
                        }
                    
                    if call_control_id:
                        # Return success immediately so agent can speak acknowledgment first
                        # The actual transfer will happen in the background after a delay
                        logger.info(f"📞 Transfer requested - will execute after agent speaks acknowledgment")
                        
                        # Schedule the transfer to happen after agent speaks (in background)
                        import asyncio
                        async def execute_transfer_after_delay():
                            # Wait for agent to speak acknowledgment (3-4 seconds should be enough)
                            await asyncio.sleep(4.0)
                            try:
                                async with httpx.AsyncClient(timeout=15.0) as transfer_client:
                                    transfer_response = await transfer_client.post(
                                        f"{backend_url}/api/telnyx/transfer",
                                        json={
                                            "room_name": room_name_clean,
                                            "call_control_id": call_control_id,
                                            "transfer_to": transfer_number
                                        }
                                    )
                                    
                                    if transfer_response.status_code == 200:
                                        logger.info(f"✅ Call transferred to {transfer_number} (after acknowledgment)")
                                    else:
                                        logger.error(f"❌ Transfer failed: {transfer_response.status_code} - {transfer_response.text}")
                            except Exception as e:
                                logger.error(f"❌ Error executing transfer: {e}")
                        
                        # Start the transfer task in the background
                        asyncio.create_task(execute_transfer_after_delay())
                        
                        # Return success immediately so agent can speak
                        return {
                            "success": True,
                            "is_web_session": False,
                            "message": "Transfer will happen after you speak the acknowledgment message."
                        }
                    else:
                        logger.warning("⚠️ No call_control_id found - not a phone call")
                        return {
                            "success": False,
                            "is_web_session": True,
                            "message": "I'm sorry, call transfers are only available for phone calls, not through this web interface. Is there anything else I can help you with today?"
                        }
                else:
                    logger.error(f"❌ Could not get session data: {response.status_code} - {response.text}")
                    logger.error(f"   Original room name: {self.room_name}")
                    logger.error(f"   Cleaned room name: {room_name_clean}")
                    logger.error(f"   Encoded room name: {encoded_room_name}")
                    return {
                        "success": False,
                        "is_web_session": False,
                        "message": "Unable to process transfer request."
                    }
        except Exception as e:
            logger.error(f"❌ Error transferring call: {e}")
            return {
                "success": False,
                "is_web_session": False,
                "message": "I'm having trouble transferring you right now."
            }
    
    @function_tool()
    async def save_collected_data(self, context: RunContext, field_name: str, value: str) -> Dict[str, Any]:
        """Save user response to collected_data and update CRM in real-time.
        
        Call this function whenever a flow question has a 'save_data_to' field and the user provides an answer.
        This will immediately update the widget/portal with the new CRM data.
        
        Args:
            field_name: The field name from save_data_to (e.g., "full_name", "email", "phone", "notes_entry")
            value: The user's response to save.
        
        Returns:
            Success status and a message.
        """
        logger.info(f"💾 Saving collected data: {field_name} = {value}")
        
        try:
            # Update internal collected_data
            if field_name == "full_name":
                self.collected_data["full_name"] = value
                # Attempt to split into first and last name for CRM update
                name_parts = value.strip().split(' ', 1)
                first_name = name_parts[0] if name_parts else ""
                last_name = name_parts[1] if len(name_parts) > 1 else ""
                # Update CRM immediately in real-time
                await self._update_crm_data(first_name=first_name, last_name=last_name)
            elif field_name == "email":
                self.collected_data["email"] = value
                # Update CRM immediately in real-time
                await self._update_crm_data(email=value)
            elif field_name == "phone":
                self.collected_data["phone"] = value
                # Update CRM immediately in real-time
                await self._update_crm_data(phone=value)
            elif field_name == "notes_entry":
                if "notes_entry" not in self.collected_data:
                    self.collected_data["notes_entry"] = []
                self.collected_data["notes_entry"].append(value)
                # Send all notes as a single string
                notes_str = " | ".join(self.collected_data.get("notes_entry", []))
                await self._update_crm_data(notes=notes_str)
            else:
                logger.warning(f"⚠️ Unknown field_name for saving data: {field_name}")
                return {
                    "success": False,
                    "message": f"Unknown field_name: {field_name}"
                }
            
            logger.info(f"✅ Data saved and CRM updated in real-time for {field_name}")
            logger.info(f"   - Current collected_data: {self.collected_data}")
            
            return {
                "success": True,
                "message": f"Data for {field_name} saved successfully and CRM updated in real-time."
            }
        except Exception as e:
            logger.error(f"❌ Error saving collected data: {e}", exc_info=True)
            return {
                "success": False,
                "message": f"Error saving data: {str(e)}"
            }
    
    # COMMENTED OUT: CRM data submission - focusing on message replication only
    # @function_tool()
    # async def submit_crm_data(self, context: RunContext) -> Dict[str, Any]:
    #     """Submit collected customer data to CRM at the end of conversation.
    #     Call this when you have collected the customer's information (name, email, notes) and the conversation is ending.
    #     
    #     NOTE: This function submits data in the background. The agent should tell the user to hold on while saving.
    #     """
    #     try:
    #         logger.info(f"📤 submit_crm_data() called - preparing to submit CRM data")
    #         # Log collected data before submission
    #         logger.info(f"📋 Collected data before submission:")
    #         logger.info(f"   - full_name: {self.collected_data.get('full_name')}")
    #         logger.info(f"   - email: {self.collected_data.get('email')}")
    #         logger.info(f"   - phone: {self.collected_data.get('phone')}")
    #         logger.info(f"   - notes_entry: {self.collected_data.get('notes_entry')}")
    #         
    #         # Prepare data for submission
    #         crm_data = {
    #             "room_name": self.room_name,
    #             "botchain_name": self.botchain_name,
    #             "org_name": self.org_name,
    #             "full_name": self.collected_data.get("full_name"),
    #             "email": self.collected_data.get("email"),
    #             "phone": self.collected_data.get("phone"),
    #             "notes": " | ".join(self.collected_data.get("notes_entry", [])) if self.collected_data.get("notes_entry") else None
    #         }
    #         
    #         # Submit to backend API in background (non-blocking)
    #         # This prevents awkward silence while waiting for API response
    #         async def submit_in_background():
    #             try:
    #                 import httpx
    #                 backend_url = os.getenv("BACKEND_URL", "http://18.210.238.67")
    #                 async with httpx.AsyncClient(timeout=60.0) as client:  # Increased timeout for slow API
    #                     response = await client.post(
    #                         f"{backend_url}/api/submit_crm",
    #                         json=crm_data
    #                     )
    #                     
    #                     if response.status_code == 200:
    #                         logger.info(f"✅ CRM data submitted successfully (background)")
    #                     else:
    #                         logger.error(f"❌ CRM submission failed: {response.status_code}")
    #             except Exception as e:
    #                 logger.error(f"❌ Error submitting CRM data (background): {e}")
    #         
    #         # Start submission in background
    #         asyncio.create_task(submit_in_background())
    #         
    #         # Return immediately so agent can continue speaking
    #         # Agent should tell user "Please hold on while I save your information"
    #         return {
    #             "success": True,
    #             "message": "I'm saving your information now. Please hold on for just a moment.",
    #             "submitting": True  # Indicates submission is in progress
    #         }
    #     except Exception as e:
    #         logger.error(f"❌ Error preparing CRM submission: {e}")
    #         return {
    #             "success": False,
    #             "message": "Information noted, though there was a technical issue with submission."
    #         }
    
    @function_tool()
    async def faq_bot_request(self, context: RunContext, faq_question: str, bot_id: str = None, isVoice: bool = None) -> Dict[str, Any]:
        """Call the FAQ bot API to get answers about company services, pricing, features, or company information.
        
        Args:
            faq_question: The user's question about the company
            bot_id: The FAQ bot ID (if None, uses session data)
            isVoice: Whether this is a voice interaction (if None, uses agent's faq_isVoice setting)
        """
        # logger.info(f"🔧 FAQ bot request: {faq_question}")
        
        # Use dynamic FAQ bot ID from agent instance or session data if not provided
        # CRITICAL: Always use the stored FAQ bot ID from agent instance, ignore LLM-provided bot_id
        # The LLM should NOT be providing bot_id - it should always be None
        # logger.info(f"🔍 FAQ bot ID check - LLM provided bot_id: {bot_id}, type: {type(bot_id)}")
        
        # CRITICAL: Check agent instance attribute directly - this is the source of truth
        stored_faq_bot_id = getattr(self, 'faq_bot_id', None)
        # logger.info(f"🔍 FAQ bot ID check - stored value on agent: {stored_faq_bot_id}, type: {type(stored_faq_bot_id)}")
        
        # ALWAYS use stored FAQ bot ID from agent instance if available (ignore LLM-provided bot_id)
        if stored_faq_bot_id and stored_faq_bot_id.strip():
            bot_id = stored_faq_bot_id
            # logger.info(f"🤖 Using FAQ bot ID from agent instance: {bot_id}")
        elif bot_id and bot_id.strip():
            # If LLM provided a bot_id and we don't have one stored, use it (but log warning)
            logger.warning(f"⚠️ Using LLM-provided bot_id (no stored value): {bot_id}")
        else:
            logger.warning(f"⚠️ FAQ bot ID not available on agent instance (value: {stored_faq_bot_id}), fetching from session data...")
            # Fallback: fetch from session data
            bot_id = await self._get_faq_bot_id()
        
        # Use agent's faq_isVoice if isVoice not specified
        if isVoice is None:
            isVoice = getattr(self, 'faq_isVoice', True)
        
        # Call FAQ with waiting callback (the function will provide the "Let me check that" message)
        async def waiting_callback(message):
            if hasattr(self, "agent_session") and self.agent_session:
                await self.agent_session.say(message)
        
        # Get org_name from agent instance (set during initialization) or session data
        org_name = None
        if hasattr(self, 'org_name') and self.org_name:
            org_name = self.org_name
            logger.debug(f"Using org_name from agent instance: {org_name}")
        else:
            # Fallback: fetch from session data
            try:
                import httpx
                from urllib.parse import quote
                backend_url = os.getenv("BACKEND_URL", "http://18.210.238.67")
                encoded_room_name = quote(self.room_name, safe='')
                async with httpx.AsyncClient(timeout=5.0) as client:
                    response = await client.get(f"{backend_url}/api/sessions/{encoded_room_name}")
                    if response.status_code == 200:
                        session_data = response.json()
                        user_data = session_data.get("user_data", {})
                        org_name = user_data.get("org_name")
                        if org_name:
                            # Store it on the agent instance for future use
                            self.org_name = org_name
            except Exception as e:
                logger.debug(f"Could not fetch org_name for filtering: {e}")
        
        # Log the FAQ bot ID and org_name being used
        # logger.info(f"🔍 FAQ request - Bot ID: {bot_id}, Org Name: {org_name}")
        
        # Try Bedrock Knowledge Base first (faster), fallback to FAQ API if needed
        # Pass FAQ bot ID and org_name for filtering
        return await handle_bedrock_knowledge_base_request(
            faq_question, 
            max_results=5, 
            waiting_callback=waiting_callback,
            faq_bot_id=bot_id,
            org_name=org_name
        )
    
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
        """Get FAQ bot ID from agent instance or session data"""
        # First check if it's stored on the agent instance (set during initialization)
        if hasattr(self, 'faq_bot_id') and self.faq_bot_id:
            # logger.info(f"🤖 Using FAQ bot from agent instance: {self.faq_bot_id}")
            return self.faq_bot_id
        
        # Fallback: fetch from session data via backend API
        try:
            if not self.room_name:
                logger.warning("⚠️ No room name available, using default FAQ bot")
                return "faq_b9952a56-fc7b-41c9-b0a0-5c662ddb039e"  # Default FAQ bot
            
            import httpx
            from urllib.parse import quote
            backend_url = os.getenv("BACKEND_URL", "http://18.210.238.67")
            encoded_room_name = quote(self.room_name, safe='')
            async with httpx.AsyncClient(timeout=5.0) as client:
                response = await client.get(f"{backend_url}/api/sessions/{encoded_room_name}")
                if response.status_code == 200:
                    data = response.json()
                    faq_bot_id = data.get("user_data", {}).get("faq_bot_id")
                    if faq_bot_id:
                        # logger.info(f"🤖 Using FAQ bot from session data: {faq_bot_id}")
                        # Store it on the agent instance for future use
                        self.faq_bot_id = faq_bot_id
                        return faq_bot_id
                    else:
                        logger.warning(f"⚠️ FAQ bot ID not found in session data for room: {self.room_name}")
                else:
                    logger.warning(f"⚠️ Failed to fetch session data: {response.status_code}")
        except Exception as e:
            logger.error(f"❌ Failed to get FAQ bot ID: {e}")
        
        logger.warning("⚠️ Using default FAQ bot ID")
        return "faq_b9952a56-fc7b-41c9-b0a0-5c662ddb039e"  # Default FAQ bot
    
    async def on_room_enter(self, room):
        """Called when agent enters the room - start with greeting"""
        # Check if room is still connected before starting
        if not room or not hasattr(room, "name"):
            logger.warning("⚠️ Room not available, skipping greeting")
            return
        # Track session start time
        import time
        self._session_start_time = int(time.time() * 1000)
        # Start the conversation with greeting
        await self._start_conversation()
    
    async def on_session_end(self):
        """Called when session is ending - cleanup livechat (skip for phone calls)"""
        # Skip livechat cleanup for phone calls
        if self.room_name.startswith("telnyx_call_"):
            logger.info(f"📞 Phone call ending - skipping livechat cleanup")
            return
        
        logger.info(f"👋 Session ending for room: {self.room_name}")
        
        # Mark session as closing to prevent further operations
        if hasattr(self, "agent_session") and self.agent_session:
            try:
                # Check if session has a closing flag
                if hasattr(self.agent_session, "_closing"):
                    self.agent_session._closing = True
            except:
                pass
        
        # Send end_voice_chat event via socket before calling backend
        # According to Alive5 docs: end_voice_chat closes the chat and thread
        try:
            if hasattr(self, 'room') and self.room and hasattr(self, 'alive5_thread_id') and self.alive5_thread_id:
                import json
                
                # Get voice_agent_id (stored from init_voice_agent_ack, or use default)
                voice_agent_id = getattr(self, 'alive5_voice_agent_id', '')
                org_name = getattr(self, 'alive5_org_name', getattr(self, 'org_name', 'alive5stage0'))
                
                socket_instruction = {
                    "action": "emit",
                    "event": "end_voice_chat",
                    "payload": {
                        "end_by": "voice_agent",  # Agent ended the call (session cleanup)
                        "message_content": "Voice call completed by agent",
                        "org_name": org_name,
                        "thread_id": self.alive5_thread_id,
                        "voice_agent_id": voice_agent_id
                    }
                }
                try:
                    await self.room.local_participant.publish_data(
                        json.dumps(socket_instruction).encode('utf-8'),
                        topic="lk.alive5.socket"
                    )
                    logger.info(f"📤 end_voice_chat event sent via data channel")
                except Exception as e:
                    logger.warning(f"⚠️ Failed to send end_voice_chat via data channel: {e}")
        except Exception as e:
            logger.warning(f"⚠️ Could not send end_voice_chat event: {e}")
        
        try:
            import httpx
            backend_url = os.getenv("BACKEND_URL", "http://18.210.238.67")
            async with httpx.AsyncClient(timeout=3.0) as client:  # Reduced timeout for faster cleanup
                response = await client.post(
                    f"{backend_url}/api/end_livechat",
                    params={"room_name": self.room_name}
                )
                if response.status_code == 200:
                    result = response.json()
                    logger.info(f"✅ Livechat session ended - Thread: {result.get('thread_id', 'N/A')}, CRM: {result.get('crm_id', 'N/A')}")
                else:
                    logger.warning(f"⚠️ Livechat end failed: {response.status_code}")
        except Exception as e:
            logger.warning(f"⚠️ Could not end livechat: {e}")
        
        logger.info(f"✅ Session cleanup complete for room: {self.room_name}")
    
    async def _start_conversation(self):
        """Start the conversation - preload flows, then let LLM handle greeting"""
        # Always ensure flows are loaded before we even consider speaking
        if not self._flows_loaded:
            # logger.info(f"🔧 Loading bot flows for {self.botchain_name}")
            try:
                # Call the underlying function directly (no longer a tool method)
                result = await handle_load_bot_flows(self.botchain_name, self.org_name)
                
                # Cache the result if successful
                if result.get("success") and result.get("data"):
                    self.bot_template = result.get("data")
                    self._flows_loaded = True
                    logger.info("✅ Bot flows preloaded successfully before greeting")
                    
                    # Inject raw flow JSON into system prompt so agent can see them
                    try:
                        import json
                        flows_json = json.dumps(self.bot_template, indent=2)
                        flows_section = f"\n\n──────────────────────────────\n:book: LOADED BOT FLOWS (JSON)\n──────────────────────────────\n**The following is the complete flow structure. Use it as your source of truth - DO NOT make up questions.**\n\n```json\n{flows_json}\n```\n"
                        current_instructions = self.instructions
                        updated_instructions = current_instructions + flows_section
                        # Use agent's own update_instructions method (handles activity automatically)
                        await self.update_instructions(updated_instructions)
                        logger.info("✅ Flow data injected into system prompt")
                    except Exception as update_error:
                        logger.warning(f"⚠️ Could not update instructions with flows: {update_error}")
                else:
                    logger.error(f"❌ Failed to load bot flows: {result.get('error', 'Unknown error')}")
                    raise Exception(f"Failed to load bot flows: {result.get('error', 'Unknown error')}")
            except Exception as preload_error:
                logger.error(f"❌ Failed to preload bot flows before greeting: {preload_error}", exc_info=True)
                if hasattr(self, "agent_session") and self.agent_session:
                    await self.agent_session.say(
                        "I'm having trouble getting set up right now. Please try again in a moment."
                    )
                return
        
        # Nova Sonic doesn't support unprompted generation - it requires user input first
        if getattr(self, '_using_nova', False):
            logger.info("🎙️ Nova Sonic detected - skipping proactive greeting (waits for user input first)")
            return  # Nova Sonic will respond naturally when user speaks
        
        try:
            # Check if session is still active before using it
            if not hasattr(self, "agent_session") or not self.agent_session:
                logger.warning("⚠️ Agent session not available")
                return
            
            # Check if session is closing/closed
            if hasattr(self.agent_session, "_closing") and self.agent_session._closing:
                logger.warning("⚠️ Agent session is closing, skipping greeting")
                return
            
            # Use generate_reply to make the agent speak first (flows already cached)
            await self.agent_session.generate_reply()
            
        except RuntimeError as e:
            # Handle session closing errors gracefully
            if "closing" in str(e).lower() or "cannot use" in str(e).lower():
                logger.info("ℹ️ Session is closing, skipping greeting (this is normal on disconnect)")
                return
            raise
        except Exception as e:
            # Handle Nova Sonic "unprompted generation" error gracefully
            if "unprompted generation" in str(e).lower() or "realtime" in str(e).lower():
                logger.info("🎙️ Nova Sonic requires user input first - this is expected behavior")
                return
            
            logger.error(f"❌ Error starting conversation: {e}")
            # Only try fallback if session is still active
            try:
                if hasattr(self, "agent_session") and self.agent_session:
                    # Check if session is closing before using it
                    if not (hasattr(self.agent_session, "_closing") and self.agent_session._closing):
                        await self.agent_session.say("Failed to load the bot flows. But you can still speak with me naturally.")
            except RuntimeError:
                # Session is closing, ignore
                pass
    
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
    
    async def _send_message_to_alive5(self, message_content: str, is_agent: bool = False):
        """Send a message to Alive5 livechat via Socket.io
        
        This function is completely isolated from agent processing - any errors here
        will NOT affect the agent's conversation flow or context.
        """
        try:
            # If session data is not ready, queue the message
            if not self.alive5_thread_id or not self.alive5_channel_id:
                self._alive5_message_queue.append((message_content, is_agent))
                return
            
            # Process queued messages first
            if self._alive5_message_queue:
                queued_messages = self._alive5_message_queue.copy()
                self._alive5_message_queue.clear()
                for queued_content, queued_is_agent in queued_messages:
                    try:
                        await asyncio.wait_for(
                            self._send_message_to_alive5_internal(queued_content, queued_is_agent),
                            timeout=5.0  # 5 second timeout to prevent hanging
                        )
                    except asyncio.TimeoutError:
                        logger.warning(f"⚠️ Timeout sending queued message to Alive5")
                    except Exception as e:
                        logger.warning(f"⚠️ Error sending queued message to Alive5: {e}")
            
            # Send current message with timeout
            try:
                await asyncio.wait_for(
                    self._send_message_to_alive5_internal(message_content, is_agent),
                    timeout=5.0  # 5 second timeout to prevent hanging
                )
            except asyncio.TimeoutError:
                logger.warning(f"⚠️ Timeout sending message to Alive5")
            except Exception as e:
                logger.warning(f"⚠️ Error sending message to Alive5: {e}")
            
        except Exception as e:
            # Completely isolate errors - never let them affect agent processing
            logger.warning(f"⚠️ Could not send message to Alive5 (isolated error): {e}")
    
    def _build_query_string(self) -> dict:
        """Build query_string with collected CRM data"""
        full_name = self.collected_data.get("full_name", "")
        name_parts = full_name.split(' ', 1) if full_name else ["", ""]
        
        return {
            "first_name": name_parts[0] if len(name_parts) > 0 else "",
            "last_name": name_parts[1] if len(name_parts) > 1 else "",
            "email": self.collected_data.get("email", ""),
            "notes": " | ".join(self.collected_data.get("notes_entry", [])) if self.collected_data.get("notes_entry") else "",
            "crm_id": self.alive5_crm_id or "",
            "agent_email": "",
            "accountid": "",
            "company": "",
            "companytitle": "",
            "phone_mobile": self.collected_data.get("phone", ""),
            "cartid": "",
            "utm_source": "",
            "utm_medium": "",
            "utm_campaign": "",
            "utm_term": "",
            "utm_content": "",
            "a5_custom0": "",
            "a5_custom1": "",
            "a5_custom2": "",
            "a5_custom3": "",
            "a5_custom4": "",
            "a5_custom5": "",
            "a5_custom6": "",
            "a5_custom7": "",
            "a5_custom8": "",
            "a5_custom9": "",
            "a5_referrer_url": "",
            "a5_page_url": ""
        }
    
    def _build_crm_data(self, timestamp: int) -> dict:
        """Build crmData object with collected CRM data"""
        full_name = self.collected_data.get("full_name", "")
        name_parts = full_name.split(' ', 1) if full_name else ["", ""]
        
        return {
            "allow_zapier_syns": True,
            "assigned_user": [],
            "created_at": timestamp,
            "crm_id": self.alive5_crm_id or "",
            "crm_thread_type": "livechat",
            "crm_type": "livechat",
            "org_name": self.alive5_org_name,
            "updated_at": timestamp,
            "firstName": name_parts[0] if len(name_parts) > 0 else "",
            "lastName": name_parts[1] if len(name_parts) > 1 else "",
            "email": self.collected_data.get("email", ""),
            "phoneMobile": self.collected_data.get("phone", ""),
            "notes": " | ".join(self.collected_data.get("notes_entry", [])) if self.collected_data.get("notes_entry") else ""
        }
    
    async def _update_crm_data(self, first_name: str = None, last_name: str = None, email: str = None, phone: str = None, notes: str = None):
        """Update CRM data using new voice agent socket save_crm_data event"""
        try:
            if not self.alive5_crm_id:
                logger.warning("⚠️ Cannot update CRM data - session not initialized")
                return
            
            # Update collected_data
            if first_name:
                # Split if full name provided
                name_parts = first_name.strip().split(' ', 1)
                if len(name_parts) == 2:
                    self.collected_data["full_name"] = first_name
                    first_name = name_parts[0]
                    last_name = name_parts[1]
                else:
                    self.collected_data["full_name"] = first_name
            if last_name:
                if self.collected_data.get("full_name"):
                    self.collected_data["full_name"] = f"{first_name or ''} {last_name}".strip()
            if email:
                self.collected_data["email"] = email
            if phone:
                self.collected_data["phone"] = phone
            if notes:
                # Store notes as a list for internal tracking, but send as string to CRM
                if "notes_entry" not in self.collected_data:
                    self.collected_data["notes_entry"] = []
                # If notes is a pipe-separated string, split it; otherwise treat as single entry
                if " | " in notes:
                    self.collected_data["notes_entry"] = notes.split(" | ")
                else:
                    self.collected_data["notes_entry"] = [notes] if notes else []
            
            # Send CRM update instructions to frontend via data channel
            if not hasattr(self, 'room') or not self.room:
                return
            
            import json
            
            updates = []
            if first_name:
                updates.append({"key": "first_name", "value": first_name})
            if last_name:
                updates.append({"key": "last_name", "value": last_name})
            if email:
                updates.append({"key": "email", "value": email})
            if phone:
                updates.append({"key": "phone", "value": phone})
            if notes:
                updates.append({"key": "notes", "value": notes})
            
            for update in updates:
                socket_instruction = {
                    "action": "emit",
                    "event": "save_crm_data",
                    "payload": {
                        "crm_id": self.alive5_crm_id or "",
                        "key": update["key"],
                        "value": update["value"]
                    }
                }
                
                await self.room.local_participant.publish_data(
                    json.dumps(socket_instruction).encode('utf-8'),
                    topic="lk.alive5.socket"
                )
                
        except Exception as e:
            logger.warning(f"⚠️ Could not update CRM data: {e}")
    
    async def _send_message_to_alive5_internal(self, message_content: str, is_agent: bool = False):
        """Send message to Alive5 via frontend socket (data channel instruction)"""
        try:
            if not hasattr(self, 'room') or not self.room:
                return
            
            self._alive5_message_count += 1
            
            # Log message (minimal)
            if is_agent:
                logger.info(f"📤 Agent: {message_content[:80]}...")
            else:
                logger.info(f"📤 User: {message_content[:80]}...")
            
            # Send socket instruction to frontend via data channel
            import json
            # Include is_agent flag so frontend can add voiceAgentId for agent messages
            # Server will map voiceAgentId to created_by and user_id automatically
            socket_instruction = {
                "action": "emit",
                "event": "post_message",
                "payload": {
                    "thread_id": self.alive5_thread_id,
                    "crm_id": self.alive5_crm_id or "",
                    "message_content": message_content,
                    "message_type": "livechat",  # Changed to match what Alive5 expects
                    "is_agent": is_agent  # Frontend will use this to add voiceAgentId for agent messages
                    # DO NOT set created_by or user_id - server will set these based on voiceAgentId presence
                }
            }
            
            try:
                await self.room.local_participant.publish_data(
                    json.dumps(socket_instruction).encode('utf-8'),
                    topic="lk.alive5.socket"
                )
                logger.info(f"📤 Message instruction sent via data channel: {message_content[:50]}...")
            except Exception as e:
                logger.error(f"❌ Failed to send message instruction via data channel: {e}")
            
        except Exception as e:
            logger.error(f"❌ Could not send message to Alive5: {e}")
    
    async def _auto_detect_and_save_data(self, user_text: str):
        """Automatically detect if user provided name/email/phone and save it if not already saved"""
        try:
            import re
            user_text_clean = user_text.strip()
            
            # Check if we already have this data saved
            # Only auto-save if the field is empty (not already saved)
            
            # Detect email pattern
            email_pattern = r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b'
            email_match = re.search(email_pattern, user_text_clean)
            if email_match and not self.collected_data.get("email"):
                email = email_match.group(0)
                logger.info(f"🔍 Auto-detected email in user message: {email}")
                # Call save_collected_data via the function (which will update CRM)
                await self.save_collected_data(None, "email", email)
            
            # Detect phone pattern (US format: (xxx) xxx-xxxx, xxx-xxx-xxxx, xxx.xxx.xxxx, or 10+ digits)
            phone_pattern = r'(\+?1[-.\s]?)?\(?([0-9]{3})\)?[-.\s]?([0-9]{3})[-.\s]?([0-9]{4})'
            phone_match = re.search(phone_pattern, user_text_clean)
            if phone_match and not self.collected_data.get("phone"):
                # Extract and normalize phone number
                phone_parts = phone_match.groups()
                phone = ''.join(filter(str.isdigit, ''.join(phone_parts[1:])))  # Get last 3 groups, remove non-digits
                if len(phone) == 10:  # Valid US phone number
                    phone_formatted = f"({phone[:3]}) {phone[3:6]}-{phone[6:]}"
                    logger.info(f"🔍 Auto-detected phone in user message: {phone_formatted}")
                    await self.save_collected_data(None, "phone", phone_formatted)
            
            # Detect name pattern (2-4 words, capitalized, not too long, doesn't look like email/phone)
            # Only if it's a short response (likely a name answer)
            if len(user_text_clean.split()) <= 4 and len(user_text_clean) < 50:
                # Check if it looks like a name (has capital letters, not all caps, not email/phone)
                if (not email_match and not phone_match and 
                    re.search(r'[A-Z][a-z]+', user_text_clean) and  # Has capitalized words
                    not self.collected_data.get("full_name")):
                    # Additional check: if user said something like "My name is X" or "It's X" or just "X"
                    name_patterns = [
                        r'(?:my name is|i\'?m|it\'?s|this is|call me|i am)\s+([A-Z][a-z]+(?:\s+[A-Z][a-z]+)?)',
                        r'^([A-Z][a-z]+(?:\s+[A-Z][a-z]+){0,2})$'  # Just a name (1-3 words, capitalized)
                    ]
                    for pattern in name_patterns:
                        name_match = re.search(pattern, user_text_clean, re.IGNORECASE)
                        if name_match:
                            name = name_match.group(1) if name_match.groups() else user_text_clean
                            # Only save if it looks like a real name (not "Yes", "No", "Great", etc.)
                            common_words = {'yes', 'no', 'ok', 'okay', 'sure', 'great', 'thanks', 'thank', 'you', 'hi', 'hello'}
                            if name.lower() not in common_words and len(name.split()) <= 3:
                                logger.info(f"🔍 Auto-detected name in user message: {name}")
                                await self.save_collected_data(None, "full_name", name)
                                break
        except Exception as e:
            # Don't let auto-detection errors break the conversation
            logger.debug(f"⚠️ Auto-detection error (non-critical): {e}")
    
    async def on_user_turn_completed(self, turn_ctx, new_message):
        """Handle user input and publish to frontend"""
        # CRITICAL: Call parent method FIRST to maintain conversation state
        # This ensures the agent's conversation history is preserved
        await super().on_user_turn_completed(turn_ctx, new_message)
        
        # Then do our side effects (non-blocking, fire-and-forget)
        user_text = new_message.text_content or ""
        if user_text.strip():
            # Publish user transcript to frontend (non-blocking)
            asyncio.create_task(self._publish_to_frontend("user_transcript", user_text, speaker="User"))
            # logger.info(f"USER: {user_text}")
        
            # Auto-detect and save name/email/phone if not already saved (fallback if LLM didn't call save_collected_data)
            asyncio.create_task(self._auto_detect_and_save_data(user_text))
        
            # Send user message to Alive5 (non-blocking to avoid interfering with agent processing)
            asyncio.create_task(self._send_message_to_alive5(user_text, is_agent=False))
    
    # Note: on_agent_speech_committed is NOT a valid method in LiveKit Agent class
    # According to https://docs.livekit.io/agents/build/text/, we should use
    # the conversation_item_added event on AgentSession to capture agent messages
    # This method is kept for backward compatibility but should not be called
    

# -----------------------------------------------------------------------------
# Entry Point
# -----------------------------------------------------------------------------
def prewarm(proc):
    """Preload VAD model - OPTIONAL based on USE_VAD env var"""
    # CRITICAL: Only preload VAD if explicitly enabled to save memory
    # VAD model uses ~100-200MB RAM, which is significant on a 1.9GB server
    use_vad = os.getenv("USE_VAD", "false").lower() == "true"  # Default to false to save memory
    
    if not use_vad:
        logger.info("🚫 VAD preloading skipped (USE_VAD=false) - saving ~100-200MB RAM")
        proc.userdata["vad"] = None
        return
    
    # Only preload if VAD is enabled
    try:
        logger.info("📦 Preloading VAD model (this uses ~100-200MB RAM)...")
        proc.userdata["vad"] = silero.VAD.load()
        logger.info("✅ VAD model preloaded successfully")
    except Exception as e:
        # If VAD loading fails or times out, continue without it
        logger.warning(f"⚠️ Could not preload VAD model: {e}. VAD will be loaded lazily if needed.")
        proc.userdata["vad"] = None

async def entrypoint(ctx: JobContext):
    """Main entry point for the simple agent"""
    try:
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
        faq_bot_id = user_data.get("faq_bot_id")  # FAQ bot ID for Bedrock filtering
        
        # Validate and log FAQ bot ID
        if not faq_bot_id or faq_bot_id.strip() == "":
            logger.warning(f"⚠️ FAQ bot ID is None or empty in session data, using default")
            faq_bot_id = "faq_b9952a56-fc7b-41c9-b0a0-5c662ddb039e"  # Default FAQ bot
        else:
            logger.info(f"✅ FAQ bot ID found in session data: {faq_bot_id}")
        
        faq_isVoice = user_data.get("faq_isVoice", True)  # Default to concise responses
        special_instructions = user_data.get("special_instructions", "")  # Load special instructions
        
        # # Log configuration for debugging
        # logger.info(f"📋 Session configuration loaded:")
        # logger.info(f"   - Botchain: {botchain_name}")
        # logger.info(f"   - Org Name: {org_name}")
        # logger.info(f"   - FAQ Bot ID: {faq_bot_id}")
        # logger.info(f"   - FAQ IsVoice: {faq_isVoice}")
        
        # Connect to the room first - using same approach as working implementation
        await ctx.connect(auto_subscribe=AutoSubscribe.SUBSCRIBE_ALL)
        
        # Get Alive5 session data from backend (frontend already called init_livechat)
        # CRITICAL: Frontend calls init_livechat and connects socket, so worker should use that session data
        # Do NOT call init_livechat again - it would create a different thread_id!
        async def get_alive5_session_data():
            """Get Alive5 session data from backend (created by frontend's init_livechat call)"""
            try:
                import httpx
                from urllib.parse import quote
                encoded_room_name = quote(room_name_clean, safe='')
                
                # Poll for session data (frontend may call init_livechat slightly after worker starts)
                max_attempts = 10
                for attempt in range(max_attempts):
                    async with httpx.AsyncClient(timeout=5.0) as client:
                        session_response = await client.get(f"{backend_url}/api/sessions/{encoded_room_name}")
                        if session_response.status_code == 200:
                            session_data = session_response.json()
                            thread_id = session_data.get("thread_id")
                            crm_id = session_data.get("crm_id")
                            channel_id = session_data.get("channel_id")
                            widget_id = session_data.get("widget_id")
                            
                            if thread_id and crm_id and channel_id:
                                # Set Alive5 session data in agent instance
                                agent.alive5_thread_id = thread_id
                                agent.alive5_crm_id = crm_id
                                agent.alive5_channel_id = channel_id
                                agent.alive5_widget_id = widget_id
                                
                                logger.info(f"✅ Alive5 session data loaded - Thread: {thread_id}, CRM: {crm_id}")
                                
                                # Process any queued messages
                                if agent._alive5_message_queue:
                                    queued_messages = agent._alive5_message_queue.copy()
                                    agent._alive5_message_queue.clear()
                                    for queued_content, queued_is_agent in queued_messages:
                                        await agent._send_message_to_alive5_internal(queued_content, queued_is_agent)
                                return
                            else:
                                logger.debug(f"⏳ Session data not ready yet (attempt {attempt + 1}/{max_attempts})")
                        else:
                            logger.debug(f"⏳ Session not found yet (attempt {attempt + 1}/{max_attempts})")
                    
                    # Wait before retrying (frontend may still be initializing)
                    await asyncio.sleep(0.5)
                
                logger.warning(f"⚠️ Could not get Alive5 session data after {max_attempts} attempts")
            except Exception as e:
                logger.warning(f"⚠️ Could not get Alive5 session data: {e}")
        
        # Get session data in background (non-blocking)
        # Frontend will call init_livechat, then we'll pick up the session data
        asyncio.create_task(get_alive5_session_data())
        if is_phone_call:
            logger.info(f"📞 Phone call detected - waiting for Alive5 session data")
        else:
            logger.info(f"🚀 Waiting for Alive5 session data (frontend will initialize)")
        
        # Create and start the agent (use cleaned room name)
        agent = SimpleVoiceAgent(room_name_clean, botchain_name, org_name, special_instructions)
        agent.faq_isVoice = faq_isVoice
        # Store FAQ bot ID and org_name on agent instance for easy access
        # IMPORTANT: Ensure faq_bot_id is not None before storing
        # logger.info(f"🔍 About to store FAQ bot ID: {faq_bot_id} (type: {type(faq_bot_id)})")
        if faq_bot_id and faq_bot_id.strip():
            agent.faq_bot_id = faq_bot_id
            # logger.info(f"💾 Stored FAQ bot ID on agent instance: {agent.faq_bot_id}")
            # logger.info(f"   - Verification: hasattr(agent, 'faq_bot_id') = {hasattr(agent, 'faq_bot_id')}")
            # logger.info(f"   - Verification: agent.faq_bot_id = {getattr(agent, 'faq_bot_id', 'NOT_SET')}")
        else:
            logger.warning(f"⚠️ FAQ bot ID is None or empty (value: {faq_bot_id}), not storing on agent instance")
            agent.faq_bot_id = None  # Explicitly set to None so we know it's missing
        agent.org_name = org_name  # Store org_name on agent for Bedrock filtering
        # logger.info(f"💾 Stored org_name on agent instance: {agent.org_name}")
        
        # Get VAD - with environment variable control for testing
        # CRITICAL: VAD can cause resource exhaustion (CPU/memory/disk)
        # If server becomes unresponsive, disable VAD by setting USE_VAD=false
        vad = None
        use_vad = os.getenv("USE_VAD", "true").lower() == "true"
        
        # Phone calls may benefit from different VAD settings
        if is_phone_call:
            phone_use_vad = os.getenv("PHONE_USE_VAD", "true").lower() == "true"
            if not phone_use_vad:
                use_vad = False
                logger.info("📞 Phone call detected - VAD disabled (using turn detection instead)")
        
        if use_vad:
            # Try to reuse VAD from prewarm to avoid loading multiple times
            # This prevents memory exhaustion from multiple VAD instances
            if "vad" in ctx.proc.userdata and ctx.proc.userdata["vad"] is not None:
                vad = ctx.proc.userdata["vad"]
                logger.info("✅ VAD loaded from prewarm (shared instance)")
            else:
                # Only load if not in prewarm - this should be rare
                try:
                    logger.warning("⚠️ VAD not in prewarm, loading lazily (may cause resource issues)")
                    vad = silero.VAD.load()
                    logger.info("✅ VAD loaded successfully (lazy load)")
                except Exception as e:
                    logger.warning(f"⚠️ Failed to load VAD: {e}, continuing without VAD")
                    logger.warning("   Consider disabling VAD if server becomes unresponsive")
        else:
            logger.info("🚫 VAD disabled (USE_VAD=false)")
        
        # Set room on agent for frontend communication
        agent.room = ctx.room
        
        # Initialize TTS (skip for Nova Sonic as it's speech-to-speech)
        tts = None
        if not getattr(agent, '_using_nova', False):
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
        else:
            logger.info("🎙️ Skipping TTS initialization (Nova Sonic handles TTS internally)")
        
        # Optimize STT model for phone calls - use faster model if configured
        # Skip STT initialization for Nova Sonic (it's speech-to-speech)
        stt_model = None
        stt_language = "en-US"  # Default to English
        if not getattr(agent, '_using_nova', False):
            stt_model = os.getenv("DEEPGRAM_STT_MODEL", "nova-2")
            if is_phone_call:
                # For phone calls, check if we should use a faster STT model
                phone_stt_model = os.getenv("PHONE_DEEPGRAM_STT_MODEL", stt_model)
                if phone_stt_model != stt_model:
                    stt_model = phone_stt_model
                    logger.info(f"📞 Phone call detected - using optimized STT model: {stt_model}")
        else:
            logger.info("🎙️ Skipping STT initialization (Nova Sonic handles STT internally)")
        
        # Initialize turn detector (as per LiveKit docs: https://docs.livekit.io/agents/build/turns/turn-detector/)
        # The turn detector improves end-of-turn detection by using conversational context
        turn_detector = None
        if not getattr(agent, '_using_nova', False) and stt_model:
            # Check if turn detector is enabled (default: true)
            use_turn_detector = os.getenv("USE_TURN_DETECTOR", "true").lower() == "true"
            if use_turn_detector:
                # Choose model based on language support
                turn_detector_type = os.getenv("TURN_DETECTOR_MODEL", "english").lower()  # "english" or "multilingual"
                try:
                    # Initialize turn detector with timeout protection
                    # If model files aren't downloaded, this will fail gracefully
                    if turn_detector_type == "multilingual":
                        turn_detector = MultilingualModel()
                        stt_language = "multi"  # Multilingual requires "multi" language setting
                        logger.info("🌍 Using LiveKit Turn Detector (Multilingual Model)")
                    else:
                        turn_detector = EnglishModel()
                        logger.info("🇺🇸 Using LiveKit Turn Detector (English Model)")
                except Exception as e:
                    # If turn detector fails to initialize (e.g., model files not downloaded), continue without it
                    logger.warning(f"⚠️ Turn detector initialization failed: {e}")
                    logger.warning("⚠️ Continuing without turn detector - end-of-turn detection may be less accurate")
                    logger.warning("⚠️ To fix: Run 'python backend/alive5-worker/worker.py download-files' to download model files")
                    turn_detector = None
            else:
                logger.info("🚫 Turn detector disabled")
        
        # Get model name for logging
        model_name_for_log = None
        if hasattr(agent.llm, '_opts') and hasattr(agent.llm._opts, 'model'):
            model_name_for_log = agent.llm._opts.model
        elif hasattr(agent.llm, 'model'):
            model_name_for_log = agent.llm.model
        
        # Initialize AgentSession based on LLM provider
        # Nova Sonic is a speech-to-speech model, so it doesn't need separate STT/TTS
        if getattr(agent, '_using_nova', False):
            logger.info("🎙️ Using Nova Sonic (speech-to-speech) - no separate STT/TTS needed")
            session = AgentSession(
                llm=agent.llm,  # Nova Sonic RealtimeModel (handles STT, LLM, and TTS)
            )
        else:
            # Traditional setup with separate STT, LLM, and TTS
            session = AgentSession(
                stt=deepgram.STT(model=stt_model, language=stt_language, api_key=os.getenv("DEEPGRAM_API_KEY")) if stt_model else None,
                llm=agent.llm,  # LLM instance (OpenAI, Bedrock, etc.)
                tts=tts,
                vad=vad,
                turn_detection=turn_detector  # Use LiveKit turn detector for better end-of-turn detection
            )
        
        # Set the agent's room and session
        agent.room = ctx.room
        agent.agent_session = session
        
        # Start the session - with environment variable control for testing
        # For phone calls, noise cancellation can add latency - optimize based on session type
        use_noise_cancellation = os.getenv("USE_NOISE_CANCELLATION", "true").lower() == "true"
        
        # Phone calls may benefit from disabling noise cancellation to reduce latency
        # Web sessions typically have better audio quality and can handle noise cancellation better
        if is_phone_call:
            # For phone calls, check if we should disable noise cancellation for better latency
            phone_noise_cancellation = os.getenv("PHONE_USE_NOISE_CANCELLATION", "false").lower() == "true"
            if not phone_noise_cancellation:
                use_noise_cancellation = False
                logger.info("📞 Phone call detected - noise cancellation disabled for lower latency")
        
        room_input_options = RoomInputOptions(text_enabled=True)
        if use_noise_cancellation:
            room_input_options.noise_cancellation = noise_cancellation.BVC()
            logger.info("🔇 Noise cancellation enabled")
        else:
            logger.info("🚫 Noise cancellation disabled")
        
        await session.start(
            room=ctx.room,
            agent=agent,
            room_input_options=room_input_options,
            room_output_options=RoomOutputOptions(
                transcription_enabled=True,
                sync_transcription=True  # Enable sync transcription for frontend display
            )
        )
        
        # Listen to conversation item added event (as per LiveKit docs)
        # According to https://docs.livekit.io/agents/build/text/, this event fires when
        # "text input or output is committed to the chat history"
        # The event is ConversationItemAddedEvent which has an 'item' property (ChatMessage)
        @session.on("conversation_item_added")
        def on_conversation_item_added(event):
            """Captures all conversation items (user and agent messages) as they're added to history"""
            try:
                # The event is ConversationItemAddedEvent, which has an 'item' property containing the ChatMessage
                # See: livekit/agents/voice/events.py - ConversationItemAddedEvent(item: ChatMessage)
                chat_message = event.item if hasattr(event, 'item') else event
                
                # ChatMessage has 'role' and 'content' properties
                # role can be 'user', 'assistant', 'system', etc.
                is_agent_msg = hasattr(chat_message, 'role') and chat_message.role == 'assistant'
                
                # Extract text content from ChatMessage
                # ChatMessage.content can be str or list of content parts
                text_content = ""
                if hasattr(chat_message, 'content'):
                    if isinstance(chat_message.content, str):
                        text_content = chat_message.content
                    elif isinstance(chat_message.content, list):
                        # Join all text parts (handles both dict and str parts)
                        text_content = ' '.join(
                            part.get('text', '') if isinstance(part, dict) else str(part)
                            for part in chat_message.content
                        )
                
                if text_content and text_content.strip():
                    # Only send agent messages here (user messages are handled in on_user_turn_completed)
                    if is_agent_msg:
                        # Silently send agent message (logging happens in _send_message_to_alive5_internal)
                        asyncio.create_task(agent._send_message_to_alive5(text_content, is_agent=True))
                else:
                    logger.warning(f"⚠️ conversation_item_added called but text_content is empty (role: {getattr(chat_message, 'role', 'unknown')})")
            except Exception as e:
                logger.warning(f"⚠️ Error in conversation_item_added (isolated): {e}", exc_info=True)
        
        # Start the conversation with greeting
        # Note: For Nova Sonic, greeting is skipped (waits for user input first)
        # Check if room is still connected before starting greeting
        try:
            if ctx.room and hasattr(ctx.room, "name"):
                await agent.on_room_enter(ctx.room)
            else:
                logger.warning("⚠️ Room disconnected before greeting could start")
        except RuntimeError as e:
            if "closing" in str(e).lower():
                logger.info("ℹ️ Session closing during startup, skipping greeting")
            else:
                raise
        
        logger.info("✅ Simple agent started successfully")
        logger.info("=" * 80)
        logger.info(f"🎯 SESSION READY - Room: {room_name_clean} | Botchain: {botchain_name} | Org: {org_name} | Model: {model_name_for_log}")
        logger.info("=" * 80)
    except Exception as e:
        logger.error(f"❌ CRITICAL ERROR in entrypoint: {e}")
        logger.error(f"   Room: {ctx.room.name if ctx and ctx.room else 'Unknown'}")
        import traceback
        logger.error(f"   Traceback: {traceback.format_exc()}")
        raise  # Re-raise to let LiveKit know the job failed

if __name__ == "__main__":
    import signal
    import time
    import traceback
    
    # Track restart attempts
    restart_count = 0
    max_restarts = 10
    restart_delay = 5  # Start with 5 seconds
    
    def signal_handler(signum, frame):
        """Handle shutdown signals gracefully"""
        logger.info("=" * 80)
        logger.info(f"🛑 Received signal {signum}, shutting down gracefully...")
        logger.info("=" * 80)
        sys.exit(0)
    
    # Register signal handlers for graceful shutdown
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    
    while restart_count < max_restarts:
        try:
            logger.info("=" * 80)
            logger.info("🚀 Starting LiveKit Worker...")
            if restart_count > 0:
                logger.info(f"🔄 Restart attempt {restart_count}/{max_restarts}")
            logger.info("=" * 80)
            
            # Verify environment variables are loaded
            livekit_url = os.getenv("LIVEKIT_URL")
            livekit_api_key = os.getenv("LIVEKIT_API_KEY")
            livekit_api_secret = os.getenv("LIVEKIT_API_SECRET")
            
            if not livekit_url:
                logger.error("❌ LIVEKIT_URL not set in environment!")
                sys.exit(1)
            if not livekit_api_key:
                logger.error("❌ LIVEKIT_API_KEY not set in environment!")
                sys.exit(1)
            if not livekit_api_secret:
                logger.error("❌ LIVEKIT_API_SECRET not set in environment!")
                sys.exit(1)
            
            logger.info(f"✅ LiveKit URL: {livekit_url}")
            logger.info(f"✅ LiveKit API Key: {livekit_api_key[:10]}...")
            logger.info("=" * 80)
            logger.info("🔌 Connecting to LiveKit server...")
            logger.info("=" * 80)
            
            # Log system resources before starting
            try:
                import psutil
                memory = psutil.virtual_memory()
                cpu_percent = psutil.cpu_percent(interval=1)
                logger.info(f"💻 System Resources - RAM: {memory.percent:.1f}% used ({memory.used / 1024**3:.2f}GB / {memory.total / 1024**3:.2f}GB), CPU: {cpu_percent:.1f}%")
            except Exception as e:
                logger.debug(f"Could not get system resources: {e}")
            
            # Increase initialization timeout to 30 seconds to allow for VAD model loading
            # Add connection timeout and retry settings
            cli.run_app(WorkerOptions(
                entrypoint_fnc=entrypoint, 
                prewarm_fnc=prewarm,
                initialize_process_timeout=30.0,  # Increase from default 10.0 to 30.0 seconds
                # Note: WorkerOptions doesn't expose connection retry settings directly,
                # but the LiveKit SDK handles reconnection automatically
            ))
            
            # If we reach here, the worker exited normally
            logger.info("✅ Worker exited normally")
            break
            
        except KeyboardInterrupt:
            logger.info("🛑 Worker stopped by user")
            break
            
        except Exception as e:
            restart_count += 1
            error_msg = str(e)
            error_type = type(e).__name__
            
            logger.error("=" * 80)
            logger.error(f"❌ Worker crashed with {error_type}: {error_msg}")
            logger.error("=" * 80)
            
            # Log full traceback for debugging
            logger.error(f"📋 Full traceback:")
            logger.error(traceback.format_exc())
            
            # Check if it's a connection error
            if "connection" in error_msg.lower() or "closed" in error_msg.lower() or "unexpectedly" in error_msg.lower():
                logger.warning("⚠️ Connection error detected - this may be due to:")
                logger.warning("   - Network connectivity issues")
                logger.warning("   - LiveKit server restarting")
                logger.warning("   - Firewall or security group blocking connection")
                logger.warning("   - DNS resolution issues")
            
            # Check if it's a memory error
            if "memory" in error_msg.lower() or "killed" in error_msg.lower() or "oom" in error_msg.lower():
                logger.warning("⚠️ Memory error detected - this may be due to:")
                logger.warning("   - Insufficient RAM on server")
                logger.warning("   - Memory leak in worker process")
                logger.warning("   - Too many concurrent sessions")
                try:
                    import psutil
                    memory = psutil.virtual_memory()
                    logger.warning(f"   Current RAM usage: {memory.percent:.1f}% ({memory.used / 1024**3:.2f}GB / {memory.total / 1024**3:.2f}GB)")
                except:
                    pass
            
            if restart_count < max_restarts:
                # Exponential backoff: 5s, 10s, 20s, 30s, then cap at 60s
                restart_delay = min(restart_delay * 2, 60)
                logger.warning(f"🔄 Restarting worker in {restart_delay} seconds... (attempt {restart_count}/{max_restarts})")
                logger.warning("=" * 80)
                time.sleep(restart_delay)
            else:
                logger.error("=" * 80)
                logger.error(f"❌ Max restart attempts ({max_restarts}) reached. Worker will not restart automatically.")
                logger.error("   Please check logs and fix the underlying issue.")
                logger.error("   The systemd service will restart the worker after RestartSec.")
                logger.error("=" * 80)
                # Exit with error code so systemd knows to restart
                sys.exit(1)
    
    # If we exhausted all restart attempts, exit
    if restart_count >= max_restarts:
        logger.error("❌ Worker failed to start after multiple attempts. Exiting.")
        sys.exit(1)