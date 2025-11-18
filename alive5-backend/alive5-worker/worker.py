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
        self.alive5_auth_token = None
        self.alive5_widget_id = None
        self.alive5_org_name = org_name
        self.alive5_botchain = botchain_name
        self._alive5_message_count = 0  # Track message count for is_new flag
        self._alive5_message_queue = []  # Queue messages until session data is ready
        
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
    async def load_bot_flows(self, context: RunContext, botchain_name: str, org_name: str) -> Dict[str, Any]:
        """Load bot flow definitions dynamically. MUST be called ONCE at startup before first user interaction.
        
        CRITICAL RULES:
        - This function MUST ONLY be called ONCE at the very beginning of the conversation
        - DO NOT call this function again during the conversation - flows are already loaded
        - The agent must wait for flows to load before speaking. Do not speak until this function completes.
        - After calling this function once, the flows are cached in memory for the entire conversation
        
        Args:
            botchain_name: The botchain name (e.g., 'voice-1')
            org_name: The organization name
            
        Returns:
            Dict with 'success', 'data', 'template', and 'intents' keys
        """
        # Prevent multiple calls - if flows are already loaded, return cached result
        if self._flows_loaded and self.bot_template is not None:
            logger.warning(f"⚠️ load_bot_flows() called again - flows already loaded. Returning cached result.")
            return {
                "success": True,
                "data": self.bot_template,
                "template": self.bot_template,
                "intents": list(self.bot_template.get("data", {}).keys()) if self.bot_template.get("data") else [],
                "note": "Flows were already loaded - this is a cached result"
            }
        
        # Load flows
        result = await handle_load_bot_flows(botchain_name, org_name)
        
        # Cache the result if successful
        if result.get("success") and result.get("data"):
            self.bot_template = result.get("data")
            self._flows_loaded = True
        
        return result
    
    
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
    
    # COMMENTED OUT: CRM data collection - will be patched later
    # @function_tool()
    # async def save_collected_data(self, context: RunContext, field_name: str, value: str) -> Dict[str, Any]:
    #     """Save user response to collected_data based on save_data_to field from flow.
    #     
    #     Call this function whenever a flow question has a 'save_data_to' field and the user provides an answer.
    #     
    #     Args:
    #         field_name: The field name from save_data_to (e.g., "full_name", "email", "phone", "notes_entry")
    #         value: The user's response value to save
    #     """
    #     try:
    #         if field_name == "full_name":
    #             self.collected_data["full_name"] = value
    #             logger.info(f"💾 Saved full_name: {value}")
    #         elif field_name == "email":
    #             self.collected_data["email"] = value
    #             logger.info(f"💾 Saved email: {value}")
    #         elif field_name == "phone":
    #             self.collected_data["phone"] = value
    #             logger.info(f"💾 Saved phone: {value}")
    #         elif field_name == "notes_entry":
    #             if "notes_entry" not in self.collected_data:
    #                 self.collected_data["notes_entry"] = []
    #             self.collected_data["notes_entry"].append(value)
    #             logger.info(f"💾 Appended to notes_entry: {value}")
    #         else:
    #             logger.warning(f"⚠️ Unknown field_name: {field_name}")
    #             return {"success": False, "message": f"Unknown field: {field_name}"}
    #         
    #         return {
    #             "success": True,
    #             "message": f"Saved {field_name} successfully"
    #         }
    #     except Exception as e:
    #         logger.error(f"❌ Error saving collected data: {e}")
    #         return {"success": False, "message": f"Error saving data: {str(e)}"}
    
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
        logger.info(f"🔍 FAQ bot ID check - LLM provided bot_id: {bot_id}, type: {type(bot_id)}")
        
        # CRITICAL: Check agent instance attribute directly - this is the source of truth
        stored_faq_bot_id = getattr(self, 'faq_bot_id', None)
        logger.info(f"🔍 FAQ bot ID check - stored value on agent: {stored_faq_bot_id}, type: {type(stored_faq_bot_id)}")
        
        # ALWAYS use stored FAQ bot ID from agent instance if available (ignore LLM-provided bot_id)
        if stored_faq_bot_id and stored_faq_bot_id.strip():
            bot_id = stored_faq_bot_id
            logger.info(f"🤖 Using FAQ bot ID from agent instance: {bot_id}")
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
        logger.info(f"🔍 FAQ request - Bot ID: {bot_id}, Org Name: {org_name}")
        
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
            logger.info(f"🤖 Using FAQ bot from agent instance: {self.faq_bot_id}")
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
                        logger.info(f"🤖 Using FAQ bot from session data: {faq_bot_id}")
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
        # Nova Sonic doesn't support unprompted generation - it requires user input first
        if getattr(self, '_using_nova', False):
            logger.info("🎙️ Nova Sonic detected - skipping proactive greeting (waits for user input first)")
            return  # Nova Sonic will respond naturally when user speaks
        
        try:
            # Use generate_reply to make the agent speak first
            if hasattr(self, "agent_session") and self.agent_session:
                await self.agent_session.generate_reply()
            else:
                logger.warning("⚠️ Agent session not available")
            
        except Exception as e:
            # Handle Nova Sonic "unprompted generation" error gracefully
            if "unprompted generation" in str(e).lower() or "realtime" in str(e).lower():
                logger.info("🎙️ Nova Sonic requires user input first - this is expected behavior")
                return
            
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
    
    async def _send_message_to_alive5(self, message_content: str, is_agent: bool = False):
        """Send a message to Alive5 livechat via Socket.io
        
        This function is completely isolated from agent processing - any errors here
        will NOT affect the agent's conversation flow or context.
        """
        try:
            # If session data is not ready, queue the message
            if not self.alive5_thread_id or not self.alive5_channel_id or not self.alive5_auth_token:
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
    
    async def _ensure_alive5_connection(self):
        """Create a new Socket.io connection to Alive5 for each message
        
        Note: Alive5 server disconnects after processing each message, so we create
        a new connection for each message. This is expected behavior.
        """
        import socketio
        sio = socketio.AsyncClient()
        
        # Connect using the same pattern as backend (query params in auth dict, not path)
        try:
            await sio.connect(
                'wss://api-v2-stage.alive5.com',
                transports=['websocket'],
                socketio_path='/socket.io',
                wait_timeout=10,
                auth={
                    'authToken': self.alive5_auth_token,
                    'thread_id': self.alive5_thread_id,
                    'crm_id': self.alive5_crm_id or '',
                    'channel_id': self.alive5_channel_id,
                    'is_mobile': 'false',
                    'EIO': '4',
                    'transport': 'websocket'
                }
            )
            return sio
        except Exception as e:
            logger.error(f"❌ Failed to connect to Alive5 Socket.io: {e}")
            raise
    
    async def _send_message_to_alive5_internal(self, message_content: str, is_agent: bool = False):
        """Internal function to actually send message to Alive5"""
        try:
            from datetime import datetime
            
            # Create new connection for each message (server disconnects after processing)
            sio = await self._ensure_alive5_connection()
            
            # Track if this is the first message (for newThread object)
            is_first_message = self._alive5_message_count == 0
            self._alive5_message_count += 1
            is_new = is_first_message
            
            # Build message payload matching Alive5 format
            message_data = {
                "channel_id": self.alive5_channel_id,
                "event_mode": "redis",
                "message_content": message_content,
                "message_type": "livechat",
                "message_data": "",
                "org_name": self.alive5_org_name,
                "route": "123",
                "thread_id": self.alive5_thread_id,
                "is_new": is_new,
                "crm_id": self.alive5_crm_id or "",
                "session_id": "",
                "attach_botchain": self.alive5_botchain,
                "webpage_title": "Voice Agent",
                "webpage_url": "",
                "old_channel_id": "",
                "old_agent_id": "",
                "old_thread_id": "",
                "assignedTo": "",
                "transfer_type": "",
                "transferred_agent": "",
                "user_interacted": "additional_action" if not is_agent else "bot_initiated",
                "widget_id": self.alive5_widget_id,
                "browsing": ["master", "agent"],
                "query_string": {
                    "first_name": "",
                    "last_name": "",
                    "email": "",
                    "notes": "",
                    "crm_id": "",
                    "agent_email": "",
                    "accountid": "",
                    "company": "",
                    "companytitle": "",
                    "phone_mobile": "",
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
            }
            
            # Add user_id for agent messages
            if is_agent:
                message_data["user_id"] = "system"
                timestamp = int(datetime.now().timestamp() * 1000)
                message_data["message_id"] = f"msg-{timestamp}"
                message_data["timestamp"] = timestamp
                message_data["created_at"] = timestamp
                message_data["updated_at"] = timestamp
            
            # Include newThread object for first message (required to create conversation in dashboard)
            if is_new:
                from datetime import datetime
                timestamp = int(datetime.now().timestamp() * 1000)
                
                # Build newThread object as per Alive5 API specification
                # This is required for the first message to create a visible conversation
                message_data["newThread"] = {
                    "assignedTo": "",
                    "botchain_label": self.alive5_botchain or "",
                    "channel_id": self.alive5_channel_id,
                    "connect_botchain": self.alive5_botchain or "",
                    "connect_orgbot": "",  # Can be left empty
                    "created_at": timestamp,
                    "crm_id": self.alive5_crm_id or "",
                    "lastmessage_at": timestamp,
                    "org_name": self.alive5_org_name,
                    "status_timestamp": f"open||{timestamp}",
                    "thread_session": "{}",  # Empty JSON object as string
                    "thread_start_chat": timestamp,
                    "thread_status": "chatting",
                    "thread_type": "livechat",
                    "time_ping": timestamp,
                    "timestamp": timestamp,
                    "transaction_id": "",  # Can be left empty
                    "updated_at": timestamp,
                    "viewed_by": [],
                    "widget_id": self.alive5_widget_id or "",
                    "thread_id": self.alive5_thread_id or "",
                    "crmData": {
                        "allow_zapier_syns": True,
                        "assigned_user": [],
                        "created_at": timestamp,
                        "crm_id": self.alive5_crm_id or "",
                        "crm_thread_type": "livechat",
                        "crm_type": "livechat",
                        "org_name": self.alive5_org_name,
                        "updated_at": timestamp
                    },
                    "tempMessageId": f"uid-{timestamp}"
                }
            
            # Only log agent messages to reduce verbosity (user messages are already logged in on_user_turn_completed)
            if is_agent:
                logger.info(f"📤 Agent: {message_content[:80]}...")
            await sio.emit('livechat-message', message_data)
            # Note: Alive5 server disconnects after processing messages, so we'll reconnect on next message
            # This is expected behavior - don't try to keep connection alive
            # Only log success for agent messages to reduce verbosity
            # if is_agent:
                # logger.info(f"✅ Agent message sent to Alive5")
            
            # Disconnect after sending (server will disconnect anyway)
            try:
                await sio.disconnect()
            except Exception:
                pass  # Ignore disconnect errors
            
        except Exception as e:
            logger.error(f"❌ Could not send message to Alive5: {e}", exc_info=True)
    
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
            logger.info(f"USER: {user_text}")
        
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
    """Preload VAD model - using same approach as working implementation"""
    proc.userdata["vad"] = silero.VAD.load()

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
        
        # Initialize livechat session with Socket.io (required for CRM data submission)
        # CRITICAL: Start this in background to avoid blocking agent startup
        # The agent can start speaking immediately while livechat initializes
        # NOTE: This is needed for BOTH web sessions AND phone calls to enable CRM data saving
        async def init_livechat_background():
            """Initialize livechat in background and set session data on agent"""
            try:
                import httpx
                from urllib.parse import quote
                async with httpx.AsyncClient(timeout=30.0) as client:
                    response = await client.post(
                        f"{backend_url}/api/init_livechat",
                        params={
                            "room_name": room_name_clean,
                            "org_name": org_name,
                            "botchain_name": botchain_name
                        }
                    )
                    if response.status_code == 200:
                        init_result = response.json()
                        thread_id = init_result.get('thread_id')
                        crm_id = init_result.get('crm_id')
                        logger.info(f"✅ Livechat initialized (background) - Thread: {thread_id}, CRM: {crm_id}")
                        
                        # Get session data to retrieve Alive5 connection info
                        try:
                            encoded_room_name = quote(room_name_clean, safe='')
                            async with httpx.AsyncClient() as session_client:
                                session_response = await session_client.get(f"{backend_url}/api/sessions/{encoded_room_name}")
                                if session_response.status_code == 200:
                                    session_data = session_response.json()
                                    # Set Alive5 session data in agent instance
                                    agent.alive5_thread_id = thread_id
                                    agent.alive5_crm_id = crm_id
                                    agent.alive5_channel_id = session_data.get("channel_id")
                                    agent.alive5_auth_token = session_data.get("auth_token")
                                    agent.alive5_widget_id = session_data.get("widget_id")
                                    logger.info(f"✅ Alive5 session data set - ready to send messages")
                                    
                                    # Process any queued messages
                                    if agent._alive5_message_queue:
                                        logger.info(f"📬 Processing {len(agent._alive5_message_queue)} queued messages...")
                                        queued_messages = agent._alive5_message_queue.copy()
                                        agent._alive5_message_queue.clear()
                                        for queued_content, queued_is_agent in queued_messages:
                                            await agent._send_message_to_alive5_internal(queued_content, queued_is_agent)
                                else:
                                    logger.warning(f"⚠️ Could not get session data: {session_response.status_code}")
                        except Exception as e:
                            logger.warning(f"⚠️ Could not set Alive5 session data: {e}")
                    else:
                        logger.warning(f"⚠️ Livechat init failed: {response.status_code}")
            except Exception as e:
                logger.warning(f"⚠️ Could not initialize livechat (background): {e}")
        
        # Start livechat initialization in background (non-blocking)
        # This is required for CRM data submission to work (both web and phone calls)
        asyncio.create_task(init_livechat_background())
        if is_phone_call:
            logger.info(f"📞 Phone call detected - initializing livechat in background for CRM support")
        else:
            logger.info(f"🚀 Started livechat initialization in background (agent will start immediately)")
        
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
        # For phone calls, VAD can sometimes be less reliable due to network latency
        # Consider using turn_detection instead for phone calls if VAD causes issues
        vad = None
        use_vad = os.getenv("USE_VAD", "true").lower() == "true"
        
        # Phone calls may benefit from different VAD settings
        if is_phone_call:
            phone_use_vad = os.getenv("PHONE_USE_VAD", "true").lower() == "true"
            if not phone_use_vad:
                use_vad = False
                logger.info("📞 Phone call detected - VAD disabled (using turn detection instead)")
        
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
            logger.info("🚫 VAD disabled")
        
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
                if turn_detector_type == "multilingual":
                    turn_detector = MultilingualModel()
                    stt_language = "multi"  # Multilingual requires "multi" language setting
                    logger.info("🌍 Using LiveKit Turn Detector (Multilingual Model)")
                else:
                    turn_detector = EnglishModel()
                    logger.info("🇺🇸 Using LiveKit Turn Detector (English Model)")
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
        await agent.on_room_enter(ctx.room)
        
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
    logger.info("=" * 80)
    logger.info("🚀 Starting LiveKit Worker...")
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
    
    cli.run_app(WorkerOptions(entrypoint_fnc=entrypoint, prewarm_fnc=prewarm))