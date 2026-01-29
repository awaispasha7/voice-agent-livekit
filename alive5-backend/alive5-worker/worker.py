"""
Voice Agent - Single LLM with Function Calling (Brand-Agnostic)
"""

import asyncio
import logging
import os
import random
import sys
from pathlib import Path
from typing import Dict, Any, Optional, List

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
from livekit.plugins import openai, deepgram, cartesia, silero, aws
from livekit.plugins.turn_detector.english import EnglishModel
from livekit.plugins.turn_detector.multilingual import MultilingualModel

from system_prompt import get_system_prompt
from functions import handle_load_bot_flows, handle_faq_bot_request, handle_bedrock_knowledge_base_request
from tags_config import get_available_tags

# Import AgentCore integration
try:
    from agentcore_integration import AgentCoreIntegration
    from agentcore.memory import AgentCoreMemory
    from agentcore.gateway_tools import AgentCoreGateway
    from agentcore_llm_wrapper import AgentCoreLLMWrapper
    AGENTCORE_INTEGRATION_AVAILABLE = True
except ImportError:
    AGENTCORE_INTEGRATION_AVAILABLE = False
    AgentCoreLLMWrapper = None
    logging.getLogger("simple-agent").warning("AgentCore integration not available. Will use direct LLM and functions.")

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
            "first_name": None,
            "last_name": None,
            "email": None,
            "phone": None,
            "account_id": None,
            "company": None,
            "company_title": None,
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
        self._pending_crm_updates = []  # Queue CRM updates until thread_id/crm_id are ready

        # Alive5 Socket.IO (Option B: server-side bridge for PSTN calls where no frontend exists)
        self._alive5_socket = None
        self._alive5_socket_connected = False
        self._alive5_socket_connect_lock = asyncio.Lock()
        self._alive5_socket_init_event = asyncio.Event()
        self.alive5_voice_agent_id = None
        
        # Conversation transcript for post-call reconciliation (fail-safe capture).
        # We keep this in-memory per session and run a background extraction at call end.
        self._conversation_log: List[Dict[str, str]] = []
        # Cooldown to prevent repeated mid-call CRM verification spam
        self._last_crm_verify_at: Dict[str, float] = {}
        # Track who ended the session (used for end_voice_chat payload alignment with web)
        self._end_by: str = "voice_agent"
        
        # HITL (Human-in-the-Loop) handoff state
        self._handoff_active = False
        self._handoff_queue = None
        self._human_agent_identity = None
        self._handoff_monitor_task: asyncio.Task | None = None
        self._session_end_started = False  # Guard against duplicate end events

        # Silence nudges (human-like check-ins when user stays silent after a question)
        self._silence_nudge_task: asyncio.Task | None = None
        self._silence_nudge_question: str | None = None
        self._silence_nudge_snooze_until: float = 0.0
        
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
        
        # Initialize AgentCore integration if available
        self.agentcore_integration = None
        self.agentcore_memory = None
        self.agentcore_gateway = None
        
        if AGENTCORE_INTEGRATION_AVAILABLE:
            try:
                self.agentcore_integration = AgentCoreIntegration()
                self.agentcore_memory = AgentCoreMemory()
                self.agentcore_gateway = AgentCoreGateway()
                
                if self.agentcore_integration.is_enabled():
                    logger.info("✅ AgentCore integration enabled - will use AgentCore Runtime for LLM")
                if self.agentcore_memory and self.agentcore_memory.is_enabled():
                    logger.info("✅ AgentCore Memory enabled - will use persistent storage")
                if self.agentcore_gateway and self.agentcore_gateway.is_enabled():
                    logger.info("✅ AgentCore Gateway enabled - will use Gateway for functions")
            except Exception as e:
                logger.warning(f"⚠️ Failed to initialize AgentCore integration: {e}")
                self.agentcore_integration = None
        
        # Wrap LLM with AgentCore wrapper if enabled
        if self.agentcore_integration and self.agentcore_integration.is_enabled() and AgentCoreLLMWrapper:
            llm_instance = AgentCoreLLMWrapper(
                base_llm=llm_instance,
                agentcore_integration=self.agentcore_integration,
                session_id=room_name,
                botchain_name=botchain_name,
                org_name=org_name
            )
            logger.info("✅ LLM wrapped with AgentCore - calls will route to AgentCore Runtime")
        
        # Initialize the base Agent class with special instructions
        system_prompt = get_system_prompt(botchain_name, org_name, special_instructions)
        super().__init__(instructions=system_prompt, llm=llm_instance)
        
    def _backend_internal_url(self) -> str:
        """Prefer an internal backend URL for server-to-server calls to avoid TLS/proxy issues."""
        return (os.getenv("BACKEND_URL_INTERNAL") or "http://127.0.0.1:8000").strip()

    
    
    @function_tool()
    async def transfer_call_to_human(self, context: RunContext, transfer_number: Optional[str] = None) -> Dict[str, Any]:
        """Transfer the current phone call to a human agent or phone number.
        
        IMPORTANT:
        - For **web sessions**, this triggers **HITL (dashboard) handoff** and returns `is_hitl: true`.
        - For **phone calls**, it can either trigger HITL (if PHONE_USE_HITL=true) or do a Telnyx transfer.
        
        If AgentCore Gateway is enabled, this will route through Gateway.
        Otherwise, uses direct function call.
        
        Args:
            transfer_number: Phone number to transfer to (e.g., "+18555518858"). 
                           Optional - if not provided, uses default call center number from environment.
                           If no transfer number is configured, returns helpful message.
        
        Returns:
            A status object. For HITL handoff, `success: true` and `is_hitl: true`.
        """
        # Try Gateway first if enabled
        if self.agentcore_gateway and self.agentcore_gateway.is_enabled():
            try:
                result = await self.agentcore_gateway.call_tool(
                    "transfer_call_to_human",
                    room_name=self.room_name,
                    transfer_number=transfer_number
                )
                if result.get("success") is not None:
                    return result
            except Exception as e:
                logger.warning(f"Gateway call failed, falling back to direct: {e}")
        
        # Fallback to direct function call
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
            
            # Ensure room name doesn't have known Telnyx prefix variants (clean it up)
            room_name_clean = self.room_name
            if room_name_clean.startswith("telnyx_call__telnyx_call_"):
                # Remove double prefix
                room_name_clean = room_name_clean.replace("telnyx_call__telnyx_call_", "telnyx_call_", 1)
                logger.warning(f"⚠️ Fixed double-prefixed room name: {self.room_name} -> {room_name_clean}")
            elif room_name_clean.startswith("telnyx_call__"):
                # Historical variant caused by configuring room_prefix with a trailing "_" while LiveKit SIP
                # also inserts "_" between prefix and callee (resulting in telnyx_call__<hash>).
                room_name_clean = room_name_clean.replace("telnyx_call__", "telnyx_call_", 1)
                logger.warning(f"⚠️ Fixed double-underscore room name: {self.room_name} -> {room_name_clean}")
            
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
                    is_phone_call = call_control_id and source == "telnyx_phone"
                    
                    if not is_phone_call:
                        # Web session - always use HITL
                        logger.info("ℹ️ Transfer requested for web session - initiating HITL handoff")
                        
                        # Use default queue (agent node answer not available in tool context)
                        queue = "general"
                        
                        # Initiate HITL handoff for web session
                        await self._initiate_human_handoff(queue=queue, trigger="transfer_tool")
                        
                        return {
                            "success": True,
                            "is_web_session": True,
                            "is_hitl": True,
                            "message": "Certainly! Let me connect you with a live agent. One moment please..."
                        }
                    
                    # Phone call - check if we should use HITL or traditional transfer
                    use_hitl_for_phone = os.getenv("PHONE_USE_HITL", "false").lower() == "true"
                    
                    if use_hitl_for_phone:
                        # Use HITL handoff for phone call
                        logger.info("📞 Phone call - using HITL handoff (PHONE_USE_HITL=true)")
                        
                        # Use default queue (agent node answer not available in tool context)
                        queue = "general"
                        
                        # Initiate HITL handoff for phone call
                        await self._initiate_human_handoff(queue=queue, trigger="transfer_tool")
                        
                        return {
                            "success": True,
                            "is_web_session": False,
                            "is_hitl": True,
                            "message": "Certainly! Let me connect you with a live agent. One moment please..."
                        }
                    
                    # Traditional Telnyx transfer for phone call
                    if call_control_id:
                        # Return success immediately so agent can speak acknowledgment first
                        # The actual transfer will happen in the background after a delay
                        logger.info(f"📞 Phone call - using Telnyx transfer (PHONE_USE_HITL=false)")
                        
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
    async def end_call(self, context: RunContext, reason: str = "user_requested_end") -> Dict[str, Any]:
        """
        End the current call/session (web or phone).

        This is LLM-managed: call this right after you say goodbye when the user indicates they want to end.
        """
        try:
            # Mark that the user ended the call (aligns end_voice_chat payload semantics)
            try:
                self._end_by = "person"
            except Exception:
                pass

            backend_url = self._backend_internal_url()

            import httpx
            from urllib.parse import quote

            room_name = (self.room_name or "").strip()
            encoded_room = quote(room_name, safe="")

            # Fetch session so we can decide phone vs web
            session_data = {}
            try:
                async with httpx.AsyncClient(timeout=3.0) as client:
                    resp = await client.get(f"{backend_url}/api/sessions/{encoded_room}")
                if resp.status_code == 200 and resp.content:
                    session_data = resp.json()
            except Exception:
                session_data = {}

            call_control_id = (session_data.get("call_control_id") or "").strip()
            source = ((session_data.get("user_data", {}) or {}).get("source") or "").strip()

            # End phone calls via Telnyx hangup
            if call_control_id and source == "telnyx_phone":
                try:
                    async with httpx.AsyncClient(timeout=10.0) as client:
                        await client.post(
                            f"{backend_url}/api/telnyx/hangup",
                            json={"call_control_id": call_control_id, "room_name": room_name},
                        )
                except Exception as e:
                    logger.warning(f"⚠️ Telnyx hangup request failed (non-fatal): {e}")
            else:
                # End web session by deleting the LiveKit room (disconnects everyone)
                try:
                    async with httpx.AsyncClient(timeout=8.0) as client:
                        await client.delete(f"{backend_url}/api/rooms/{encoded_room}")
                except Exception as e:
                    logger.warning(f"⚠️ Room delete request failed (non-fatal): {e}")

            # Best-effort local shutdown / reconciliation
            try:
                if hasattr(self, "agent_session") and self.agent_session and hasattr(self.agent_session, "_closing"):
                    self.agent_session._closing = True
            except Exception:
                pass

            try:
                # Fire-and-forget; on_session_end has guards against duplicates.
                asyncio.create_task(self.on_session_end())
            except Exception:
                pass

            return {"success": True, "message": "ok", "reason": reason}
        except Exception as e:
            logger.error(f"❌ end_call failed: {e}")
            return {"success": False, "message": "Unable to end the call right now."}
    
    @function_tool()
    async def save_collected_data(self, context: RunContext, field_name: str, value: str) -> Dict[str, Any]:
        """Save user response to collected_data and update CRM in real-time.
        
        Call this function whenever a flow question has a 'save_data_to' field and the user provides an answer.
        This will immediately update the widget/portal with the new CRM data.
        
        If AgentCore Gateway is enabled, this will route through Gateway.
        Otherwise, uses direct function call.
        
        Args:
            field_name: The field name from save_data_to (e.g., "full_name", "email", "phone", "notes_entry", "company", "company_title", "account_id")
            value: The user's response to save.
        
        Returns:
            Success status and a message.
        """
        logger.info(f"🔔 save_collected_data CALLED: field_name='{field_name}', value='{value[:50] if value else 'None'}...'")
        
        def _normalize_field_name(name: str) -> str:
            n = (name or "").strip()
            n_l = n.lower().replace("-", "_").replace(" ", "_")
            # Common aliases from flows / UI / integrations
            if n_l in {"fullname", "full_name", "name"}:
                return "full_name"
            if n_l in {"first", "first_name", "firstname"}:
                return "first_name"
            if n_l in {"last", "last_name", "lastname"}:
                return "last_name"
            if n_l in {"email", "email_address"}:
                return "email"
            if n_l in {"phone", "phone_number", "phone_mobile", "phonemobile"}:
                return "phone"
            if n_l in {"notes_entry", "notes", "note"}:
                return "notes_entry"
            if n_l in {"accountid", "account_id", "account"}:
                return "account_id"
            if n_l in {"company"}:
                return "company"
            if n_l in {"companytitle", "company_title", "title", "company_position"}:
                return "company_title"
            # Fallback: keep original normalized token so we can at least store it
            return n_l or n
        
        def _normalize_phone_number(phone: str) -> str:
            """Normalize US phone numbers to include +1 prefix"""
            import re
            if not phone:
                return phone
            
            # Remove all non-digit characters except +
            digits_only = re.sub(r'[^\d+]', '', phone)
            
            # If it already starts with +1, return as is
            if digits_only.startswith('+1'):
                return digits_only
            
            # If it starts with 1 (without +), add +
            if digits_only.startswith('1') and len(digits_only) == 11:
                return '+' + digits_only
            
            # If it's 10 digits (US number without country code), add +1
            if len(digits_only) == 10:
                return '+1' + digits_only
            
            # If it's 11 digits starting with 1, add +
            if len(digits_only) == 11 and digits_only[0] == '1':
                return '+' + digits_only
            
            # Return original if we can't normalize
            return phone

        normalized_field = _normalize_field_name(field_name)

        # Normalize phone number if needed (before Gateway or direct path)
        normalized_value = value
        if normalized_field == "phone":
            normalized_value = _normalize_phone_number(value)
            if normalized_value != value:
                logger.info(f"📞 Normalized phone number: {value} -> {normalized_value}")

        # Check if this exact value was already saved (prevent duplicate saves that cause hiccups)
        if normalized_field in self.collected_data:
            existing_value = self.collected_data[normalized_field]
            # For notes_entry, compare the list
            if normalized_field == "notes_entry":
                if isinstance(existing_value, list) and normalized_value in existing_value:
                    logger.info(f"⏭️ Skipping duplicate save for {normalized_field}: value already exists")
                    return {"success": True, "message": f"{normalized_field} already saved (duplicate prevented)"}
            # For other fields, compare directly
            elif existing_value == normalized_value:
                logger.info(f"⏭️ Skipping duplicate save for {normalized_field}: value unchanged ({normalized_value})")
                return {"success": True, "message": f"{normalized_field} already saved (duplicate prevented)"}

        # Try Gateway first if enabled
        if self.agentcore_gateway and self.agentcore_gateway.is_enabled():
            try:
                logger.info(f"🔧 Gateway enabled - calling save_collected_data via Gateway for {normalized_field}")
                result = await self.agentcore_gateway.call_tool(
                    "save_collected_data",
                    room_name=self.room_name,
                    field_name=normalized_field,
                    value=normalized_value  # Use normalized value
                )
                logger.info(f"🔧 Gateway result for {normalized_field}: {result}")
                if result.get("success") is not None:
                    # Also update local collected_data for consistency
                    if result.get("success"):
                        if normalized_field in ["full_name", "first_name", "last_name", "email", "phone", "account_id", "company", "company_title"]:
                            self.collected_data[normalized_field] = normalized_value  # Use normalized value
                            logger.info(f"✅ Updated local collected_data[{normalized_field}] = {normalized_value}")
                        elif normalized_field == "notes_entry":
                            if "notes_entry" not in self.collected_data:
                                self.collected_data["notes_entry"] = []
                            self.collected_data["notes_entry"].append(normalized_value)
                        # IMPORTANT: Gateway call doesn't emit realtime CRM updates.
                        # Keep behavior consistent with the direct path by emitting the CRM update here too.
                        # Fire-and-forget CRM updates so the agent can continue speaking immediately
                        logger.info(f"📤 Triggering CRM update for {normalized_field} via Gateway path")
                        if normalized_field == "full_name":
                            name_parts = normalized_value.strip().split(' ', 1)
                            first_name = name_parts[0] if name_parts else ""
                            last_name = name_parts[1] if len(name_parts) > 1 else ""
                            if first_name:
                                self.collected_data["first_name"] = first_name
                            if last_name:
                                self.collected_data["last_name"] = last_name
                            asyncio.create_task(self._update_crm_data(first_name=first_name, last_name=last_name))
                        elif normalized_field == "first_name":
                            asyncio.create_task(self._update_crm_data(first_name=normalized_value))
                        elif normalized_field == "last_name":
                            asyncio.create_task(self._update_crm_data(last_name=normalized_value))
                        elif normalized_field == "email":
                            asyncio.create_task(self._update_crm_data(email=normalized_value))
                        elif normalized_field == "phone":
                            asyncio.create_task(self._update_crm_data(phone=normalized_value))  # Use normalized value
                        elif normalized_field == "account_id":
                            logger.info(f"💼 Gateway path: Triggering CRM update for account_id: {normalized_value}")
                            asyncio.create_task(self._update_crm_data(account_id=normalized_value))
                        elif normalized_field == "company":
                            logger.info(f"💼 Gateway path: Triggering CRM update for company: {normalized_value}")
                            asyncio.create_task(self._update_crm_data(company=normalized_value))
                        elif normalized_field == "company_title":
                            logger.info(f"💼 Gateway path: Triggering CRM update for company_title: {normalized_value}")
                            asyncio.create_task(self._update_crm_data(company_title=normalized_value))
                        elif normalized_field == "notes_entry":
                            notes_str = " | ".join(self.collected_data.get("notes_entry", []))
                            logger.info(f"📝 Gateway path: Triggering CRM update for notes: {notes_str[:50]}...")
                            asyncio.create_task(self._update_crm_data(notes=notes_str))
                        logger.info(f"✅ Gateway path: CRM update completed for {normalized_field}")
                    else:
                        logger.warning(f"⚠️ Gateway call returned success=False for {normalized_field}: {result.get('message', 'Unknown error')}")
                    return result
                else:
                    logger.warning(f"⚠️ Gateway call returned no success field for {normalized_field}, falling back to direct")
            except Exception as e:
                logger.error(f"❌ Gateway call failed for {normalized_field}, falling back to direct: {e}", exc_info=True)
        
        # Fallback to direct function call
        logger.info(f"💾 Saving collected data: {normalized_field} = {normalized_value}")
        
        try:
            # Update internal collected_data
            if normalized_field == "full_name":
                self.collected_data["full_name"] = normalized_value
                # Attempt to split into first and last name for CRM update
                name_parts = normalized_value.strip().split(' ', 1)
                first_name = name_parts[0] if name_parts else ""
                last_name = name_parts[1] if len(name_parts) > 1 else ""
                # Also store split fields for downstream payload builders
                if first_name:
                    self.collected_data["first_name"] = first_name
                if last_name:
                    self.collected_data["last_name"] = last_name
                # Fire-and-forget CRM update so the agent can continue speaking immediately
                asyncio.create_task(self._update_crm_data(first_name=first_name, last_name=last_name))
            elif normalized_field == "first_name":
                self.collected_data["first_name"] = normalized_value
                asyncio.create_task(self._update_crm_data(first_name=normalized_value))
            elif normalized_field == "last_name":
                self.collected_data["last_name"] = normalized_value
                asyncio.create_task(self._update_crm_data(last_name=normalized_value))
            elif normalized_field == "email":
                self.collected_data["email"] = normalized_value
                asyncio.create_task(self._update_crm_data(email=normalized_value))
            elif normalized_field == "phone":
                self.collected_data["phone"] = normalized_value  # Use normalized value
                asyncio.create_task(self._update_crm_data(phone=normalized_value))
            elif normalized_field == "account_id":
                self.collected_data["account_id"] = normalized_value
                asyncio.create_task(self._update_crm_data(account_id=normalized_value))
            elif normalized_field == "company":
                self.collected_data["company"] = normalized_value
                logger.info(f"💼 Saving company: {normalized_value}")
                asyncio.create_task(self._update_crm_data(company=normalized_value))
            elif normalized_field == "company_title":
                self.collected_data["company_title"] = normalized_value
                logger.info(f"💼 Saving company title: {normalized_value}")
                asyncio.create_task(self._update_crm_data(company_title=normalized_value))
            elif normalized_field == "notes_entry":
                if "notes_entry" not in self.collected_data:
                    self.collected_data["notes_entry"] = []
                self.collected_data["notes_entry"].append(normalized_value)
                # Send all notes as a single string
                notes_str = " | ".join(self.collected_data.get("notes_entry", []))
                logger.info(f"📝 Saving notes: {notes_str[:50]}...")
                asyncio.create_task(self._update_crm_data(notes=notes_str))
            else:
                logger.warning(f"⚠️ Unknown field_name for saving data: {field_name}")
                return {
                    "success": False,
                    "message": f"Unknown field_name: {field_name}"
                }
            
            logger.info(f"✅ Data saved and CRM updated in real-time for {normalized_field}")
            logger.info(f"   - Current collected_data: {self.collected_data}")
            
            return {
                "success": True,
                # Keep tool output minimal to reduce chance of LLM parroting it.
                "message": "ok"
            }
        except Exception as e:
            logger.error(f"❌ Error saving collected data: {e}", exc_info=True)
            return {
                "success": False,
                "message": f"Error saving data: {str(e)}"
            }

    # -------------------------------------------------------------------------
    # Silence nudges (LLM-managed; avoids heuristic pattern matching in worker)
    # -------------------------------------------------------------------------
    @function_tool()
    async def expect_user_response(self, context: RunContext, question_text: str = "") -> Dict[str, Any]:
        """
        LLM should call this silently right after it asks a question.
        Starts a timer that will gently check in if the user stays silent.
        """
        try:
            self._cancel_silence_nudge()
        except Exception:
            pass

        self._silence_nudge_question = (question_text or "").strip()

        async def _runner():
            import time as _time

            delays = [
                float(os.getenv("SILENCE_NUDGE_SECONDS", "8") or "8"),
                float(os.getenv("SILENCE_NUDGE_REPEAT_SECONDS", "12") or "12"),
                float(os.getenv("SILENCE_NUDGE_REASK_SECONDS", "18") or "18"),
            ]

            for idx, delay in enumerate(delays):
                # Respect snooze (e.g. user asked to wait)
                snooze_until = float(getattr(self, "_silence_nudge_snooze_until", 0.0) or 0.0)
                now = _time.monotonic()
                if snooze_until > now:
                    await asyncio.sleep(snooze_until - now)

                await asyncio.sleep(delay)

                # Don't nudge during HITL handoff / shutdown
                if getattr(self, "_handoff_active", False):
                    return
                if hasattr(self, "agent_session") and self.agent_session:
                    if hasattr(self.agent_session, "_closing") and self.agent_session._closing:
                        return

                if idx == 0:
                    msg = os.getenv("SILENCE_NUDGE_MSG_1", "Are you still there?")
                elif idx == 1:
                    msg = os.getenv("SILENCE_NUDGE_MSG_2", "No rush — take your time. Ready when you are.")
                else:
                    q = (self._silence_nudge_question or "").strip()
                    msg = f"Just to repeat — {q}" if q else "Are you ready to answer now?"

                try:
                    if hasattr(self, "agent_session") and self.agent_session:
                        await self.agent_session.say(msg)
                except Exception:
                    return

        self._silence_nudge_task = asyncio.create_task(_runner())
        return {"success": True, "message": "ok"}

    @function_tool()
    async def snooze_user_response(self, context: RunContext, seconds: int = 60) -> Dict[str, Any]:
        """
        LLM should call this silently when the user says things like "wait/hold on".
        Snoozes silence nudges for a while so we don't pester the user.
        """
        try:
            import time as _time
            self._silence_nudge_snooze_until = _time.monotonic() + float(seconds or 0)
            self._cancel_silence_nudge()
        except Exception:
            pass
        return {"success": True, "snoozed_seconds": int(seconds or 0)}
    
    @function_tool()
    async def apply_tag(self, context: RunContext, tags: Optional[List[str]] = None, conversation_summary: Optional[str] = None) -> Dict[str, Any]:
        """Apply tags to the conversation.
        
        This function is called automatically when a bot flow reaches a "Tag this Conversation" action bot.
        Tags are now provided directly in the flow JSON under actionsToPerform[].tag_chat.tags[].
        
        Args:
            tags: List of tags to apply (extracted from flow JSON). If provided, these tags are used directly.
            conversation_summary: Optional summary (legacy parameter, not used if tags are provided)
        
        Returns:
            Success status and applied tags
        """
        # If tags are provided directly from flow JSON, use them immediately
        if tags and isinstance(tags, list) and len(tags) > 0:
            logger.info(f"🏷️ apply_tag() called with tags from flow JSON: {tags}")
            try:
                # Fire-and-forget: apply tags in background
                asyncio.create_task(self._update_crm_data(tags=tags))
                logger.info(f"✅ Tags applied successfully from flow JSON: {tags}")
                return {
                    "success": True,
                    "tags": tags,
                    "message": "ok"
                }
            except Exception as e:
                logger.error(f"❌ Error applying tags from flow JSON: {e}", exc_info=True)
                return {
                    "success": False,
                    "message": f"Failed to apply tags: {str(e)}"
                }
        
        # Legacy fallback: Try Gateway first if enabled (for backward compatibility)
        if self.agentcore_gateway and self.agentcore_gateway.is_enabled():
            try:
                logger.info("🔧 Gateway enabled - calling apply_tag via Gateway")
                result = await self.agentcore_gateway.call_tool(
                    "apply_tag",
                    room_name=self.room_name,
                    conversation_summary=conversation_summary
                )
                logger.info(f"🔧 Gateway result for apply_tag: {result}")
                if result.get("success") is not None:
                    if result.get("success"):
                        # Gateway selected tags - apply them via _update_crm_data for socket event
                        selected_tags = result.get("tags", [])
                        if selected_tags:
                            logger.info(f"📤 Triggering CRM update for tags via Gateway path: {selected_tags}")
                            await self._update_crm_data(tags=selected_tags)
                            logger.info(f"✅ Gateway path: Tags applied successfully: {selected_tags}")
                        return result
                    else:
                        logger.warning(f"⚠️ Gateway call returned success=False for apply_tag: {result.get('message', 'Unknown error')}")
                        return result
                else:
                    logger.warning(f"⚠️ Gateway call returned no success field for apply_tag, falling back to direct")
            except Exception as e:
                logger.error(f"❌ Gateway call failed for apply_tag, falling back to direct: {e}", exc_info=True)
        
        # Legacy fallback: If no tags provided, return error (tags should always come from flow JSON now)
        logger.warning("⚠️ apply_tag() called without tags parameter - tags should be extracted from flow JSON")
        return {
            "success": False,
            "message": "No tags provided. Tags must be extracted from flow JSON actionsToPerform[].tag_chat.tags[]"
        }
    
    
    @function_tool()
    async def faq_bot_request(self, context: RunContext, faq_question: str, bot_id: str = None, isVoice: bool = None) -> Dict[str, Any]:
        """Query FAQ bot for company/service information
        
        If AgentCore Gateway is enabled, this will route through Gateway.
        Otherwise, uses direct function call.
        """
        # Try Gateway first if enabled
        if self.agentcore_gateway and self.agentcore_gateway.is_enabled():
            try:
                result = await self.agentcore_gateway.call_tool(
                    "faq_bot_request",
                    query_text=faq_question,
                    faq_bot_id=bot_id or self.faq_bot_id,
                    org_name=self.org_name,
                    is_voice=isVoice if isVoice is not None else True
                )
                if result.get("success"):
                    return result
            except Exception as e:
                logger.warning(f"Gateway call failed, falling back to direct: {e}")
        
        # Fallback to direct function call (original implementation)
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
        
        # IMPORTANT: Tool calls must be silent. Do not speak "please wait" or any progress updates.
        waiting_callback = None
        
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
        """Called when session is ending - ensure CRM is finalized, then close thread."""
        # Guard: on disconnect we can get multiple triggers (Telnyx hangup + room delete + SIP disconnect).
        # Ensure we only run end-of-session logic once.
        try:
            if getattr(self, "_session_end_started", False):
                return
            self._session_end_started = True
        except Exception:
            pass

        # IMPORTANT: Do NOT fire-and-forget here. On disconnect, background tasks can get cancelled.
        # We do a short, best-effort reconciliation before ending the Alive5 thread.
        try:
            timeout_s = float(os.getenv("POSTCALL_RECONCILIATION_TIMEOUT_SECONDS", "4") or "4")
            await asyncio.wait_for(self._post_call_reconcile_crm(), timeout=timeout_s)
        except Exception:
            pass

        # Phone calls: no frontend exists to forward lk.alive5.socket instructions.
        # Send end_voice_chat directly via Alive5 socket (Option B) and skip backend web cleanup.
        if self._is_phone_call_room():
            logger.info("📞 Phone call ending - notifying Alive5 via socket (PSTN bridge)")
            try:
                org_name = getattr(self, "alive5_org_name", getattr(self, "org_name", "alive5stage0"))
                # IMPORTANT:
                # Alive5 UI historically expects a raw UUID voice_agent_id (what init_voice_agent_ack returns).
                # Using a tagged id (voice_agent_phone_*) can cause end_voice_chat to be ignored.
                raw_voice_agent_id = (getattr(self, "alive5_voice_agent_id", None) or "").strip()
                voice_agent_id = raw_voice_agent_id or (self._tagged_voice_agent_id("phone") or "")
                # Best-effort: end the Telnyx call so the dialer doesn't keep the call alive.
                try:
                    import httpx
                    from urllib.parse import quote

                    backend_url = self._backend_internal_url()
                    encoded_room = quote(self.room_name, safe="")
                    async with httpx.AsyncClient(timeout=4.0) as client:
                        sess = await client.get(f"{backend_url}/api/sessions/{encoded_room}")
                        if sess.status_code == 200:
                            call_control_id = (sess.json() or {}).get("call_control_id")
                            if call_control_id:
                                r = await client.post(
                                    f"{backend_url}/api/telnyx/hangup",
                                    json={"call_control_id": call_control_id, "room_name": self.room_name},
                                )
                                if r.status_code not in (200, 201):
                                    logger.warning(f"⚠️ Telnyx hangup request failed: {r.status_code}")
                except Exception:
                    pass

                # Fallback: delete LiveKit room so the agent session ends just like web.
                try:
                    livekit_url = os.getenv("LIVEKIT_URL")
                    livekit_api_key = os.getenv("LIVEKIT_API_KEY")
                    livekit_api_secret = os.getenv("LIVEKIT_API_SECRET")
                    if all([livekit_url, livekit_api_key, livekit_api_secret]):
                        from livekit import api

                        livekit_api_url = livekit_url.replace("ws://", "http://").replace("wss://", "https://")
                        if "18.210.238.67" in livekit_api_url:
                            livekit_api_url = livekit_api_url.replace("18.210.238.67", "localhost")
                        async with api.LiveKitAPI(livekit_api_url, livekit_api_key, livekit_api_secret) as lk_api:
                            try:
                                await lk_api.room.delete_room(api.DeleteRoomRequest(room=self.room_name))
                                logger.info(f"✅ Closed LiveKit room on end (PSTN): {self.room_name}")
                            except Exception:
                                pass
                except Exception:
                    pass

                # Give a small grace period for in-flight post_message/save_crm_data emits to reach Alive5 UI
                try:
                    await asyncio.sleep(0.8)
                except Exception:
                    pass
                await self._emit_alive5_socket_event(
                    "end_voice_chat",
                    {
                        # Match web semantics: when the caller hangs up, end_by should be "person"
                        "end_by": getattr(self, "_end_by", "voice_agent"),
                        "message_content": "Voice call completed by user" if getattr(self, "_end_by", "voice_agent") == "person" else "Voice call completed by agent",
                        "org_name": org_name,
                        "thread_id": self.alive5_thread_id,
                        "voice_agent_id": voice_agent_id,
                    },
                )
                logger.info(
                    f"✅ end_voice_chat emitted (PSTN bridge) end_by={getattr(self, '_end_by', 'voice_agent')} thread_id={getattr(self, 'alive5_thread_id', None) or 'N/A'} voice_agent_id={voice_agent_id or 'N/A'}"
                )
                try:
                    await asyncio.sleep(0.3)
                except Exception:
                    pass
            except Exception:
                pass
            await self._disconnect_alive5_socket()
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

    def _is_phone_call_room(self) -> bool:
        try:
            return bool(self.room_name and self.room_name.startswith("telnyx_call_"))
        except Exception:
            return False

    def _tagged_voice_agent_id(self, platform: str) -> str:
        """
        Return a stable, human-friendly voice agent id for Alive5 attribution/display.
        - platform: "phone" | "web"
        Result format:
          - {prefix}_phone_{id}
          - {prefix}_web_{id}
        where prefix defaults to "voice_agent" and can be overridden with A5_VOICE_AGENT_ID_PREFIX.
        """
        try:
            prefix = (os.getenv("A5_VOICE_AGENT_ID_PREFIX") or "voice_agent").strip() or "voice_agent"
            raw = (getattr(self, "alive5_voice_agent_id", None) or "").strip()
            if not raw:
                # Deterministic fallback based on room name
                raw = (self.room_name or "").strip()
            if not raw:
                return ""

            # Avoid double-tagging
            if raw.startswith(f"{prefix}_phone_") or raw.startswith(f"{prefix}_web_"):
                return raw

            # Keep the prefix (default "voice_agent_") so Alive5 recognizes it as an agent
            suffix = raw
            if raw.startswith(f"{prefix}_"):
                suffix = raw[len(prefix) + 1 :]

            platform = (platform or "").strip().lower() or "web"
            if platform not in ("phone", "web"):
                platform = "web"
            return f"{prefix}_{platform}_{suffix}"
        except Exception:
            return ""

    # ===== HITL (Human-in-the-Loop) Methods =====
    
    def _detect_agent_node(self, response_text: str) -> tuple[bool, Optional[str]]:
        """
        Detect if the current flow node is type 'agent' requiring human handoff.
        This is a simplified detection - in production, you'd parse the actual flow structure.
        Returns: (is_agent_node, queue_name)
        """
        # TODO: Implement actual flow parsing when bot_template structure is available
        # For now, return False (no agent node detected)
        return False, None
    
    async def _initiate_human_handoff(self, queue: str, trigger: str = "agent_node"):
        """Start HITL handoff sequence"""
        try:
            logger.info(f"🙋 Initiating human handoff - queue: {queue}, trigger: {trigger}")
            
            # Set handoff state
            self._handoff_active = True
            self._handoff_queue = queue

            # Start a background monitor so the AI can react to dashboard reject/end/resume
            try:
                if self._handoff_monitor_task and not self._handoff_monitor_task.done():
                    self._handoff_monitor_task.cancel()
                self._handoff_monitor_task = asyncio.create_task(self._monitor_handoff_state())
            except Exception:
                pass
            
            # Prepare payload
            payload = {
                "thread_id": self.alive5_thread_id or "",
                "crm_id": self.alive5_crm_id or "",
                "room_name": self.room_name,
                "caller_phone": "",  # TODO: Extract from session
                "queue": queue,
                "timestamp": int(asyncio.get_event_loop().time() * 1000),
                "context": f"Agent requested human assistance ({trigger})"
            }
            
            # Emit incoming_human_call event to Alive5
            if self._alive5_socket and self._alive5_socket_connected:
                try:
                    await self._emit_alive5_socket_event("incoming_human_call", payload)
                    logger.info(f"✅ Emitted incoming_human_call event to Alive5 for queue: {queue}")
                except Exception as e:
                    logger.error(f"❌ Failed to emit incoming_human_call to Alive5: {e}")
            
            # Notify dashboard via backend API
            try:
                import httpx
                backend_url = os.getenv("BACKEND_URL", "http://localhost:8000")
                async with httpx.AsyncClient() as client:
                    response = await client.post(
                        f"{backend_url}/api/dashboard/notify-call",
                        json=payload,
                        timeout=5.0
                    )
                    if response.status_code == 200:
                        result = response.json()
                        logger.info(f"✅ Notified {result.get('dashboards_notified', 0)} dashboard(s)")
                    else:
                        logger.warning(f"⚠️ Dashboard notification failed: {response.status_code}")
            except Exception as e:
                logger.error(f"❌ Failed to notify dashboard: {e}")
            
            # TODO: Play hold music or message to caller
            logger.info("📞 Waiting for human agent to join...")
            
        except Exception as e:
            logger.error(f"❌ Error initiating handoff: {e}")

    async def _monitor_handoff_state(self):
        """
        Poll backend session state to detect:
        - Dashboard rejected handoff
        - Human ended handoff with resume_ai true/false
        This lets the AI respond appropriately and exit shadow mode.
        """
        try:
            import time as _time
            import httpx
            from urllib.parse import quote

            backend_url = self._backend_internal_url()
            last_notice_at = 0.0

            while True:
                if getattr(self, "_session_end_started", False):
                    return
                if not getattr(self, "_handoff_active", False):
                    return

                # Avoid hammering backend
                await asyncio.sleep(0.8)

                try:
                    encoded_room = quote((self.room_name or "").strip(), safe="")
                    async with httpx.AsyncClient(timeout=2.0) as client:
                        resp = await client.get(f"{backend_url}/api/sessions/{encoded_room}")
                    if resp.status_code != 200:
                        continue
                    session = resp.json() if resp.content else {}
                except Exception:
                    continue

                hs = (session or {}).get("handoff_state", {}) or {}
                active = bool(hs.get("active"))
                if active:
                    continue

                rejected = bool(hs.get("rejected"))
                resume_ai = bool(hs.get("resume_ai"))

                # Handoff ended/rejected — clear state and optionally speak
                try:
                    self._handoff_active = False
                    self._handoff_queue = None
                    self._human_agent_identity = None
                except Exception:
                    pass

                # Avoid duplicate "resume" messages if multiple updates come through
                now = _time.monotonic()
                if now - last_notice_at < 1.5:
                    return
                last_notice_at = now

                if rejected:
                    msg = "No problem — it looks like no one is available right now. I can help you here. What can I assist with?"
                    try:
                        if hasattr(self, "agent_session") and self.agent_session:
                            await self.agent_session.say(msg)
                    except Exception:
                        pass
                    return

                if resume_ai:
                    try:
                        # Ensure we exit any shadow state
                        await self._exit_shadow_mode()
                    except Exception:
                        pass
                    msg = "I’m back — I’ll take it from here. How can I help?"
                    try:
                        if hasattr(self, "agent_session") and self.agent_session:
                            await self.agent_session.say(msg)
                    except Exception:
                        pass
                    return

                # Ended without resuming AI
                msg = "Okay — the call has ended. Goodbye."
                try:
                    if hasattr(self, "agent_session") and self.agent_session:
                        await self.agent_session.say(msg)
                except Exception:
                    pass
                return
        except asyncio.CancelledError:
            return
        except Exception:
            return
    
    async def _enter_shadow_mode(self):
        """Mute AI, continue transcription only"""
        try:
            logger.info("🔇 Entering shadow mode - AI muted, transcription continues")
            
            # TODO: Implement audio track muting
            # This would involve unpublishing the AI's audio track
            # while keeping STT active
            
            self._handoff_active = True
            
        except Exception as e:
            logger.error(f"❌ Error entering shadow mode: {e}")
    
    async def _exit_shadow_mode(self):
        """Resume AI after human leaves"""
        try:
            logger.info("🔊 Exiting shadow mode - AI resuming")
            
            # TODO: Implement audio track unmuting
            
            self._handoff_active = False
            self._human_agent_identity = None

            # Stop any monitor task (handoff is over)
            try:
                if self._handoff_monitor_task and not self._handoff_monitor_task.done():
                    self._handoff_monitor_task.cancel()
            except Exception:
                pass
            
        except Exception as e:
            logger.error(f"❌ Error exiting shadow mode: {e}")
    
    def _alive5_socket_base_url(self) -> str:
        """
        Determine Alive5 Socket.IO base URL.
        - If A5_SOCKET_URL is set, use it (recommended).
        - Otherwise, derive from A5_BASE_URL (api-v2-stage -> api-stage, api-v2 -> api).
        """
        explicit = (os.getenv("A5_SOCKET_URL") or "").strip()
        if explicit:
            return explicit

        base = (os.getenv("A5_BASE_URL") or "https://api-stage.alive5.com").strip()
        try:
            from urllib.parse import urlparse

            parsed = urlparse(base if "://" in base else ("https://" + base))
            host = (parsed.netloc or "").strip()
            if not host:
                return "wss://api-stage.alive5.com"

            host = host.replace("api-v2-stage.", "api-stage.")
            host = host.replace("api-v2.", "api.")
            return f"wss://{host}"
        except Exception:
            return "wss://api-stage.alive5.com"

    async def _ensure_alive5_socket_connected(self) -> bool:
        """
        Connect directly to Alive5 Socket.IO and initialize voice agent.
        Originally for PSTN calls only, but now also enabled for web sessions
        to ensure CRM updates reach Alive5 even if frontend widget doesn't forward them.
        This is additive and isolated; failures never affect the call flow.
        """
        # Allow both web and phone sessions to connect
        # if not self._is_phone_call_room():
        #     return False

        if not (self.alive5_thread_id and self.alive5_crm_id and self.alive5_channel_id):
            return False

        if self._alive5_socket_connected and self._alive5_socket is not None:
            return True

        async with self._alive5_socket_connect_lock:
            if self._alive5_socket_connected and self._alive5_socket is not None:
                return True

            try:
                import socketio
                from urllib.parse import urlencode

                api_key = (os.getenv("A5_API_KEY") or "").strip()
                if not api_key:
                    logger.warning("⚠️ A5_API_KEY missing - cannot connect Alive5 socket (PSTN bridge)")
                    return False

                base_url = self._alive5_socket_base_url()
                qs = urlencode(
                    {
                        "type": "voice_agent",
                        "x-a5-apikey": api_key,
                        "thread_id": self.alive5_thread_id,
                        "crm_id": self.alive5_crm_id,
                        "channel_id": self.alive5_channel_id,
                    }
                )
                url = f"{base_url}?{qs}"

                self._alive5_socket_init_event.clear()

                sio = socketio.AsyncClient(
                    reconnection=False,
                    logger=False,
                    engineio_logger=False,
                )

                @sio.event
                async def connect():
                    self._alive5_socket_connected = True
                    logger.info("✅ Alive5 socket connected (PSTN bridge)")
                    # Match frontend behavior: wait a beat after connect before init
                    try:
                        await asyncio.sleep(0.1)
                        await sio.emit(
                            "init_voice_agent",
                            {
                                "thread_id": self.alive5_thread_id,
                                "crm_id": self.alive5_crm_id,
                                "channel_id": self.alive5_channel_id,
                            },
                        )
                    except Exception as e:
                        logger.warning(f"⚠️ Failed to emit init_voice_agent: {e}")

                @sio.event
                async def disconnect():
                    self._alive5_socket_connected = False
                    logger.info("ℹ️ Alive5 socket disconnected (PSTN bridge)")

                @sio.on("init_voice_agent_ack")
                async def _on_init_ack(data):
                    try:
                        logger.info(f"📨 init_voice_agent_ack received (PSTN): {data}")
                        voice_agent_id = None
                        if isinstance(data, dict):
                            voice_agent_id = data.get("voice_agent_id") or data.get("voiceAgentId") or data.get("assigned_to")
                        if voice_agent_id:
                            # Store the RAW value from server (will be shortened in post_message for display)
                            self.alive5_voice_agent_id = voice_agent_id
                            logger.info(f"🆔 Stored voice agent id (PSTN): {voice_agent_id} (from server)")
                        else:
                            # IMPORTANT: Alive5 sometimes does not return voice_agent_id (seen in frontend logs).
                            # Frontend falls back to a stable id derived from the socket id (first 8 chars).
                            # For PSTN, we default to "phone" platform to distinguish from web sessions.
                            try:
                                prefix = (os.getenv("A5_VOICE_AGENT_ID_PREFIX") or "voice_agent").strip() or "voice_agent"
                                sid = getattr(sio, "sid", None) or ""
                                sid_short = (str(sid)[:8] or "").lower()
                                if sid_short:
                                    # Default to "phone" for PSTN calls, can be overridden
                                    platform_override = (os.getenv("A5_VOICE_AGENT_ID_PLATFORM_FOR_PSTN") or "phone").strip().lower()
                                    if platform_override not in ("web", "phone"):
                                        platform_override = "phone"
                                    fallback_id = f"{prefix}_{platform_override}_{sid_short}"
                                    self.alive5_voice_agent_id = fallback_id
                                    logger.info(f"🆔 Generated fallback voice agent id (PSTN): {fallback_id} (socket_id={sid[:16] if sid else 'N/A'})")
                            except Exception:
                                pass
                        logger.info("✅ Alive5 init_voice_agent_ack received (PSTN bridge)")
                    except Exception:
                        pass
                    finally:
                        self._alive5_socket_init_event.set()

                await sio.connect(
                    url,
                    transports=["websocket"],
                    socketio_path="socket.io",
                    wait=True,
                    wait_timeout=5,
                )

                try:
                    await asyncio.wait_for(self._alive5_socket_init_event.wait(), timeout=3.0)
                except Exception:
                    logger.warning("⚠️ Alive5 init_voice_agent_ack not received (PSTN bridge) — events may be ignored by Alive5")

                self._alive5_socket = sio
                return True
            except Exception as e:
                self._alive5_socket = None
                self._alive5_socket_connected = False
                logger.warning(f"⚠️ Could not connect Alive5 socket (PSTN bridge): {e}")
                return False

    async def _emit_alive5_socket_event(self, event_name: str, payload: dict) -> bool:
        """Emit an event directly to Alive5 Socket.IO for PSTN calls (additive + isolated)."""
        try:
            ok = await self._ensure_alive5_socket_connected()
            if not ok or not self._alive5_socket:
                return False
            await self._alive5_socket.emit(event_name, payload)
            return True
        except Exception as e:
            logger.debug(f"Alive5 socket emit failed ({event_name}): {e}")
            return False

    async def _disconnect_alive5_socket(self):
        """Disconnect Alive5 socket client if connected (PSTN bridge)."""
        try:
            if self._alive5_socket:
                try:
                    await self._alive5_socket.disconnect()
                except Exception:
                    pass
        finally:
            self._alive5_socket = None
            self._alive5_socket_connected = False
    
    def _build_query_string(self) -> dict:
        """Build query_string with collected CRM data"""
        # Prefer explicit first/last; fallback to split full_name
        first_name = self.collected_data.get("first_name", "") or ""
        last_name = self.collected_data.get("last_name", "") or ""
        if not (first_name or last_name):
            full_name = self.collected_data.get("full_name", "")
            name_parts = full_name.split(' ', 1) if full_name else ["", ""]
            first_name = name_parts[0] if len(name_parts) > 0 else ""
            last_name = name_parts[1] if len(name_parts) > 1 else ""
        
        return {
            "first_name": first_name,
            "last_name": last_name,
            "email": self.collected_data.get("email", ""),
            "notes": " | ".join(self.collected_data.get("notes_entry", [])) if self.collected_data.get("notes_entry") else "",
            "crm_id": self.alive5_crm_id or "",
            "agent_email": "",
            "accountid": self.collected_data.get("account_id", "") or "",
            "company": self.collected_data.get("company", "") or "",
            "companytitle": self.collected_data.get("company_title", "") or "",
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
        # Prefer explicit first/last; fallback to split full_name
        first_name = self.collected_data.get("first_name", "") or ""
        last_name = self.collected_data.get("last_name", "") or ""
        if not (first_name or last_name):
            full_name = self.collected_data.get("full_name", "")
            name_parts = full_name.split(' ', 1) if full_name else ["", ""]
            first_name = name_parts[0] if len(name_parts) > 0 else ""
            last_name = name_parts[1] if len(name_parts) > 1 else ""
        
        return {
            "allow_zapier_syns": True,
            "assigned_user": [],
            "created_at": timestamp,
            "crm_id": self.alive5_crm_id or "",
            "crm_thread_type": "livechat",
            "crm_type": "livechat",
            "org_name": self.alive5_org_name,
            "updated_at": timestamp,
            "firstName": first_name,
            "lastName": last_name,
            "email": self.collected_data.get("email", ""),
            "phoneMobile": self.collected_data.get("phone", ""),
            "notes": " | ".join(self.collected_data.get("notes_entry", [])) if self.collected_data.get("notes_entry") else "",
            # Keep these fields if CRM accepts them; safe to include as blanks if unused
            "accountId": self.collected_data.get("account_id", "") or "",
            "company": self.collected_data.get("company", "") or "",
            "companyTitle": self.collected_data.get("company_title", "") or ""
        }
    
    async def _update_crm_data(
        self,
        first_name: str = None,
        last_name: str = None,
        email: str = None,
        phone: str = None,
        notes: str = None,
        account_id: str = None,
        company: str = None,
        company_title: str = None,
        tags: List[str] = None
    ):
        """Update CRM data using new voice agent socket save_crm_data event"""
        try:
            logger.info(f"🔄 _update_crm_data called with: first_name={first_name}, last_name={last_name}, email={email}, phone={phone}, account_id={account_id}, company={company}, company_title={company_title}, tags={tags}")
            session_ready = bool(self.alive5_crm_id and self.alive5_thread_id)
            
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
                self.collected_data["first_name"] = first_name
            if last_name:
                if self.collected_data.get("full_name"):
                    self.collected_data["full_name"] = f"{first_name or ''} {last_name}".strip()
                self.collected_data["last_name"] = last_name
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
            if account_id:
                self.collected_data["account_id"] = account_id
            if company:
                self.collected_data["company"] = company
                logger.debug(f"💼 Updated collected_data with company: {company}")
            if company_title:
                self.collected_data["company_title"] = company_title
                logger.debug(f"💼 Updated collected_data with company_title: {company_title}")
            if tags:
                # Store tags in collected_data
                self.collected_data["tags"] = tags if isinstance(tags, list) else [tags] if tags else []
                logger.debug(f"🏷️ Updated collected_data with tags: {tags}")
            
            # Send CRM update instructions to frontend via data channel
            if not hasattr(self, 'room') or not self.room:
                return
            
            import json
            
            def _add_updates(key_base: str, value_str: str, aliases: List[str]):
                for k in aliases:
                    updates.append({"key": k, "value": value_str})
            
            def _add_tags_update(tags_list: List[str]):
                """Add tags update - value must be an array"""
                if tags_list:
                    updates.append({"key": "tags", "value": tags_list})

            # Some environments expect different key formats; emit a small set of safe aliases.
            updates = []
            if first_name and first_name.strip():
                _add_updates("first_name", first_name.strip(), ["first_name", "firstName"])
            if last_name and last_name.strip():
                _add_updates("last_name", last_name.strip(), ["last_name", "lastName"])
            if email and email.strip():
                _add_updates("email", email.strip(), ["email"])
            if phone and phone.strip():
                _add_updates("phone", phone.strip(), ["phone", "phone_mobile", "phoneMobile"])
            if notes and notes.strip():
                # Some Alive5 consumers expect notes under different keys depending on integration/version.
                _add_updates("notes", notes.strip(), ["notes", "note", "notes_entry", "notesEntry"])
            if account_id and str(account_id).strip():
                # Send camelCase version first (matches _build_crm_data format), then aliases
                account_id_str = str(account_id).strip()
                _add_updates("accountId", account_id_str, ["accountId", "account_id", "accountid"])
                logger.info(f"💼 Sending accountId to CRM: {account_id_str}")
            if company and company.strip():
                company_str = company.strip()
                # Some Alive5 consumers expect company in different formats.
                _add_updates("company", company_str, ["company", "company_name", "companyName"])
                logger.info(f"💼 Sending company to CRM: {company_str}")
            if company_title and company_title.strip():
                # Send camelCase version first (matches _build_crm_data format), then aliases
                company_title_str = company_title.strip()
                _add_updates(
                    "companyTitle",
                    company_title_str,
                    ["companyTitle", "company_title", "companytitle", "companyTitleName", "title"],
                )
                logger.info(f"💼 Sending companyTitle to CRM: {company_title_str}")
            if tags:
                # Tags must be sent as an array
                tags_list = tags if isinstance(tags, list) else [tags] if tags else []
                if tags_list:
                    _add_tags_update(tags_list)
                    logger.info(f"🏷️ Sending tags to CRM: {tags_list}")

            # If session isn't initialized yet, queue updates so early fields (like name) aren't lost.
            if not session_ready:
                if updates:
                    logger.warning(
                        f"⚠️ CRM update queued - session not initialized yet "
                        f"(crm_id={self.alive5_crm_id}, thread_id={self.alive5_thread_id})"
                    )
                    self._pending_crm_updates.extend(updates)
                return
            
            for update in updates:
                socket_instruction = {
                    "action": "emit",
                    "event": "save_crm_data",
                    "payload": {
                        "crm_id": self.alive5_crm_id or "",
                        "thread_id": self.alive5_thread_id or "",
                        "key": update["key"],
                        "value": update["value"]
                    }
                }
                
                # For tags, show full array; for other fields, truncate long values
                if update['key'] == 'tags' and isinstance(update['value'], list):
                    logger.info(f"📤 Sending CRM update: key={update['key']}, value={update['value']}")
                else:
                    value_str = str(update['value'])
                    logger.info(f"📤 Sending CRM update: key={update['key']}, value={value_str[:50]}...")
                await self.room.local_participant.publish_data(
                    json.dumps(socket_instruction).encode('utf-8'),
                    topic="lk.alive5.socket"
                )
                logger.debug(f"✅ CRM update sent via data channel: {update['key']}")

                # Also emit directly to Alive5 socket for reliability
                # (originally PSTN-only, but web sessions may have frontend widgets that don't forward all fields)
                # Run in background to avoid blocking the agent's response
                try:
                    asyncio.create_task(
                        self._emit_alive5_socket_event("save_crm_data", socket_instruction["payload"])
                    )
                except Exception:
                    pass
                
        except Exception as e:
            logger.warning(f"⚠️ Could not update CRM data: {e}")

    async def _flush_pending_crm_updates(self):
        """Flush queued CRM updates once Alive5 session (thread_id/crm_id) is ready."""
        try:
            if not self._pending_crm_updates:
                return
            if not (self.alive5_crm_id and self.alive5_thread_id):
                return
            if not hasattr(self, "room") or not self.room:
                return

            import json

            queued = self._pending_crm_updates
            self._pending_crm_updates = []

            logger.info(f"📤 Flushing {len(queued)} queued CRM updates (session ready)")
            for update in queued:
                socket_instruction = {
                    "action": "emit",
                    "event": "save_crm_data",
                    "payload": {
                        "crm_id": self.alive5_crm_id,
                        "thread_id": self.alive5_thread_id,
                        "key": update.get("key"),
                        "value": update.get("value"),
                    },
                }
                await self.room.local_participant.publish_data(
                    json.dumps(socket_instruction).encode("utf-8"),
                    topic="lk.alive5.socket",
                )
                # PSTN bridge: also emit directly (no frontend in phone rooms)
                try:
                    await self._emit_alive5_socket_event("save_crm_data", socket_instruction["payload"])
                except Exception:
                    pass
        except Exception as e:
            logger.warning(f"⚠️ Could not flush queued CRM updates: {e}")
    
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

            # PSTN bridge: also emit directly to Alive5 socket (phone rooms have no frontend widget to relay).
            # IMPORTANT: For web sessions, the frontend already relays `post_message` from lk.alive5.socket,
            # so direct socket emit would duplicate messages in Alive5.
            if self._is_phone_call_room():
                try:
                    payload = dict(socket_instruction.get("payload") or {})
                    # Match frontend semantics:
                    # - remove is_agent flag (server doesn't expect it)
                    # - include voiceAgentId ONLY for agent messages (server maps to created_by/user_id)
                    payload.pop("is_agent", None)
                    # Frontend also clears any existing voiceAgentId fields to ensure a clean state.
                    payload.pop("voiceAgentId", None)
                    payload.pop("voice_agent_id", None)
                    if is_agent:
                        # CRITICAL FIX: Match web format for phone IDs.
                        # Web uses: voice_agent_web_8aqz5xbl (8 chars)
                        # Phone should use: voice_agent_phone_9e4a83df (first 8 chars of UUID, no dashes)
                        raw_id = getattr(self, 'alive5_voice_agent_id', None)
                        if raw_id:
                            # Extract short ID: first 8 alphanumeric chars (remove dashes/underscores)
                            clean_id = raw_id.replace("-", "").replace("_", "")[:8].lower()
                            voice_agent_id = f"voice_agent_phone_{clean_id}"
                            payload["voiceAgentId"] = voice_agent_id
                            logger.info(f"🆔 Using voiceAgentId for post_message: {voice_agent_id} (short format, raw={raw_id[:16]}...)")
                        else:
                            logger.warning(f"⚠️ No voiceAgentId available for agent message (alive5_voice_agent_id not set)")
                    ok = await self._emit_alive5_socket_event("post_message", payload)
                    try:
                        if is_agent:
                            logger.info(f"📤 PSTN post_message emitted (agent) ok={ok} voiceAgentId={payload.get('voiceAgentId','N/A')}")
                        else:
                            logger.info(f"📤 PSTN post_message emitted (user) ok={ok}")
                    except Exception:
                        pass
                except Exception:
                    pass
            
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

            # Name auto-detection disabled: it was too aggressive for voice (e.g., "I'm looking forward..."
            # being mistaken as a name). Names should be collected explicitly by the flow/tooling.
            
            # Detect phone pattern (US format: (xxx) xxx-xxxx, xxx-xxx-xxxx, xxx.xxx.xxxx, or 10+ digits)
            phone_pattern = r'(\+?1[-.\s]?)?\(?([0-9]{3})\)?[-.\s]?([0-9]{3})[-.\s]?([0-9]{4})'
            phone_match = re.search(phone_pattern, user_text_clean)
            if phone_match and not self.collected_data.get("phone"):
                # Extract and normalize phone number
                phone_parts = phone_match.groups()
                phone = ''.join(filter(str.isdigit, ''.join(phone_parts[1:])))  # Get last 3 groups, remove non-digits
                if len(phone) == 10:  # Valid US phone number
                    # Normalize to +1 format (save_collected_data will handle normalization, but we can format it here too)
                    phone_normalized = f"+1{phone}"
                    logger.info(f"🔍 Auto-detected phone in user message: {phone_normalized}")
                    await self.save_collected_data(None, "phone", phone_normalized)
            
            # Auto-detect company if user mentions it and we don't have it yet
            # Look for patterns like "company is X", "I work at X", "organization is X", etc.
            if not self.collected_data.get("company"):
                company_patterns = [
                    r'(?:company|organization|org|employer|work at|work for|with)\s+(?:is|are|called|named)?\s+([A-Z][A-Za-z0-9\s&.,-]{2,30})',
                    r'(?:I\s+)?(?:work\s+at|work\s+for|am\s+with)\s+([A-Z][A-Za-z0-9\s&.,-]{2,30})',
                    r'([A-Z][A-Za-z0-9\s&.,-]{2,30})\s+(?:is\s+my\s+company|is\s+the\s+company)',
                ]
                for pattern in company_patterns:
                    company_match = re.search(pattern, user_text_clean, re.IGNORECASE)
                    if company_match:
                        company = company_match.group(1).strip()
                        # Filter out common false positives
                        # Don't block real company names like "Google" / "Apple" etc.
                        # Only block obvious non-values.
                        if company.lower() not in ['the', 'a', 'an'] and len(company) >= 2:
                            logger.info(f"🔍 Auto-detected company in user message: {company}")
                            await self.save_collected_data(None, "company", company)
                            break
            
            # Auto-detect account ID if user mentions it and we don't have it yet
            # Look for patterns like "account id is X", "account number is X", "ID is X", etc.
            if not self.collected_data.get("account_id"):
                account_id_patterns = [
                    # Require the literal "account" phrase to avoid false positives like "dID that"
                    r'\baccount\s*(?:id|number|#)\b\s*(?:is\s*)?([A-Za-z0-9-]{2,20})\b',
                    r'\bmy\s+account\s*(?:id|number)\b\s*(?:is\s*)?([A-Za-z0-9-]{2,20})\b',
                ]
                for pattern in account_id_patterns:
                    account_match = re.search(pattern, user_text_clean, re.IGNORECASE)
                    if account_match:
                        account_id = account_match.group(1).strip()
                        logger.info(f"🔍 Auto-detected account_id in user message: {account_id}")
                        await self.save_collected_data(None, "account_id", account_id)
                        break
            
            # Auto-detect company title if user mentions it and we don't have it yet
            # Look for patterns like "title is X", "I'm a X", "position is X", etc.
            if not self.collected_data.get("company_title"):
                title_patterns = [
                    r'(?:title|position|role|job\s+title)\s+(?:is|is\s+)?([A-Z][A-Za-z\s-]{2,40})',
                    r'(?:I\s+am\s+(?:a|an)\s+|I\'m\s+(?:a|an)\s+)([A-Z][A-Za-z\s-]{2,40})(?:\s+at|\s+for|$)',
                ]
                for pattern in title_patterns:
                    title_match = re.search(pattern, user_text_clean, re.IGNORECASE)
                    if title_match:
                        title = title_match.group(1).strip()
                        # Filter out common false positives
                        if title.lower() not in ['the', 'a', 'an', 'manager', 'director'] or len(title) > 5:
                            logger.info(f"🔍 Auto-detected company_title in user message: {title}")
                            await self.save_collected_data(None, "company_title", title)
                            break
            
            # Name auto-detection disabled (see note above).
        except Exception as e:
            # Don't let auto-detection errors break the conversation
            logger.debug(f"⚠️ Auto-detection error (non-critical): {e}")

    async def _post_call_reconcile_crm(self):
        """
        Fail-safe: at call end, run an LLM extraction over the full transcript and patch CRM for
        missing/clearly-wrong fields.
        """
        try:
            if (os.getenv("POSTCALL_RECONCILIATION", "true") or "true").lower() != "true":
                return
            if not self._conversation_log:
                return
            # Need session identifiers to update CRM
            if not (self.alive5_crm_id and self.alive5_thread_id):
                return
            if not hasattr(self, "llm") or not self.llm:
                return

            import json
            import re
            from livekit.agents.llm import ChatContext, ChatMessage

            # Build a compact transcript (avoid huge prompts)
            lines: List[str] = []
            for item in self._conversation_log[-80:]:
                role = item.get("role", "")
                txt = (item.get("text") or "").strip()
                if not txt:
                    continue
                prefix = "User" if role == "user" else "Agent"
                lines.append(f"{prefix}: {txt}")
            transcript = "\n".join(lines)

            current = {
                "full_name": self.collected_data.get("full_name"),
                "first_name": self.collected_data.get("first_name"),
                "last_name": self.collected_data.get("last_name"),
                "email": self.collected_data.get("email"),
                "phone": self.collected_data.get("phone"),
                "account_id": self.collected_data.get("account_id"),
                "company": self.collected_data.get("company"),
                "company_title": self.collected_data.get("company_title"),
            }

            system = (
                "You extract structured CRM fields from a conversation transcript.\n"
                "Return ONLY valid JSON (no markdown, no extra text).\n"
                "If a field is unknown, set value to null and confidence to 0.\n"
                "Use confidence 0..1.\n"
            )

            user = {
                "task": "extract_or_correct_contact_fields",
                "current_fields": current,
                "transcript": transcript,
                "schema": {
                    "full_name": {"value": "string|null", "confidence": "number"},
                    "email": {"value": "string|null", "confidence": "number"},
                    "phone": {"value": "string|null", "confidence": "number"},
                    "account_id": {"value": "string|null", "confidence": "number"},
                    "company": {"value": "string|null", "confidence": "number"},
                    "company_title": {"value": "string|null", "confidence": "number"},
                },
            }

            # Use the same extraction helper as mid-call verification (non-stream for Bedrock).
            await self._llm_extract_and_patch(
                fields=["full_name", "email", "phone", "account_id", "company", "company_title"],
                max_lines=80,
                min_confidence=float(os.getenv("POSTCALL_RECONCILIATION_CONFIDENCE", "0.75") or "0.75"),
            )
            return

            threshold = float(os.getenv("POSTCALL_RECONCILIATION_CONFIDENCE", "0.75") or "0.75")

            def _looks_wrong_name(v: str | None) -> bool:
                if not v:
                    return True
                vv = v.strip().lower()
                # Alive5 sometimes defaults to "Voice Caller" for voice sessions until a name is set.
                if vv in {"voice caller", "caller"} or "voice caller" in vv:
                    return True
                if "looking forward" in vv:
                    return True
                if any(ch.isdigit() for ch in vv):
                    return True
                if "@" in vv:
                    return True
                if len(vv.split()) > 4:
                    return True
                return False

            updates = {}
            for field in ["full_name", "email", "phone", "account_id", "company", "company_title"]:
                obj = data.get(field) or {}
                val = obj.get("value")
                conf = float(obj.get("confidence") or 0.0)
                if conf < threshold or not val or not isinstance(val, str) or not val.strip():
                    continue

                if field == "full_name":
                    if _looks_wrong_name(self.collected_data.get("full_name")):
                        updates["first_name"] = val.strip().split(" ", 1)[0]
                        if " " in val.strip():
                            updates["last_name"] = val.strip().split(" ", 1)[1]
                else:
                    if not self.collected_data.get(field):
                        updates[field] = val.strip()

            if updates:
                logger.info(f"🧾 Post-call reconciliation applying updates: {list(updates.keys())}")
                # Await so updates are actually sent before the session shuts down.
                try:
                    await asyncio.wait_for(self._update_crm_data(**updates), timeout=3.0)
                except Exception:
                    pass

        except Exception as e:
            logger.warning(f"⚠️ Post-call reconciliation failed (non-fatal): {e}")
    
    async def on_user_turn_completed(self, turn_ctx, new_message):
        """Handle user input and publish to frontend"""
        # During HITL handoff/shadow mode, do NOT let the AI generate replies.
        # We still publish user transcripts + forward user messages to Alive5 for logging.
        user_text = new_message.text_content or ""
        if user_text.strip():
            if getattr(self, "_handoff_active", False):
                # Append to transcript (for post-call reconciliation)
                try:
                    self._conversation_log.append({"role": "user", "text": user_text.strip()})
                except Exception:
                    pass

                # Publish user transcript to frontend (non-blocking)
                asyncio.create_task(self._publish_to_frontend("user_transcript", user_text, speaker="User"))

                # During handoff, skip auto-detection/LLM-verification to avoid accidental saves.
                # Still send the user message to Alive5 so the conversation log remains complete.
                asyncio.create_task(self._send_message_to_alive5(user_text, is_agent=False))
                return

        # CRITICAL: Call parent method FIRST to maintain conversation state
        # This ensures the agent's conversation history is preserved (normal AI mode only)
        await super().on_user_turn_completed(turn_ctx, new_message)
        
        # Then do our side effects (non-blocking, fire-and-forget)
        user_text = new_message.text_content or ""
        if user_text.strip():
            # Any user speech cancels pending silence nudges
            try:
                self._cancel_silence_nudge()
            except Exception:
                pass

            # Append to transcript (for post-call reconciliation)
            try:
                self._conversation_log.append({"role": "user", "text": user_text.strip()})
            except Exception:
                pass

            # Publish user transcript to frontend (non-blocking)
            asyncio.create_task(self._publish_to_frontend("user_transcript", user_text, speaker="User"))
            # logger.info(f"USER: {user_text}")
        
            # Auto-detect and save email/phone/company/etc (name auto-detection is intentionally disabled inside this helper).
            asyncio.create_task(self._auto_detect_and_save_data(user_text))

            # LLM-based verification trigger (web + phone):
            # If user mentions "name" and CRM name looks missing/default/wrong, run a quick extraction + patch.
            try:
                asyncio.create_task(self._maybe_verify_crm_fields(user_text))
            except Exception:
                pass
        
            # Send user message to Alive5 (non-blocking to avoid interfering with agent processing)
            asyncio.create_task(self._send_message_to_alive5(user_text, is_agent=False))

    def _cancel_silence_nudge(self):
        task = getattr(self, "_silence_nudge_task", None)
        if task and not task.done():
            task.cancel()
        self._silence_nudge_task = None
        self._silence_nudge_question = None

    def _schedule_silence_nudge_if_question(self, agent_text: str):
        """
        If the agent just asked a question and the user is silent, do a gentle human-like check-in.
        """
        try:
            # Deprecated: silence nudges are now LLM-managed via expect_user_response().
            return

            async def _runner():
                import time as _time

                # Stage delays: gentle check-in -> no rush -> repeat question
                delays = [
                    float(os.getenv("SILENCE_NUDGE_SECONDS", "8") or "8"),
                    float(os.getenv("SILENCE_NUDGE_REPEAT_SECONDS", "12") or "12"),
                    float(os.getenv("SILENCE_NUDGE_REASK_SECONDS", "18") or "18"),
                ]

                for idx, delay in enumerate(delays):
                    # Respect snooze if user said "wait"
                    snooze_until = float(getattr(self, "_silence_nudge_snooze_until", 0.0) or 0.0)
                    now = _time.monotonic()
                    if snooze_until > now:
                        await asyncio.sleep(snooze_until - now)

                    await asyncio.sleep(delay)

                    # Don't nudge during HITL handoff / shutdown
                    if getattr(self, "_handoff_active", False):
                        return
                    if hasattr(self, "agent_session") and self.agent_session:
                        if hasattr(self.agent_session, "_closing") and self.agent_session._closing:
                            return

                    if idx == 0:
                        msg = os.getenv("SILENCE_NUDGE_MSG_1", "Are you still there?")
                    elif idx == 1:
                        msg = os.getenv("SILENCE_NUDGE_MSG_2", "No rush — take your time. Ready when you are.")
                    else:
                        q = self._silence_nudge_question or ""
                        msg = f"Just to repeat — {q}" if q else "Are you ready to answer now?"

                    try:
                        if hasattr(self, "agent_session") and self.agent_session:
                            await self.agent_session.say(msg)
                    except Exception:
                        return

            self._silence_nudge_task = asyncio.create_task(_runner())
        except Exception:
            return

    async def _maybe_verify_crm_fields(self, user_text: str):
        """
        LLM-based mid-call verification to avoid missing CRM fields.
        Triggered when user mentions key words (e.g. "name") and current CRM fields look missing/default.
        """
        try:
            # Default OFF: this mid-call extractor can introduce dead-air because it competes with
            # the main response generation (especially right after the name question).
            # If you want it, explicitly set MIDCALL_CRM_VERIFY=true.
            if not (os.getenv("MIDCALL_CRM_VERIFY", "false") or "false").lower() == "true":
                return
            if not hasattr(self, "llm") or not self.llm:
                return
            if not (self.alive5_crm_id and self.alive5_thread_id):
                return

            t = (user_text or "").lower()
            wants_name = "name" in t
            wants_email = "email" in t or "e-mail" in t or "mail" in t
            wants_phone = "phone" in t or "number" in t
            wants_company = "company" in t or "organization" in t or "organisation" in t
            wants_account = "account" in t or "account id" in t
            wants_title = "title" in t or "position" in t or "role" in t

            # Find the last assistant prompt (for asked-for-field triggers)
            last_assistant = ""
            try:
                for item in reversed(self._conversation_log[-12:]):
                    if item.get("role") == "assistant":
                        last_assistant = (item.get("text") or "").strip()
                        break
            except Exception:
                last_assistant = ""
            a = last_assistant.lower() if last_assistant else ""

            asked_for_name = any(p in a for p in ["may i have your name", "what is your name", "your name?", "can i have your name", "could i have your name", "may i get your name"])
            asked_for_email = "email" in a or "e-mail" in a
            asked_for_company = "company" in a or "organization" in a or "organisation" in a
            asked_for_phone = ("phone" in a and "number" in a) or "phone number" in a
            asked_for_account = "account" in a and ("id" in a or "number" in a)
            asked_for_title = "title" in a or "position" in a or "role" in a

            # Fast-path for the common "name" question to avoid an extra Bedrock roundtrip.
            # This was a major source of ~5s dead-air because we were doing:
            #   1) Bedrock extract to parse name
            #   2) Bedrock response generation for the next question
            # If we just asked for the name, treat the user's reply as the name (with light cleanup).
            try:
                cur_name = (self.collected_data.get("full_name") or "").strip().lower()
                needs_name = (wants_name or asked_for_name) and (
                    not cur_name or cur_name in {"voice caller", "caller"} or "looking forward" in cur_name
                )
                if asked_for_name and needs_name:
                    import re

                    v = (user_text or "").strip()
                    # Strip common leading phrases
                    v = re.sub(
                        r"^(yes|yeah|yep|sure|okay|ok|my name is|i am|i'm|this is|it's|it is)\b[:\s,.-]*",
                        "",
                        v,
                        flags=re.IGNORECASE,
                    ).strip()
                    # Keep only letters/spaces/apostrophes/hyphens
                    v = re.sub(r"[^A-Za-z\s'\-]", " ", v)
                    v = re.sub(r"\s+", " ", v).strip()

                    # Heuristic acceptance: 1-4 words, no digits/@, not empty
                    parts = [p for p in v.split(" ") if p]
                    if 1 <= len(parts) <= 4 and all(p.isalpha() or ("'" in p) or ("-" in p) for p in parts):
                        fast_name = " ".join(parts)
                        logger.info(f"⚡ Fast name capture (no LLM): full_name='{fast_name}' (from user_text)")
                        try:
                            asyncio.create_task(self.save_collected_data(None, "full_name", fast_name))
                        except Exception:
                            pass
                        return
            except Exception:
                pass

            # Determine which fields we should verify right now (only if missing / default)
            fields: List[str] = []
            cur_name = (self.collected_data.get("full_name") or "").strip().lower()
            if (wants_name or asked_for_name) and (not cur_name or cur_name in {"voice caller", "caller"} or "looking forward" in cur_name):
                fields.append("full_name")
            if (wants_email or asked_for_email) and not (self.collected_data.get("email") or "").strip():
                fields.append("email")
            if (wants_phone or asked_for_phone) and not (self.collected_data.get("phone") or "").strip():
                fields.append("phone")
            if (wants_company or asked_for_company) and not (self.collected_data.get("company") or "").strip():
                fields.append("company")
            if (wants_account or asked_for_account) and not (self.collected_data.get("account_id") or "").strip():
                fields.append("account_id")
            if (wants_title or asked_for_title) and not (self.collected_data.get("company_title") or "").strip():
                fields.append("company_title")

            if not fields:
                return

            # Cooldown per field to avoid spam
            try:
                import time
                now = time.monotonic()
                cooldown = float(os.getenv("MIDCALL_CRM_VERIFY_COOLDOWN_SECONDS", "2") or "2")
                fields = [f for f in fields if (now - float(self._last_crm_verify_at.get(f, 0.0))) >= cooldown]
                for f in fields:
                    self._last_crm_verify_at[f] = now
            except Exception:
                pass

            if not fields:
                return

            logger.info(
                f"🔎 Mid-call CRM verify triggered fields={fields} (asked_for={bool(last_assistant)})"
            )
            await self._llm_extract_and_patch(fields=fields, max_lines=12, min_confidence=0.70)
        except Exception as e:
            logger.debug(f"⚠️ Mid-call CRM verify failed (ignored): {e}")
            return

    async def _llm_extract_and_patch(self, fields: List[str], max_lines: int = 40, min_confidence: float = 0.75):
        """Run a small extraction over recent transcript and patch CRM for the given fields."""
        try:
            if not self._conversation_log:
                return
            if not hasattr(self, "llm") or not self.llm:
                return

            import json
            import re
            from livekit.agents.llm import ChatContext, ChatMessage

            async def _bedrock_invoke_json(system_text: str, user_obj: dict) -> dict | None:
                """Call Bedrock (non-stream) to get JSON back. Avoids ConverseStream permissions."""
                try:
                    import boto3

                    model_id = os.getenv("BEDROCK_MODEL", "anthropic.claude-3-5-sonnet-20240620-v1:0")
                    region = os.getenv("BEDROCK_REGION", "us-east-1")

                    def _call():
                        client = boto3.client("bedrock-runtime", region_name=region)
                        body = {
                            "anthropic_version": "bedrock-2023-05-31",
                            "max_tokens": 512,
                            "temperature": 0,
                            "system": system_text,
                            "messages": [
                                {
                                    "role": "user",
                                    "content": [{"type": "text", "text": json.dumps(user_obj)}],
                                }
                            ],
                        }
                        resp = client.invoke_model(
                            modelId=model_id,
                            body=json.dumps(body).encode("utf-8"),
                            contentType="application/json",
                            accept="application/json",
                        )
                        raw_bytes = resp["body"].read()
                        out = json.loads(raw_bytes)
                        # Claude on Bedrock returns content: [{type:"text", text:"..."}]
                        text = ""
                        for part in out.get("content", []) or []:
                            if isinstance(part, dict) and part.get("type") == "text":
                                text += part.get("text", "")
                        text = (text or "").strip()
                        # Strip fences if any
                        text = re.sub(r"^```(?:json)?\\s*", "", text)
                        text = re.sub(r"\\s*```$", "", text)
                        return json.loads(text) if text else None

                    return await asyncio.to_thread(_call)
                except Exception as e:
                    logger.debug(f"⚠️ Bedrock JSON extract failed (ignored): {e}")
                    return None

            lines: List[str] = []
            for item in self._conversation_log[-max_lines:]:
                role = item.get("role", "")
                txt = (item.get("text") or "").strip()
                if not txt:
                    continue
                prefix = "User" if role == "user" else "Agent"
                lines.append(f"{prefix}: {txt}")
            transcript = "\n".join(lines)

            system = (
                "You extract specific CRM fields from a conversation transcript.\n"
                "Return ONLY valid JSON.\n"
                "If a field is unknown, set value to null and confidence to 0.\n"
                "Use confidence 0..1.\n"
            )
            schema = {f: {"value": "string|null", "confidence": "number"} for f in fields}
            user = {
                "task": "extract_fields",
                "fields": fields,
                "schema": schema,
                "transcript": transcript,
                "notes": (
                    "If extracting full_name, ONLY return a person's real name. "
                    "Do not return phrases like 'looking forward'."
                ),
            }

            data = None
            llm_provider = os.getenv("LLM_PROVIDER", "bedrock").lower()
            if llm_provider == "bedrock":
                data = await _bedrock_invoke_json(system, user)
            else:
                # Best-effort for non-bedrock: call plugin LLM via keyword-only arg.
                # NOTE: Many plugin LLMs return a stream; if unsupported, we just skip.
                try:
                    ctx = ChatContext(items=[
                        ChatMessage(role="system", content=[system]),
                        ChatMessage(role="user", content=[json.dumps(user)]),
                    ])
                    stream = self.llm.chat(chat_ctx=ctx)
                    text = ""
                    async for ev in stream:
                        delta = getattr(ev, "delta", None) or getattr(ev, "text", None)
                        if isinstance(delta, str):
                            text += delta
                    raw = (text or "").strip()
                    raw = re.sub(r"^```(?:json)?\\s*", "", raw)
                    raw = re.sub(r"\\s*```$", "", raw)
                    data = json.loads(raw) if raw else None
                except Exception as e:
                    logger.debug(f"⚠️ Plugin LLM extract failed (ignored): {e}")
                    data = None

            if not isinstance(data, dict):
                return

            # Apply patches (only for fields requested)
            for field in fields:
                obj = data.get(field) or {}
                val = obj.get("value")
                conf = float(obj.get("confidence") or 0.0)
                if conf < min_confidence or not isinstance(val, str) or not val.strip():
                    continue
                v = val.strip()

                if field == "full_name":
                    c_l = v.lower()
                    if "looking forward" in c_l:
                        continue
                    if any(ch.isdigit() for ch in v) or "@" in v or len(v.split()) > 4:
                        continue
                    logger.info(f"🧠 Mid-call LLM patch: full_name='{v}' (conf={conf:.2f})")
                    await self.save_collected_data(None, "full_name", v)
                elif field == "email":
                    # Let existing normalization happen in save_collected_data path
                    logger.info(f"🧠 Mid-call LLM patch: email='{v}' (conf={conf:.2f})")
                    await self.save_collected_data(None, "email", v)
                elif field == "phone":
                    logger.info(f"🧠 Mid-call LLM patch: phone='{v}' (conf={conf:.2f})")
                    await self.save_collected_data(None, "phone", v)
                elif field == "company":
                    logger.info(f"🧠 Mid-call LLM patch: company='{v}' (conf={conf:.2f})")
                    await self.save_collected_data(None, "company", v)
                elif field == "account_id":
                    logger.info(f"🧠 Mid-call LLM patch: account_id='{v}' (conf={conf:.2f})")
                    await self.save_collected_data(None, "account_id", v)
                elif field == "company_title":
                    logger.info(f"🧠 Mid-call LLM patch: company_title='{v}' (conf={conf:.2f})")
                    await self.save_collected_data(None, "company_title", v)
        except Exception as e:
            logger.debug(f"⚠️ LLM extract/patch failed (ignored): {e}")
            return
    
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
        
        # Remove known Telnyx prefix variants if present
        if room_name_clean.startswith("telnyx_call__telnyx_call_"):
            # Remove double prefix
            room_name_clean = room_name_clean.replace("telnyx_call__telnyx_call_", "telnyx_call_", 1)
            logger.warning(f"⚠️ Fixed double-prefixed room name: {ctx.room.name} -> {room_name_clean}")
        elif room_name_clean.startswith("telnyx_call__"):
            room_name_clean = room_name_clean.replace("telnyx_call__", "telnyx_call_", 1)
            logger.warning(f"⚠️ Fixed double-underscore room name: {ctx.room.name} -> {room_name_clean}")
        
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

        # Attach LiveKit room event logs (SIP debugging).
        # This is critical because with self-hosted SIP, "transfer succeeded" on Telnyx
        # does NOT guarantee the SIP participant actually joined the LiveKit room.
        try:
            @ctx.room.on("participant_connected")
            def _on_participant_connected(participant: rtc.Participant):
                try:
                    identity = getattr(participant, 'identity', 'unknown')
                    kind = getattr(participant, 'kind', 'unknown')
                    logger.info(
                        f"👤 participant_connected identity={identity}"
                        f" kind={kind}"
                    )
                    
                    # HITL: Detect human agent joining
                    if identity.startswith("human_agent_"):
                        logger.info(f"🙋 Human agent detected: {identity}")
                        if agent._handoff_active:
                            agent._human_agent_identity = identity
                            # Enter shadow mode asynchronously
                            asyncio.create_task(agent._enter_shadow_mode())
                            # Notify Alive5 that human is now live
                            asyncio.create_task(agent._emit_alive5_socket_event("human_agent_joined", {
                                "thread_id": agent.alive5_thread_id or "",
                                "agent_id": identity,
                                "timestamp": int(asyncio.get_event_loop().time() * 1000)
                            }))
                except Exception as e:
                    logger.warning(f"⚠️ Error in participant_connected handler: {e}")
                    logger.info("👤 participant_connected (details unavailable)")

            @ctx.room.on("participant_disconnected")
            def _on_participant_disconnected(participant: rtc.Participant):
                try:
                    identity = getattr(participant, 'identity', 'unknown')
                    kind = getattr(participant, 'kind', 'unknown')
                    logger.info(
                        f"👤 participant_disconnected identity={identity}"
                        f" kind={kind}"
                    )
                    
                    # HITL: Detect human agent leaving
                    if identity.startswith("human_agent_") and agent._human_agent_identity == identity:
                        logger.info(f"👋 Human agent left: {identity}")
                        # Exit shadow mode or end call based on configuration
                        asyncio.create_task(agent._exit_shadow_mode())
                except Exception as e:
                    logger.warning(f"⚠️ Error in participant_disconnected handler: {e}")
                    logger.info("👤 participant_disconnected (details unavailable)")

            @ctx.room.on("track_published")
            def _on_track_published(publication: rtc.RemoteTrackPublication, participant: rtc.RemoteParticipant):
                try:
                    logger.info(
                        f"🎵 track_published by={getattr(participant, 'identity', 'unknown')}"
                        f" kind={getattr(participant, 'kind', 'unknown')}"
                        f" source={getattr(publication, 'source', 'unknown')}"
                        f" name={getattr(publication, 'name', 'unknown')}"
                    )
                except Exception:
                    logger.info("🎵 track_published (details unavailable)")

            @ctx.room.on("track_subscribed")
            def _on_track_subscribed(track: rtc.Track, publication: rtc.RemoteTrackPublication, participant: rtc.RemoteParticipant):
                try:
                    logger.info(
                        f"🎧 track_subscribed by={getattr(participant, 'identity', 'unknown')}"
                        f" kind={getattr(participant, 'kind', 'unknown')}"
                        f" source={getattr(publication, 'source', 'unknown')}"
                        f" name={getattr(publication, 'name', 'unknown')}"
                        f" track_kind={getattr(track, 'kind', 'unknown')}"
                    )
                except Exception:
                    logger.info("🎧 track_subscribed (details unavailable)")

            @ctx.room.on("sip_dtmf_received")
            def _on_sip_dtmf_received(dtmf: rtc.SipDTMF):
                try:
                    logger.info(
                        f"☎️ sip_dtmf_received digit={getattr(dtmf, 'digit', 'unknown')}"
                        f" participant_identity={getattr(getattr(dtmf, 'participant_identity', None), 'identity', getattr(dtmf, 'participant_identity', 'unknown'))}"
                    )
                except Exception:
                    logger.info("☎️ sip_dtmf_received (details unavailable)")
        except Exception as _e:
            logger.debug(f"Could not attach room event handlers: {_e}")

        # High-signal diagnostics: confirm what participants exist in the room at job start.
        # This is especially important for SIP rooms where the caller participant may show up as SIP/INGRESS.
        try:
            rp = getattr(ctx.room, "remote_participants", {}) or {}
            logger.info(
                f"👥 Room participants after connect - local_identity={getattr(getattr(ctx.room, 'local_participant', None), 'identity', 'unknown')}"
                f" | remote_count={len(rp)}"
            )
            for _sid, p in rp.items():
                logger.info(
                    f"   - remote identity={getattr(p, 'identity', 'unknown')}"
                    f" kind={getattr(p, 'kind', 'unknown')}"
                    f" tracks={len(getattr(p, 'tracks', {}) or {})}"
                )
        except Exception as _e:
            logger.debug(f"Could not dump room participants: {_e}")

        # PSTN reliability: if the SIP participant disconnects (caller hung up), proactively end the Alive5 thread.
        # This avoids cases where on_session_end isn't triggered promptly.
        # NOTE: Use an agent_ref to avoid races where the callback fires before agent is instantiated.
        agent_ref = {"agent": None}
        try:
            @ctx.room.on("participant_disconnected")
            def _on_participant_disconnected_end_chat(participant: rtc.Participant):
                try:
                    if not is_phone_call:
                        return
                    # LiveKit Python SDKs sometimes expose participant.kind as an int (e.g. "kind=3" in logs),
                    # while rtc.ParticipantKind values are enum members. Normalize to ints for comparison.
                    kind_raw = getattr(participant, "kind", None)
                    kind_val = None
                    try:
                        if kind_raw is not None:
                            kind_val = int(getattr(kind_raw, "value", kind_raw))
                    except Exception:
                        kind_val = None

                    # Prefer kind match when possible, but also allow identity-based detection.
                    sip_kind_vals = set()
                    try:
                        for _k in (getattr(rtc.ParticipantKind, "SIP", None), getattr(rtc.ParticipantKind, "INGRESS", None)):
                            if _k is None:
                                continue
                            try:
                                sip_kind_vals.add(int(getattr(_k, "value", _k)))
                            except Exception:
                                pass
                    except Exception:
                        pass

                    identity = (getattr(participant, "identity", "") or "").lower()
                    is_sip_like = False
                    if kind_val is not None and sip_kind_vals and kind_val in sip_kind_vals:
                        is_sip_like = True
                    elif identity.startswith("sip_") or identity.startswith("ingress") or "sip" in identity:
                        # Fallback for SDKs that report kind differently.
                        is_sip_like = True

                    if not is_sip_like:
                        return

                    logger.info(
                        f"📞 PSTN participant_disconnected trigger (caller) identity={getattr(participant, 'identity', 'unknown')} kind={kind_val if kind_val is not None else kind_raw}"
                    )

                    async def _end_after_disconnect():
                        try:
                            # Keep this very short to beat backend room deletion on Telnyx hangup.
                            await asyncio.sleep(0.1)
                            a = agent_ref.get("agent")
                            if a is None:
                                return
                            # Wait briefly for Alive5 session identifiers to be present (phone init is async).
                            try:
                                for _ in range(10):  # ~2s max
                                    if getattr(a, "alive5_thread_id", None) and getattr(a, "alive5_channel_id", None) and getattr(a, "alive5_crm_id", None):
                                        break
                                    await asyncio.sleep(0.2)
                            except Exception:
                                pass
                            # Caller hung up
                            try:
                                a._end_by = "person"
                            except Exception:
                                pass
                            await a.on_session_end()
                            # Best-effort: disconnect worker from room (agent side) so the job terminates cleanly.
                            try:
                                await ctx.room.disconnect()
                            except Exception:
                                pass
                        except Exception:
                            pass

                    asyncio.create_task(_end_after_disconnect())
                except Exception:
                    pass
        except Exception as _e:
            logger.debug(f"Could not attach PSTN end-chat handler: {_e}")
        
        # Get Alive5 session data from backend (frontend already called init_livechat)
        # CRITICAL: Frontend calls init_livechat and connects socket, so worker should use that session data
        # Do NOT call init_livechat again - it would create a different thread_id!
        async def get_alive5_session_data():
            """Get Alive5 session data from backend (created by frontend's init_livechat call)"""
            try:
                import httpx
                from urllib.parse import quote
                encoded_room_name = quote(room_name_clean, safe='')
                
                # Poll for session data.
                # Web: frontend calls init_livechat shortly after worker starts.
                # Phone: backend initializes the Alive5 session asynchronously in the Telnyx webhook.
                # Give phone calls a bit more time to avoid false "not ready" failures.
                max_attempts = 30 if is_phone_call else 10
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

                                # PSTN bridge: connect early (no frontend), so init/ACK happens before we emit messages/CRM.
                                if is_phone_call:
                                    try:
                                        asyncio.create_task(agent._ensure_alive5_socket_connected())
                                    except Exception:
                                        pass
                                
                                # Process any queued messages
                                if agent._alive5_message_queue:
                                    queued_messages = agent._alive5_message_queue.copy()
                                    agent._alive5_message_queue.clear()
                                    for queued_content, queued_is_agent in queued_messages:
                                        await agent._send_message_to_alive5_internal(queued_content, queued_is_agent)

                                # Flush any CRM updates that happened before thread_id/crm_id were ready
                                await agent._flush_pending_crm_updates()
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
        try:
            agent_ref["agent"] = agent
        except Exception:
            pass
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
        
        # Optimize STT for phone calls (PSTN audio is typically narrowband 8kHz)
        # Skip STT initialization for Nova Sonic (it's speech-to-speech)
        stt_model = None
        stt_language = "en-US"  # Default to English
        stt_sample_rate = 16000
        stt_endpointing_ms = 25
        stt_no_delay = True
        stt_punctuate = True
        stt_smart_format = False
        stt_filler_words = True
        if not getattr(agent, '_using_nova', False):
            def _clean_env_token(v: str) -> str:
                """
                Systemd EnvironmentFile / .env files often include inline comments like:
                  PHONE_DEEPGRAM_STT_MODEL=nova  # faster for phone
                Those comments can get treated as part of the value and break provider APIs.
                """
                if not v:
                    return v
                # Strip inline comments and surrounding whitespace
                v = v.split("#", 1)[0].strip()
                # Keep only the first whitespace-separated token (defensive)
                v = v.split()[0].strip() if v.split() else v.strip()
                return v

            stt_model = _clean_env_token(os.getenv("DEEPGRAM_STT_MODEL", "nova-2"))
            if is_phone_call:
                # For phone calls, check if we should use a faster STT model
                phone_stt_model = _clean_env_token(os.getenv("PHONE_DEEPGRAM_STT_MODEL", stt_model))
                if phone_stt_model != stt_model:
                    stt_model = phone_stt_model
                    logger.info(f"📞 Phone call detected - using optimized STT model: {stt_model}")

                # Phone calls are usually 8kHz; matching sample rate improves accuracy.
                # Allow override via env for providers/codecs that deliver 16kHz.
                try:
                    stt_sample_rate = int(_clean_env_token(os.getenv("PHONE_DEEPGRAM_SAMPLE_RATE", "8000")))
                except Exception:
                    stt_sample_rate = 8000

                # Slightly less aggressive endpointing reduces chopped/garbled words on phone audio.
                try:
                    stt_endpointing_ms = int(_clean_env_token(os.getenv("PHONE_DEEPGRAM_ENDPOINTING_MS", "120")))
                except Exception:
                    stt_endpointing_ms = 120

                stt_no_delay = (_clean_env_token(os.getenv("PHONE_DEEPGRAM_NO_DELAY", "true")) or "true").lower() == "true"
                stt_punctuate = (_clean_env_token(os.getenv("PHONE_DEEPGRAM_PUNCTUATE", "true")) or "true").lower() == "true"
                stt_smart_format = (_clean_env_token(os.getenv("PHONE_DEEPGRAM_SMART_FORMAT", "true")) or "true").lower() == "true"
                stt_filler_words = (_clean_env_token(os.getenv("PHONE_DEEPGRAM_FILLER_WORDS", "false")) or "false").lower() == "true"
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
                stt=(
                    deepgram.STT(
                        model=stt_model,
                        language=stt_language,
                        api_key=os.getenv("DEEPGRAM_API_KEY"),
                        sample_rate=stt_sample_rate,
                        endpointing_ms=stt_endpointing_ms,
                        no_delay=stt_no_delay,
                        punctuate=stt_punctuate,
                        smart_format=stt_smart_format,
                        filler_words=stt_filler_words,
                    )
                    if stt_model
                    else None
                ),
                llm=agent.llm,  # LLM instance (OpenAI, Bedrock, etc.)
                tts=tts,
                vad=vad,
                turn_detection=turn_detector  # Use LiveKit turn detector for better end-of-turn detection
            )
        
        # Set the agent's room and session
        agent.room = ctx.room
        agent.agent_session = session
        
        # Start the session - with environment variable control for testing
        #
        # IMPORTANT (Self-hosted LiveKit):
        # LiveKit "audio filters" (noise cancellation/BVC/krisp) are a LiveKit Cloud feature.
        # Enabling them against a self-hosted server results in:
        #   - "failed to fetch server settings: http status: 404"
        #   - "audio filter cannot be enabled: LiveKit Cloud is required"
        #
        # So we keep them OFF unless explicitly enabled (and you're using LiveKit Cloud).
        # IMPORTANT (Self-hosted LiveKit):
        # Do NOT enable LiveKit Cloud audio filters from the agent (noise cancellation/BVC/krisp).
        # They trigger:
        #   - "failed to fetch server settings: http status: 404"
        #   - "audio filter cannot be enabled: LiveKit Cloud is required"
        #
        # If you ever move to LiveKit Cloud, we can re-enable this behind a flag.
        logger.info("🚫 Noise cancellation disabled (self-hosted LiveKit)")

        # Explicitly enable audio I/O.
        # We set this explicitly because SIP calls can otherwise look like "agent not joining"
        # (job runs, but no audio published/subscribed).
        #
        # Also disable pre-connect audio to avoid any early feature negotiation that can trip cloud-only paths.
        audio_sample_rate = 8000 if is_phone_call else 24000

        if is_phone_call:
            # SIP trunks may show up as SIP or INGRESS participants.
            sip_kinds = [
                rtc.ParticipantKind.Value("PARTICIPANT_KIND_SIP"),
                rtc.ParticipantKind.Value("PARTICIPANT_KIND_INGRESS"),
                rtc.ParticipantKind.Value("PARTICIPANT_KIND_STANDARD"),
            ]
            room_input_options = RoomInputOptions(
                text_enabled=True,
                audio_enabled=True,
                video_enabled=False,
                audio_sample_rate=audio_sample_rate,
                audio_num_channels=1,
                participant_kinds=sip_kinds,
                pre_connect_audio=False,
            )
        else:
            room_input_options = RoomInputOptions(
                text_enabled=True,
                audio_enabled=True,
                video_enabled=False,
                audio_sample_rate=audio_sample_rate,
                audio_num_channels=1,
                pre_connect_audio=False,
            )
        
        await session.start(
            room=ctx.room,
            agent=agent,
            room_input_options=room_input_options,
            room_output_options=RoomOutputOptions(
                transcription_enabled=True,
                audio_enabled=True,
                audio_sample_rate=audio_sample_rate,
                audio_num_channels=1,
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
                        # During HITL/shadow mode, suppress AI messages so Alive5 doesn't attribute them to the voice agent.
                        if getattr(agent, "_handoff_active", False):
                            return
                        # Silently send agent message (logging happens in _send_message_to_alive5_internal)
                        asyncio.create_task(agent._send_message_to_alive5(text_content, is_agent=True))
                        # Append to transcript (for post-call reconciliation)
                        try:
                            agent._conversation_log.append({"role": "assistant", "text": text_content.strip()})
                        except Exception:
                            pass
                else:
                    logger.warning(f"⚠️ conversation_item_added called but text_content is empty (role: {getattr(chat_message, 'role', 'unknown')})")
            except Exception as e:
                logger.warning(f"⚠️ Error in conversation_item_added (isolated): {e}", exc_info=True)
        
        # Start the conversation with greeting
        # Note: For Nova Sonic, greeting is skipped (waits for user input first)
        # Check if room is still connected before starting greeting
        try:
            # For phone calls, wait briefly for the SIP participant to actually join before greeting.
            # Otherwise the agent may "speak into an empty room" and the caller hears nothing.
            if is_phone_call:
                try:
                    import time as _time
                    deadline = _time.monotonic() + float(os.getenv("PHONE_WAIT_FOR_PARTICIPANT_SECONDS", "12"))
                    while _time.monotonic() < deadline:
                        rp = getattr(ctx.room, "remote_participants", {}) or {}
                        if len(rp) > 0:
                            break
                        await asyncio.sleep(0.25)
                    rp = getattr(ctx.room, "remote_participants", {}) or {}
                    logger.info(f"📞 Phone participant wait complete - remote_count={len(rp)}")
                except Exception as _e:
                    logger.debug(f"Phone participant wait failed: {_e}")

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
            # Use LIVEKIT_WORKER_URL if set (for localhost on same server), otherwise use LIVEKIT_URL
            livekit_url = os.getenv("LIVEKIT_WORKER_URL") or os.getenv("LIVEKIT_URL")
            livekit_api_key = os.getenv("LIVEKIT_API_KEY")
            livekit_api_secret = os.getenv("LIVEKIT_API_SECRET")
            
            if not livekit_url:
                logger.error("❌ LIVEKIT_URL or LIVEKIT_WORKER_URL not set in environment!")
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
            
            # CRITICAL: Set LIVEKIT_URL environment variable for LiveKit SDK
            # The SDK reads LIVEKIT_URL directly, not LIVEKIT_WORKER_URL
            os.environ["LIVEKIT_URL"] = livekit_url
            
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