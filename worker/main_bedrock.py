import logging
import os
import uuid
import asyncio
from datetime import datetime
from dotenv import load_dotenv
from typing import Dict, Any, Optional
import json
import re
import boto3

from livekit.agents import (
    AgentSession,
    Agent,
    JobContext,
    WorkerOptions,
    cli,
    RoomInputOptions,
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

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("bedrock-voice-agent")

# Verify environment variables
required_vars = [
    "OPENAI_API_KEY", "DEEPGRAM_API_KEY", "CARTESIA_API_KEY", 
    "LIVEKIT_URL", "LIVEKIT_API_KEY", "LIVEKIT_API_SECRET",
    "AWS_ACCESS_KEY_ID", "AWS_SECRET_ACCESS_KEY", "AWS_REGION"
]
for var in required_vars:
    if not os.getenv(var):
        logger.error(f"Missing required environment variable: {var}")
    else:
        # Don't log full API keys for security
        value = os.getenv(var)
        masked_value = value[:5] + "*" * (len(value) - 5) if len(value) > 5 else "*****"
        logger.info(f"Loaded {var}: {masked_value}")

# Initialize AWS Bedrock client
try:
    bedrock_runtime = boto3.client(
        service_name="bedrock-runtime",
        region_name=os.getenv("AWS_REGION"),
        aws_access_key_id=os.getenv("AWS_ACCESS_KEY_ID"),
        aws_secret_access_key=os.getenv("AWS_SECRET_ACCESS_KEY"),
    )
    logger.info("AWS Bedrock client initialized successfully")
except Exception as e:
    logger.error(f"Failed to initialize AWS Bedrock client: {e}")
    bedrock_runtime = None

# Global session tracking
active_sessions = {}

# Data extraction logging
def log_data_extraction(session_id: str, data: Dict[str, Any], source: str = "unknown"):
    """Log extracted data to a dedicated file and console"""
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    # Create data extraction log entry
    log_entry = f"""
=== DATA EXTRACTION LOG ===
Timestamp: {timestamp}
Session ID: {session_id}
Source: {source}
Extracted Data: {data}
========================
"""
    
    # Write to dedicated data extraction log file
    try:
        with open("data_extraction.log", "a") as f:
            f.write(log_entry)
    except Exception as e:
        logger.error(f"Failed to write to data extraction log: {e}")
    
    # Log to console for immediate visibility
    logger.info(f"🔍 DATA EXTRACTED - Session {session_id}: {data}")
    
    # Also log to the main log file
    logger.info(f"Data extraction completed for session {session_id}: {data}")

# Intent descriptions for LLM-based detection
INTENT_DESCRIPTIONS = {
    "sales": "Questions about pricing, plans, demos, buying, or team licenses. Examples include questions about cost, plans, packages, upgrades, or setting up a meeting with sales.",
    "support": "Technical issues, troubleshooting help, or how-to questions. Examples include problems with installation, errors, bugs, or requests for setup guides.",
    "billing": "Questions about invoices, payments, account management, or subscription changes. Examples include billing issues, refund requests, or payment method updates."
}

# Legacy intent patterns for fallback (optional)
LEGACY_INTENT_PATTERNS = {
    "sales": [
        r"(?i)\b(pricing|cost|plan|demo|meeting|sales|purchase|buy|upgrade|package|starter|pro|enterprise)\b",
        r"(?i)\b(how much|price|pricing|what does it cost|interested in buying)\b",
        r"(?i)\b(team size|users|licenses|subscription)\b"
    ],
    "support": [
        r"(?i)\b(help|support|problem|issue|bug|error|not working|broken|setup|install|troubleshoot)\b",
        r"(?i)\b(can't|cannot|won't|doesn't work|failed|failing)\b",
        r"(?i)\b(how to|how do I|tutorial|guide|documentation)\b"
    ],
    "billing": [
        r"(?i)\b(billing|invoice|payment|charge|refund|account|subscription|cancel)\b",
        r"(?i)\b(charged|billed|paid|credit card|payment method)\b",
        r"(?i)\b(account settings|billing info|payment info)\b"
    ]
}

# Knowledge base for support
SUPPORT_KB = {
    "setup": {
        "question": "How do I install Alive5?",
        "answer": "To install Alive5, add our JavaScript snippet to your website's head tag. I can walk you through the process step by step."
    },
    "chat_issues": {
        "question": "Chat widget not loading",
        "answer": "Try clearing your browser cache or switching devices. If chats still don't load, your plan might need an upgrade for dedicated servers."
    },
    "account_settings": {
        "question": "How to manage team permissions",
        "answer": "Admins can adjust team permissions under 'Settings' in your Alive5 dashboard. Would you like me to guide you through it?"
    }
}

class DynamicAssistant(Agent):
    def __init__(self, session_id: str) -> None:
        self.session_id = session_id
        self.detected_intent = None
        self.user_data = {}
        self.conversation_stage = "greeting"
        
        super().__init__(instructions=self._get_dynamic_instructions())

    def _get_dynamic_instructions(self) -> str:
        """Generate dynamic instructions based on detected intent"""
        base_instructions = f"""
You are Scott, the AI voice assistant for Alive5 Support (Session: {self.session_id}). 

Your primary goal is to detect user intent and provide appropriate assistance:
- SALES: Route qualified leads and gather contact information
- SUPPORT: Resolve technical issues using knowledge base
- BILLING: Handle account and billing inquiries

CONVERSATION FLOW:
1. Start with a warm greeting
2. Listen carefully to detect user intent
3. Route to appropriate specialized assistance
4. Collect necessary information
5. Provide resolution or escalation

TONE: Professional, helpful, empathetic, and conversational
VOICE: Clear, warm, engaging
LANGUAGE: Simple, jargon-free

Current conversation stage: {self.conversation_stage}
Detected intent: {self.detected_intent or 'Not yet determined'}
"""

        # Add intent-specific instructions
        if self.detected_intent == "sales":
            base_instructions += """
            
SALES INTENT ACTIVE:
- Ask about team size (1-5, 6-20, 21+)
- Suggest appropriate plan (Starter $29/mo, Pro $79/mo, Enterprise custom)
- Collect: name, email, company
- Determine if lead is HOT (ready to buy, has budget, decision maker)
- If HOT: offer immediate transfer to sales team
- If not ready: schedule demo meeting and send email with demo link
- Keep conversation focused on qualifying the lead
"""
        elif self.detected_intent == "support":
            base_instructions += """
            
SUPPORT INTENT ACTIVE:
- Identify specific technical issue
- Use knowledge base to provide solutions
- Walk through troubleshooting steps
- If unresolved, offer escalation to technical team
- Common issues: setup, widget not loading, account settings
- Always confirm understanding before moving to next step
"""
        elif self.detected_intent == "billing":
            base_instructions += """
            
BILLING INTENT ACTIVE:
- Gather account details (email, company name)
- Address billing questions using standard responses
- Issues: payment problems, subscription changes, refunds
- Collect contact information for follow-up
- Escalate complex billing issues to billing department
"""

        base_instructions += """

IMPORTANT RULES:
- Always confirm you're listening when asked
- If intent changes mid-conversation, adapt accordingly
- End calls gracefully when resolution is achieved
- Never make promises about pricing or technical capabilities you're unsure about
- Be honest about limitations and offer human escalation when needed
"""

        return base_instructions

    async def detect_intent_with_bedrock(self, user_message: str) -> Optional[str]:
        """Detect user intent using Amazon Bedrock LLM"""
        if not user_message or user_message.strip() == "":
            return None
        
        if not bedrock_runtime:
            logger.error("Bedrock client not initialized. Falling back to pattern matching.")
            return self.detect_intent_with_patterns(user_message)
            
        try:
            # Format prompt for intent detection
            prompt = f"""
You are an intent classifier for a customer service AI.
Classify the following user message into exactly one of these intents:
- sales: {INTENT_DESCRIPTIONS["sales"]}
- support: {INTENT_DESCRIPTIONS["support"]}
- billing: {INTENT_DESCRIPTIONS["billing"]}

User message: "{user_message}"

Respond with ONLY one word (the intent): sales, support, or billing.
"""
            
            # Prepare request for Anthropic Claude model (Recommended for classification tasks)
            model_id = "anthropic.claude-3-sonnet-20240229-v1:0"  # Choose the appropriate model
            body = json.dumps({
                "anthropic_version": "bedrock-2023-05-31",
                "max_tokens": 10,
                "temperature": 0.0,
                "messages": [
                    {
                        "role": "user",
                        "content": prompt
                    }
                ]
            })
            
            # Call Bedrock API
            response = bedrock_runtime.invoke_model(
                modelId=model_id,
                body=body
            )
            
            # Parse response
            response_body = json.loads(response.get('body').read())
            detected_intent = response_body.get('content', [{}])[0].get('text', '').strip().lower()
            
            # Validate that the response is one of our expected intents
            if detected_intent in ["sales", "support", "billing"]:
                logger.info(f"\nBedrock Intent detected: {detected_intent} from message: '{user_message[:50]}...'\n")
                return detected_intent
            else:
                logger.warning(f"Bedrock returned unexpected intent: {detected_intent}. Falling back to pattern matching.")
        except Exception as e:
            logger.error(f"Error using Bedrock for intent detection: {e}")
            logger.warning("Falling back to pattern matching for intent detection")
        
        # Fallback to regex pattern matching if Bedrock fails
        return self.detect_intent_with_patterns(user_message)
    
    def detect_intent_with_patterns(self, user_message: str) -> Optional[str]:
        """Legacy pattern-based intent detection as fallback"""
        user_message_lower = user_message.lower()
        
        # Check each intent pattern
        for intent, patterns in LEGACY_INTENT_PATTERNS.items():
            for pattern in patterns:
                if re.search(pattern, user_message_lower):
                    logger.info(f"\nPattern Intent detected: {intent} from message: '{user_message[:50]}...'\n")
                    return intent
        
        return None
        
    def detect_intent(self, user_message: str) -> Optional[str]:
        """Detect user intent from their message (synchronous wrapper)"""
        # Create a new event loop for async function if needed
        try:
            loop = asyncio.get_event_loop()
            if loop.is_running():
                # Use a thread to run the async function if we're already in an event loop
                import concurrent.futures
                with concurrent.futures.ThreadPoolExecutor() as executor:
                    future = executor.submit(
                        asyncio.run, 
                        self.detect_intent_with_bedrock(user_message)
                    )
                    return future.result()
            else:
                # If no event loop is running, we can just run the async function
                return asyncio.run(self.detect_intent_with_bedrock(user_message))
        except Exception as e:
            logger.error(f"Error in async intent detection: {e}")
            # Fall back to pattern matching if async fails
            return self.detect_intent_with_patterns(user_message)

    def update_conversation_context(self, intent: str, user_message: str):
        """Update agent's context based on detected intent and user message"""
        if self.detected_intent != intent:
            self.detected_intent = intent
            logger.info(f"Session {self.session_id}: Intent changed to {intent}")
            
            # Update instructions dynamically
            self.instructions = self._get_dynamic_instructions()

    def extract_user_info(self, message: str):
        """Extract user information from messages"""
        original_data = self.user_data.copy()
        
        # Extract email
        email_pattern = r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b'
        emails = re.findall(email_pattern, message)
        if emails:
            self.user_data['email'] = emails[0]
            
        # Extract phone numbers
        phone_pattern = r'\b\d{3}[-.]?\d{3}[-.]?\d{4}\b'
        phones = re.findall(phone_pattern, message)
        if phones:
            self.user_data['phone'] = phones[0]
            
        # Extract names (simple heuristic)
        name_indicators = ['my name is', "i'm", 'this is', 'name:', 'i am']
        for indicator in name_indicators:
            if indicator in message.lower():
                # Extract potential name after indicator
                parts = message.lower().split(indicator)
                if len(parts) > 1:
                    potential_name = parts[1].strip().split()[0]
                    if len(potential_name) > 1:
                        self.user_data['name'] = potential_name.title()
                break
        
        # Log extracted data if any new data was found
        if self.user_data != original_data:
            new_data = {k: v for k, v in self.user_data.items() if k not in original_data or original_data[k] != v}
            if new_data:
                log_data_extraction(self.session_id, new_data, f"message: '{message}'")

def prewarm(proc):
    """Preload models for better performance"""
    proc.userdata["vad"] = silero.VAD.load()

async def entrypoint(ctx: JobContext):
    """Main entry point for the Bedrock voice agent"""
    
    # Generate unique session identifier
    session_id = str(uuid.uuid4())[:8]
    room_name = ctx.room.name
    
    # Check if this room already has an active agent
    if room_name in active_sessions:
        logger.warning(f"Room {room_name} already has an active session. Skipping.")
        return
    
    # Mark this room as having an active session
    active_sessions[room_name] = session_id
    logger.info(f"Starting Bedrock agent session {session_id} in room: {room_name}")
    
    agent_session = None
    assistant = DynamicAssistant(session_id)
    
    try:
        # Connect to the room with auto-subscribe to audio only
        await ctx.connect(auto_subscribe=AutoSubscribe.AUDIO_ONLY)
        logger.info(f"Connected to room {room_name}")
        
        # Wait for a participant to join
        try:
            participant = await asyncio.wait_for(ctx.wait_for_participant(), timeout=600.0)
            logger.info(f"Participant {participant.identity} joined session {session_id}")
            
            # Give a moment for the participant to fully connect
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
                voice="a0e99841-438c-4a64-b679-ae501e7d6091",  # British lady voice
                api_key=os.getenv("CARTESIA_API_KEY")
            ),
            vad=ctx.proc.userdata["vad"],
            turn_detection=MultilingualModel(),
        )
        
        # Custom message handler for dynamic intent detection
        async def on_user_speech_async(message: str):
            logger.info(f"🔍 DEBUG: on_user_speech_async called with message: '{message}'")
            
            # Detect intent from user message
            detected_intent = assistant.detect_intent(message)
            if detected_intent:
                assistant.update_conversation_context(detected_intent, message)
                logger.info(f"Session {session_id}: Intent detected as {detected_intent}")
            
            # Extract user information
            logger.info(f"🔍 DEBUG: Calling extract_user_info for message: '{message}'")
            assistant.extract_user_info(message)
            logger.info(f"🔍 DEBUG: Current user_data after extraction: {assistant.user_data}")
            
            # Log collected user data
            if assistant.user_data:
                logger.info(f"Session {session_id}: Collected user data: {assistant.user_data}")
        
        # Wrap async function in synchronous callback
        def on_user_speech_sync(message: str):
            logger.info(f"🔍 DEBUG: on_user_speech_sync called with message: '{message}'")
            asyncio.create_task(on_user_speech_async(message))
        
        # Try multiple event names to see which one works
        logger.info("🔍 DEBUG: Registering event handlers for multiple events")
        
        # Register handlers for various event names to ensure we catch user messages
        try:
            agent_session.on("user_speech", on_user_speech_sync)
            agent_session.on("transcript", on_user_speech_sync)
            agent_session.on("message", on_user_speech_sync)
            logger.info("🔍 DEBUG: Registered handlers for various events")
        except Exception as e:
            logger.error(f"🔍 DEBUG: Failed to register event handlers: {e}")
        
        # Start the session
        await agent_session.start(
            room=ctx.room,
            agent=assistant,
            room_input_options=RoomInputOptions(
                noise_cancellation=noise_cancellation.BVC(),
            ),
        )
        
        logger.info(f"Bedrock agent session started for {session_id}")
        
        # Initial greeting
        greeting_message = "Hello! I'm Scott from Alive5. How can I help you today? Are you looking for sales information, technical support, or have questions about billing?"
        
        await agent_session.generate_reply(
            instructions=f"Say exactly: '{greeting_message}'"
        )
        
        logger.info(f"Initial greeting sent for session {session_id}")
        
        # Keep session alive while participants are connected
        while ctx.room.remote_participants:
            await asyncio.sleep(1)
        
    except Exception as e:
        logger.error(f"Error in session {session_id}: {str(e)}")
        raise
    finally:
        # Cleanup: Remove from active sessions and properly close
        if room_name in active_sessions:
            del active_sessions[room_name]
        
        # Log final session summary
        if assistant.user_data or assistant.detected_intent:
            logger.info(f"Session {session_id} summary - Intent: {assistant.detected_intent}, Data: {assistant.user_data}")
        
        if agent_session:
            try:
                await agent_session.aclose()
                logger.info(f"Agent session {session_id} closed properly")
            except Exception as e:
                logger.error(f"Error closing session {session_id}: {e}")
        
        # Disconnect from room
        if ctx.room and ctx.room.connection_state == "connected":
            await ctx.room.disconnect()
            
        logger.info(f"Bedrock session {session_id} ended and cleaned up")

if __name__ == "__main__":
    cli.run_app(
        WorkerOptions(
            entrypoint_fnc=entrypoint,
            prewarm_fnc=prewarm,
        ),
    )
