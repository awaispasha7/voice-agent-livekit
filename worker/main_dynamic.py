import logging
import os
import uuid
import asyncio
from datetime import datetime
from dotenv import load_dotenv
from typing import Dict, Any, Optional
import json
import re

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
logger = logging.getLogger("dynamic-voice-agent")

# Verify environment variables
required_vars = ["OPENAI_API_KEY", "DEEPGRAM_API_KEY", "CARTESIA_API_KEY", "LIVEKIT_URL", "LIVEKIT_API_KEY", "LIVEKIT_API_SECRET"]
for var in required_vars:
    if not os.getenv(var):
        logger.error(f"Missing required environment variable: {var}")
    else:
        logger.info(f"Loaded {var}: {os.getenv(var)[:10]}...")

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

# Intent detection patterns
INTENT_PATTERNS = {
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

    def detect_intent(self, user_message: str) -> Optional[str]:
        """Detect user intent from their message"""
        user_message_lower = user_message.lower()
        
        # Check each intent pattern
        for intent, patterns in INTENT_PATTERNS.items():
            for pattern in patterns:
                if re.search(pattern, user_message_lower):
                    logger.info(f"\nIntent detected: {intent} from message: '{user_message[:50]}...'\n")
                    return intent
        
        return None

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
    """Main entry point for the dynamic voice agent"""
    
    # Generate unique session identifier
    session_id = str(uuid.uuid4())[:8]
    room_name = ctx.room.name
    
    # Check if this room already has an active agent
    if room_name in active_sessions:
        logger.warning(f"Room {room_name} already has an active session. Skipping.")
        return
    
    # Mark this room as having an active session
    active_sessions[room_name] = session_id
    logger.info(f"Starting dynamic agent session {session_id} in room: {room_name}")
    
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
        
        # Method 1: Try the original event name
        try:
            agent_session.on("user_speech", on_user_speech_sync)
            logger.info("🔍 DEBUG: Registered handler for 'user_speech' event")
        except Exception as e:
            logger.error(f"🔍 DEBUG: Failed to register 'user_speech' handler: {e}")
        
        # Method 2: Try alternative event names
        try:
            agent_session.on("transcript", on_user_speech_sync)
            logger.info("🔍 DEBUG: Registered handler for 'transcript' event")
        except Exception as e:
            logger.error(f"🔍 DEBUG: Failed to register 'transcript' handler: {e}")
        
        # Method 3: Try message event
        try:
            agent_session.on("message", on_user_speech_sync)
            logger.info("🔍 DEBUG: Registered handler for 'message' event")
        except Exception as e:
            logger.error(f"🔍 DEBUG: Failed to register 'message' handler: {e}")
        
        # Method 4: Add a general event listener to see what events are available
        def on_any_event(event_name: str, *args):
            logger.info(f"🔍 DEBUG: Event received: {event_name} with args: {args}")
            if event_name in ["user_speech", "transcript", "message"] and args:
                on_user_speech_sync(args[0])
        
        try:
            agent_session.on("*", on_any_event)
            logger.info("🔍 DEBUG: Registered general event listener")
        except Exception as e:
            logger.error(f"🔍 DEBUG: Failed to register general event listener: {e}")
        
        # Method 5: Direct extraction from conversation context
        # Since events aren't working, let's extract data directly from the conversation
        def extract_from_conversation():
            try:
                # Get the conversation history from the agent session
                if hasattr(agent_session, 'llm') and hasattr(agent_session.llm, '_conversation'):
                    conversation = agent_session.llm._conversation
                    for msg in conversation:
                        if msg.get('role') == 'user':
                            user_message = msg.get('content', '')
                            logger.info(f"🔍 DEBUG: Found user message in conversation: '{user_message}'")
                            
                            # Extract user information
                            assistant.extract_user_info(user_message)
                            
                            # Log collected user data
                            if assistant.user_data:
                                logger.info(f"Session {session_id}: Collected user data: {assistant.user_data}")
            except Exception as e:
                logger.debug(f"Error in direct extraction: {e}")
        
        # Method 6: Direct transcript interception
        # Intercept user transcripts directly as they come in
        original_process_transcript = agent_session.process_transcript if hasattr(agent_session, 'process_transcript') else None
        
        def process_transcript_with_extraction(transcript: str, *args, **kwargs):
            logger.info(f"🔍 DEBUG: Intercepted transcript: '{transcript}'")
            
            # Extract user information from transcript
            assistant.extract_user_info(transcript)
            
            # Log collected user data
            if assistant.user_data:
                logger.info(f"Session {session_id}: Collected user data: {assistant.user_data}")
            
            # Call original method if it exists
            if original_process_transcript:
                return original_process_transcript(transcript, *args, **kwargs)
        
        # Try to replace the transcript processing method
        try:
            agent_session.process_transcript = process_transcript_with_extraction
            logger.info("🔍 DEBUG: Successfully replaced process_transcript method")
        except Exception as e:
            logger.error(f"🔍 DEBUG: Failed to replace process_transcript: {e}")
        
        # Method 7: Direct message interception in the conversation loop
        # Override the message handling in the conversation
        original_handle_message = agent_session.handle_message if hasattr(agent_session, 'handle_message') else None
        
        def handle_message_with_extraction(message: str, *args, **kwargs):
            logger.info(f"🔍 DEBUG: Intercepted message: '{message}'")
            
            # Extract user information from message
            assistant.extract_user_info(message)
            
            # Log collected user data
            if assistant.user_data:
                logger.info(f"Session {session_id}: Collected user data: {assistant.user_data}")
            
            # Call original method if it exists
            if original_handle_message:
                return original_handle_message(message, *args, **kwargs)
        
        # Try to replace the message handling method
        try:
            agent_session.handle_message = handle_message_with_extraction
            logger.info("🔍 DEBUG: Successfully replaced handle_message method")
        except Exception as e:
            logger.error(f"🔍 DEBUG: Failed to replace handle_message: {e}")
        
        # Call extraction after each LLM interaction
        original_generate_reply = agent_session.generate_reply
        
        async def generate_reply_with_extraction(*args, **kwargs):
            result = await original_generate_reply(*args, **kwargs)
            extract_from_conversation()
            return result
        
        agent_session.generate_reply = generate_reply_with_extraction
        
        # Start the session
        await agent_session.start(
            room=ctx.room,
            agent=assistant,
            room_input_options=RoomInputOptions(
                noise_cancellation=noise_cancellation.BVC(),
            ),
        )
        
        logger.info(f"Dynamic agent session started for {session_id}")
        
        # Initial greeting
        greeting_message = "Hello! I'm Scott from Alive5. How can I help you today? Are you looking for sales information, technical support, or have questions about billing?"
        
        await agent_session.generate_reply(
            instructions=f"Say exactly: '{greeting_message}'"
        )
        
        logger.info(f"Initial greeting sent for session {session_id}")
        
        # Keep session alive while participants are connected
        while ctx.room.remote_participants:
            await asyncio.sleep(1)
            
            # Log session status periodically
            if hasattr(assistant, 'detected_intent') and assistant.detected_intent:
                logger.debug(f"Session {session_id} active - Intent: {assistant.detected_intent}")
            
            # Check for new user messages in the conversation context
            # This is a workaround since event handlers aren't working
            try:
                # Get the latest conversation messages
                if hasattr(agent_session, '_conversation') and agent_session._conversation:
                    latest_messages = agent_session._conversation[-5:]  # Get last 5 messages
                    for msg in latest_messages:
                        if msg.get('role') == 'user' and not msg.get('processed_for_extraction'):
                            user_message = msg.get('content', '')
                            logger.info(f"🔍 DEBUG: Processing user message for extraction: '{user_message}'")
                            
                            # Extract user information
                            assistant.extract_user_info(user_message)
                            
                            # Log collected user data
                            if assistant.user_data:
                                logger.info(f"Session {session_id}: Collected user data: {assistant.user_data}")
                            
                            # Mark as processed
                            msg['processed_for_extraction'] = True
            except Exception as e:
                logger.debug(f"Error in data extraction loop: {e}")
                pass
        
        # Method 8: Simple transcript monitoring
        # Monitor the user transcripts directly from the debug logs
        def monitor_user_transcripts():
            try:
                # Get the latest user transcripts from the agent session
                if hasattr(agent_session, '_user_transcripts'):
                    for transcript in agent_session._user_transcripts:
                        if not transcript.get('processed_for_extraction'):
                            user_message = transcript.get('text', '')
                            logger.info(f"🔍 DEBUG: Processing transcript for extraction: '{user_message}'")
                            
                            # Extract user information
                            assistant.extract_user_info(user_message)
                            
                            # Log collected user data
                            if assistant.user_data:
                                logger.info(f"Session {session_id}: Collected user data: {assistant.user_data}")
                            
                            # Mark as processed
                            transcript['processed_for_extraction'] = True
            except Exception as e:
                logger.debug(f"Error in transcript monitoring: {e}")
        
        # Method 9: Direct conversation monitoring
        # Monitor the conversation messages directly
        def monitor_conversation_messages():
            try:
                # Get the conversation from the agent session
                if hasattr(agent_session, 'conversation'):
                    for msg in agent_session.conversation:
                        if msg.get('role') == 'user' and not msg.get('processed_for_extraction'):
                            user_message = msg.get('content', '')
                            logger.info(f"🔍 DEBUG: Processing conversation message: '{user_message}'")
                            
                            # Extract user information
                            assistant.extract_user_info(user_message)
                            
                            # Log collected user data
                            if assistant.user_data:
                                logger.info(f"Session {session_id}: Collected user data: {assistant.user_data}")
                            
                            # Mark as processed
                            msg['processed_for_extraction'] = True
            except Exception as e:
                logger.debug(f"Error in conversation monitoring: {e}")
        
        # Add monitoring to the main loop
        async def monitor_and_extract():
            while True:
                try:
                    monitor_user_transcripts()
                    monitor_conversation_messages()
                    await asyncio.sleep(1)  # Check every second
                except Exception as e:
                    logger.debug(f"Error in monitoring loop: {e}")
                    await asyncio.sleep(1)
        
        # Start monitoring in background
        asyncio.create_task(monitor_and_extract())
        logger.info("🔍 DEBUG: Started background monitoring for data extraction")
        
        # Method 10: Direct transcript processing from LiveKit logs
        # Process transcripts directly as they appear in the debug logs
        def process_livekit_transcript(transcript: str):
            logger.info(f"🔍 DEBUG: Processing LiveKit transcript: '{transcript}'")
            
            # Extract user information from transcript
            assistant.extract_user_info(transcript)
            
            # Log collected user data
            if assistant.user_data:
                logger.info(f"Session {session_id}: Collected user data: {assistant.user_data}")
        
        # Method 11: Override the transcript processing at the LiveKit level
        # Try to intercept at the LiveKit agents level
        try:
            # Get the original transcript processor
            if hasattr(agent_session, '_transcript_processor'):
                original_processor = agent_session._transcript_processor
                
                def new_transcript_processor(transcript: str, *args, **kwargs):
                    logger.info(f"🔍 DEBUG: Intercepted at transcript processor: '{transcript}'")
                    process_livekit_transcript(transcript)
                    return original_processor(transcript, *args, **kwargs)
                
                agent_session._transcript_processor = new_transcript_processor
                logger.info("🔍 DEBUG: Successfully replaced _transcript_processor")
        except Exception as e:
            logger.error(f"🔍 DEBUG: Failed to replace _transcript_processor: {e}")
        
        # Method 12: Direct message queue monitoring
        # Monitor the message queue directly
        def monitor_message_queue():
            try:
                if hasattr(agent_session, '_message_queue'):
                    for msg in agent_session._message_queue:
                        if msg.get('type') == 'transcript' and not msg.get('processed_for_extraction'):
                            transcript = msg.get('content', '')
                            logger.info(f"🔍 DEBUG: Found transcript in message queue: '{transcript}'")
                            process_livekit_transcript(transcript)
                            msg['processed_for_extraction'] = True
            except Exception as e:
                logger.debug(f"Error in message queue monitoring: {e}")
        
        # Method 13: Direct stream monitoring
        # Monitor the audio stream directly
        def monitor_audio_stream():
            try:
                if hasattr(agent_session, '_audio_stream'):
                    # Process any pending transcripts in the audio stream
                    if hasattr(agent_session._audio_stream, '_pending_transcripts'):
                        for transcript in agent_session._audio_stream._pending_transcripts:
                            if not transcript.get('processed_for_extraction'):
                                text = transcript.get('text', '')
                                logger.info(f"🔍 DEBUG: Found transcript in audio stream: '{text}'")
                                process_livekit_transcript(text)
                                transcript['processed_for_extraction'] = True
            except Exception as e:
                logger.debug(f"Error in audio stream monitoring: {e}")
        
        # Method 14: Direct conversation history monitoring
        # Monitor the conversation history directly
        def monitor_conversation_history():
            try:
                if hasattr(agent_session, '_conversation_history'):
                    for entry in agent_session._conversation_history:
                        if entry.get('role') == 'user' and not entry.get('processed_for_extraction'):
                            content = entry.get('content', '')
                            logger.info(f"🔍 DEBUG: Found user message in conversation history: '{content}'")
                            process_livekit_transcript(content)
                            entry['processed_for_extraction'] = True
            except Exception as e:
                logger.debug(f"Error in conversation history monitoring: {e}")
        
        # Enhanced monitoring loop with all methods
        async def enhanced_monitor_and_extract():
            while True:
                try:
                    monitor_user_transcripts()
                    monitor_conversation_messages()
                    monitor_message_queue()
                    monitor_audio_stream()
                    monitor_conversation_history()
                    await asyncio.sleep(0.5)  # Check every 500ms
                except Exception as e:
                    logger.debug(f"Error in enhanced monitoring loop: {e}")
                    await asyncio.sleep(0.5)
        
        # Start enhanced monitoring
        asyncio.create_task(enhanced_monitor_and_extract())
        logger.info("🔍 DEBUG: Started enhanced monitoring for data extraction")
        
        # Simple transcript processor for direct data extraction
        def process_user_transcript(transcript: str):
            """Process user transcript and extract data"""
            if not transcript or transcript.strip() == "":
                return
                
            logger.info(f"🔍 Processing transcript: '{transcript}'")
            
            # Extract user information
            assistant.extract_user_info(transcript)
            
            # Log the transcript for debugging
            logger.info(f"🔍 Transcript processed: '{transcript}'")
        
        # Method 24: Direct transcript processing from LiveKit logs
        # Process transcripts directly as they appear in the debug logs
        def process_livekit_transcript(transcript: str):
            logger.info(f"🔍 DEBUG: Processing LiveKit transcript: '{transcript}'")
            process_user_transcript(transcript)
        
        # Method 25: Simple transcript handler
        # Add a simple handler that processes transcripts directly
        def simple_transcript_handler(transcript: str):
            logger.info(f"🔍 DEBUG: Simple transcript handler called with: '{transcript}'")
            process_user_transcript(transcript)
        
        # Method 26: Direct transcript injection
        # Inject our transcript handler into the agent session
        try:
            # Add our transcript handler to the agent session
            agent_session.transcript_handler = simple_transcript_handler
            logger.info("🔍 DEBUG: Added transcript_handler to agent session")
        except Exception as e:
            logger.error(f"🔍 DEBUG: Failed to add transcript_handler: {e}")
        
        # Method 27: Direct message processing
        # Process messages directly in the conversation loop
        def process_message_directly(message: str):
            logger.info(f"🔍 DEBUG: Processing message directly: '{message}'")
            process_user_transcript(message)
        
        # Method 28: Direct conversation monitoring with manual processing
        # Manually process the conversation messages we can see in the logs
        async def manual_conversation_processor():
            processed_transcripts = set()
            
            while True:
                try:
                    # Get the conversation from the agent session
                    if hasattr(agent_session, 'conversation'):
                        for msg in agent_session.conversation:
                            if msg.get('role') == 'user':
                                content = msg.get('content', '')
                                if content and content not in processed_transcripts:
                                    logger.info(f"🔍 DEBUG: Manually processing conversation message: '{content}'")
                                    process_message_directly(content)
                                    processed_transcripts.add(content)
                    
                    # Also check for any new transcripts in the session
                    if hasattr(agent_session, '_transcripts'):
                        for transcript in agent_session._transcripts:
                            text = transcript.get('text', '')
                            if text and text not in processed_transcripts:
                                logger.info(f"🔍 DEBUG: Manually processing transcript: '{text}'")
                                process_message_directly(text)
                                processed_transcripts.add(text)
                    
                    await asyncio.sleep(1)  # Check every second
                except Exception as e:
                    logger.debug(f"Error in manual conversation processor: {e}")
                    await asyncio.sleep(1)
        
        # Start manual conversation processor
        asyncio.create_task(manual_conversation_processor())
        logger.info("🔍 DEBUG: Started manual conversation processor")
        
        # Test the extraction functionality
        # logger.info("🔍 DEBUG: Testing extraction functionality...")
        # test_message = "My name is Alice, my email is alice@company.com, and my phone is 555-987-6543"
        # assistant.extract_user_info(test_message)
        # logger.info(f"🔍 DEBUG: Test extraction result: {assistant.user_data}")
        
        # Test the transcript processor
        # logger.info("🔍 DEBUG: Testing transcript processor...")
        # process_user_transcript("Hi, my name is Bob and my email is bob@test.com")

        test_messages = [
            "I want to know about pricing.",
            "Can you help me with a bug?",
            "How do I change my billing info?"
        ]
        for msg in test_messages:
            intent = assistant.detect_intent(msg)
            print(f"Message: '{msg}' -> Detected intent: {intent}")
        
        # Method 29: Direct transcript processing from LiveKit logs
        # Process transcripts directly as they appear in the debug logs
        def process_livekit_transcript(transcript: str):
            logger.info(f"🔍 DEBUG: Processing LiveKit transcript: '{transcript}'")
            
            # Extract user information from transcript
            assistant.extract_user_info(transcript)
            
            # Log collected user data
            if assistant.user_data:
                logger.info(f"Session {session_id}: Collected user data: {assistant.user_data}")
        
        # Method 20: Override the transcript processing at the LiveKit level
        # Try to intercept at the LiveKit agents level
        try:
            # Get the original transcript processor
            if hasattr(agent_session, '_transcript_processor'):
                original_processor = agent_session._transcript_processor
                
                def new_transcript_processor(transcript: str, *args, **kwargs):
                    logger.info(f"🔍 DEBUG: Intercepted at transcript processor: '{transcript}'")
                    process_livekit_transcript(transcript)
                    return original_processor(transcript, *args, **kwargs)
                
                agent_session._transcript_processor = new_transcript_processor
                logger.info("🔍 DEBUG: Successfully replaced _transcript_processor")
        except Exception as e:
            logger.error(f"🔍 DEBUG: Failed to replace _transcript_processor: {e}")
        
        # Method 21: Direct message queue monitoring
        # Monitor the message queue directly
        def monitor_message_queue():
            try:
                if hasattr(agent_session, '_message_queue'):
                    for msg in agent_session._message_queue:
                        if msg.get('type') == 'transcript' and not msg.get('processed_for_extraction'):
                            transcript = msg.get('content', '')
                            logger.info(f"🔍 DEBUG: Found transcript in message queue: '{transcript}'")
                            process_livekit_transcript(transcript)
                            msg['processed_for_extraction'] = True
            except Exception as e:
                logger.debug(f"Error in message queue monitoring: {e}")
        
        # Method 22: Direct stream monitoring
        # Monitor the audio stream directly
        def monitor_audio_stream():
            try:
                if hasattr(agent_session, '_audio_stream'):
                    # Process any pending transcripts in the audio stream
                    if hasattr(agent_session._audio_stream, '_pending_transcripts'):
                        for transcript in agent_session._audio_stream._pending_transcripts:
                            if not transcript.get('processed_for_extraction'):
                                text = transcript.get('text', '')
                                logger.info(f"🔍 DEBUG: Found transcript in audio stream: '{text}'")
                                process_livekit_transcript(text)
                                transcript['processed_for_extraction'] = True
            except Exception as e:
                logger.debug(f"Error in audio stream monitoring: {e}")
        
        # Method 23: Direct conversation history monitoring
        # Monitor the conversation history directly
        def monitor_conversation_history():
            try:
                if hasattr(agent_session, '_conversation_history'):
                    for entry in agent_session._conversation_history:
                        if entry.get('role') == 'user' and not entry.get('processed_for_extraction'):
                            content = entry.get('content', '')
                            logger.info(f"🔍 DEBUG: Found user message in conversation history: '{content}'")
                            process_livekit_transcript(content)
                            entry['processed_for_extraction'] = True
            except Exception as e:
                logger.debug(f"Error in conversation history monitoring: {e}")
        
        # Enhanced monitoring loop with all methods
        async def enhanced_monitor_and_extract():
            while True:
                try:
                    monitor_user_transcripts()
                    monitor_conversation_messages()
                    monitor_message_queue()
                    monitor_audio_stream()
                    monitor_conversation_history()
                    await asyncio.sleep(0.5)  # Check every 500ms
                except Exception as e:
                    logger.debug(f"Error in enhanced monitoring loop: {e}")
                    await asyncio.sleep(0.5)
        
        # Start enhanced monitoring
        asyncio.create_task(enhanced_monitor_and_extract())
        logger.info("🔍 DEBUG: Started enhanced monitoring for data extraction")
        
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
            
        logger.info(f"Dynamic session {session_id} ended and cleaned up")

if __name__ == "__main__":
    cli.run_app(
        WorkerOptions(
            entrypoint_fnc=entrypoint,
            prewarm_fnc=prewarm,
        ),
    )