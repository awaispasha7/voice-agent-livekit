# worker_refactored.py — Orchestrator-first
# Worker acts only as transport: speech in → backend orchestrator → TTS out.

import asyncio, json, logging, os, threading, time, uuid
from datetime import datetime
from typing import Dict, Any, Optional, AsyncIterable

import httpx, psutil
from dotenv import load_dotenv
from pathlib import Path

from livekit.agents import AgentSession, Agent, JobContext, WorkerOptions, cli, RoomInputOptions, RoomOutputOptions, AutoSubscribe
from livekit.plugins import deepgram, cartesia, silero, noise_cancellation
from livekit.agents import llm

# -----------------------------------------------------------------------------
# Env & Logging
# -----------------------------------------------------------------------------
current_dir = Path(__file__).parent
env_paths = [
    current_dir / "../../.env",      # relative to worker/backend
    current_dir / "../../../.env",   # project root fallback
    Path("/home/ubuntu/alive5-voice-agent/.env"),  # production path
    Path(".env"),                    # current working directory
]

env_loaded = False
for env_path in env_paths:
    if env_path.exists():
        load_dotenv(dotenv_path=str(env_path), override=True)
        print(f"✅ Loaded .env from: {env_path}")
        env_loaded = True
        break

if not env_loaded:
    load_dotenv()  # fallback, picks up whatever is in environment already

logging.basicConfig(level=logging.INFO, format="%(message)s", force=True)
logger = logging.getLogger("orchestrator-worker")

BACKEND_URL = os.getenv("BACKEND_URL", "http://localhost:8000")
BACKEND_TIMEOUT = 20

# -----------------------------------------------------------------------------
# Utils
# -----------------------------------------------------------------------------
def preprocess_text_for_tts(text: str) -> str:
    if not text: return text
    return (text.replace("SSO", "S-S-O")
                .replace("CRM", "C-R-M")
                .replace("API", "A-P-I")
                .replace("URL", "U-R-L"))

def monitor_memory():
    p = psutil.Process(os.getpid())
    logger.info(f"💾 Memory: {p.memory_info().rss / 1024 / 1024:.1f} MB")

# -----------------------------------------------------------------------------
# Backend LLM Proxy
# -----------------------------------------------------------------------------
class BackendLLM(llm.LLM):
    def __init__(self, backend_url: str):
        super().__init__()
        self.backend_url = backend_url
        self.room_name = None

    def set_room_name(self, room_name: str):
        self.room_name = room_name

    async def chat(self, ctx: Optional[llm.ChatContext] = None, *, chat_ctx: Optional[llm.ChatContext] = None):
        ctx = ctx or chat_ctx
        try:
            user_message = None
            for msg in reversed(ctx.messages):
                if msg.role == "user":
                    user_message = msg.content
                    break

            if not user_message:
                return llm.ChatCompletion(
                    choices=[llm.ChatCompletionChoice(
                        message=llm.ChatMessage(
                            role="assistant",
                            content="I didn’t receive your message. Could you please repeat that?"
                        )
                    )]
                )

            history = [
                {"role": m.role, "content": m.content, "timestamp": datetime.now().isoformat()}
                for m in ctx.messages if m.role in ["user", "assistant"]
            ]

            result = await self.call_backend(user_message, history)
            response = result.get("response", "Sorry, I'm having trouble. Let me connect you with a human agent.")

            return llm.ChatCompletion(
                choices=[llm.ChatCompletionChoice(
                    message=llm.ChatMessage(role="assistant", content=response)
                )]
            )

        except Exception as e:
            logger.error(f"Chat error: {e}")
            return llm.ChatCompletion(
                choices=[llm.ChatCompletionChoice(
                    message=llm.ChatMessage(role="assistant", content="Sorry, I’m having trouble reaching the backend.")
                )]
            )


    async def call_backend(self, msg: str, history: list) -> Dict[str,Any]:
        try:
            async with httpx.AsyncClient(timeout=BACKEND_TIMEOUT) as client:
                payload = {
                    "room_name": self.room_name,
                    "user_message": msg,
                    "conversation_history": history
                }
                resp = await client.post(f"{self.backend_url}/api/process_flow_message", json=payload)
                if resp.status_code == 200:
                    return resp.json().get("flow_result", {})
        except Exception as e:
            logger.error(f"Backend call failed: {e}")

        return {"type":"error","response":"Sorry, I'm having trouble. Let me connect you with a human agent."}

# -----------------------------------------------------------------------------
# Assistant
# -----------------------------------------------------------------------------
class OrchestratorAssistant(Agent):
    def __init__(self, sid: str, proxy: BackendLLM):
        self.sid=sid; self.room_name=None
        self.selected_voice="f114a467-c40a-4db8-964d-aaba89cd08fa"  # Miles - Yogi
        self._speech_lock=asyncio.Lock(); self._buffer=""; self._task=None; self._window=1.0
        self.proxy=proxy
        super().__init__(instructions="Voice agent (orchestrator-first)", llm=proxy)
    
    async def on_enter(self):
        logger.info(f"🎤 Room entered {self.sid}")
        # Set up room event listeners
        if self.room:
            def _handle_data_received(data):
                asyncio.create_task(self._on_data_received(data))
            self.room.on("data_received", _handle_data_received)
        # Trigger orchestrator to start greeting
        await self._process_text("__start__")
    
    async def on_user_turn_completed(self, turn_ctx: llm.ChatContext, new_message: llm.ChatMessage) -> None:
        text=(new_message.text_content or "").strip(); 
        if not text: return
        self._buffer=(self._buffer+" "+text).strip()
        if self._task and not self._task.done(): self._task.cancel()
        self._task=asyncio.create_task(self._flush())
    
    async def _flush(self):
        await asyncio.sleep(self._window)
        text=self._buffer.strip(); self._buffer=""
        if text: await self._process_text(text)
    
    async def _on_data_received(self, data):
        """Handle data messages from frontend"""
        try:
            # Voice changes are no longer supported during calls
            # Voice can only be changed before starting the session
            if data.topic == "lk.voice.change":
                logger.info(f"🎤 Voice change request received but ignored - voice changes only supported before session start")
        except Exception as e:
            logger.error(f"Error handling data message: {e}")


    async def _get_current_voice(self):
        """Get current voice from session data"""
        try:
            if not self.room_name:
                return self.selected_voice
            
            # Get session info from backend
            async with httpx.AsyncClient(timeout=5.0) as client:
                response = await client.get(f"{BACKEND_URL}/api/sessions/{self.room_name}")
                if response.status_code == 200:
                    session_data = response.json()
                    return session_data.get("selected_voice") or session_data.get("voice_id") or self.selected_voice
        except Exception as e:
            logger.error(f"Failed to get current voice: {e}")
        
        return self.selected_voice
    
    async def _process_text(self, user_text: str):
        logger.info(f"USER: {user_text}")
        history=[{"role":"user","content":user_text,"timestamp":datetime.now().isoformat()}]
        
        # Get current voice from session data
        current_voice = await self._get_current_voice()
        if current_voice != self.selected_voice:
            logger.info(f"🎤 Voice updated from session: {current_voice}")
            self.selected_voice = current_voice
        
        # Send thinking indicator to frontend
        try:
            thinking_data = {
                "type": "thinking_start",
                "timestamp": datetime.now().isoformat()
            }
            await self.room.local_participant.publish_data(
                json.dumps(thinking_data).encode('utf-8'),
                topic="lk.conversation.control"
            )
            logger.info(f"🔍 Thinking indicator started")
        except Exception as e:
            logger.error(f"Failed to start thinking indicator: {e}")
        
        result=await self.proxy.call_backend(user_text, history)
        response=result.get("response") or "..."
        logger.info(f"ASSISTANT: {response}")
        
        # Stop thinking indicator
        try:
            thinking_data = {
                "type": "thinking_stop",
                "timestamp": datetime.now().isoformat()
            }
            await self.room.local_participant.publish_data(
                json.dumps(thinking_data).encode('utf-8'),
                topic="lk.conversation.control"
            )
            logger.info(f"🔍 Thinking indicator stopped")
        except Exception as e:
            logger.error(f"Failed to stop thinking indicator: {e}")
        
        # Send intent update to frontend if flow was started
        flow_type = result.get("type", "")
        flow_name = result.get("flow_name", "")
        if flow_type == "flow_started" and flow_name:
            try:
                intent_data = {
                    "type": "intent_update",
                    "intent": flow_name,
                    "source": "Flow System",
                    "timestamp": datetime.now().isoformat()
                }
                await self.room.local_participant.publish_data(
                    json.dumps(intent_data).encode('utf-8'),
                    topic="lk.conversation.control"
                )
                logger.info(f"🔍 Intent update sent to frontend: {flow_name}")
            except Exception as e:
                logger.error(f"Failed to send intent update: {e}")
        
        # Send agent response to frontend for chat display FIRST
        try:
            agent_transcript_data = {
                "type": "agent_transcript",
                "message": response,
                "speaker": "Assistant",
                "timestamp": datetime.now().isoformat()
            }
            await self.room.local_participant.publish_data(
                json.dumps(agent_transcript_data).encode('utf-8'),
                topic="lk.agent.transcript"
            )
            logger.info(f"🔍 Agent transcript sent to frontend: {response}")
            
            # Give frontend a moment to display the message before starting TTS
            await asyncio.sleep(0.5)
            
        except Exception as e:
            logger.error(f"Failed to send agent transcript: {e}")
        
        # THEN start TTS after frontend has displayed the message
        async with self._speech_lock: 
            await self.session.say(preprocess_text_for_tts(response))
        
        if result.get("type") in ["conversation_end", "call_ended"]:
            logger.info(f"🔍 Session ending, sending conversation end signal...")
            await asyncio.sleep(2)  # Give time for the response to be spoken
            if self.room: 
                # Send data message to frontend to notify of conversation end
                try:
                    end_data = {
                        "type": "conversation_end",
                        "reason": "user_requested",
                        "timestamp": datetime.now().isoformat()
                    }
                    await self.room.local_participant.publish_data(
                        json.dumps(end_data).encode('utf-8'),
                        topic="lk.conversation.control"
                    )
                    logger.info(f"🔍 Conversation end signal sent to frontend")
                except Exception as e:
                    logger.error(f"Failed to send conversation end signal: {e}")
                    # Fallback to direct disconnect
                    await self.room.disconnect()
                    logger.info(f"🔍 Room disconnected (fallback)")

# -----------------------------------------------------------------------------
# Entrypoint
# -----------------------------------------------------------------------------
def prewarm(proc): proc.userdata["vad"]=silero.VAD.load()
async def entrypoint(ctx: JobContext):
    sid,room=str(uuid.uuid4())[:8],ctx.room.name
    proxy=BackendLLM(BACKEND_URL); proxy.set_room_name(room)
    assistant=OrchestratorAssistant(sid, proxy); assistant.room_name=room
    await ctx.connect(auto_subscribe=AutoSubscribe.SUBSCRIBE_ALL)
    
    # Get current voice from session data
    current_voice = await assistant._get_current_voice()
    assistant.selected_voice = current_voice
    logger.info(f"🎤 Initializing TTS with voice: {current_voice}")

    agent_session=AgentSession(
        stt=deepgram.STT(model="nova-2",language="en-US",api_key=os.getenv("DEEPGRAM_API_KEY")),
        llm=proxy,
        tts=cartesia.TTS(model="sonic-2",voice=current_voice,api_key=os.getenv("CARTESIA_API_KEY")),
        vad=ctx.proc.userdata["vad"],turn_detection=None,
    )
    assistant.room,assistant.agent_session=ctx.room,agent_session
    await agent_session.start(
        room=ctx.room,agent=assistant,
        room_input_options=RoomInputOptions(noise_cancellation=noise_cancellation.BVC(),text_enabled=True),
        room_output_options=RoomOutputOptions(transcription_enabled=True,sync_transcription=False),
    )

if __name__=="__main__":
    cli.run_app(WorkerOptions(entrypoint_fnc=entrypoint,prewarm_fnc=prewarm))