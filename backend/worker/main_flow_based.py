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
for path in [current_dir / "../../.env", Path(".env")]:
    if path.exists(): load_dotenv(path)

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
    while True: _ = p.memory_info().rss; time.sleep(5)
threading.Thread(target=monitor_memory, daemon=True).start()

# -----------------------------------------------------------------------------
# Backend proxy LLM
# -----------------------------------------------------------------------------
class BackendLLM(llm.LLM):
    def __init__(self, backend_url: str):
        super().__init__(); self.backend_url = backend_url; self.room_name=None
    def set_room_name(self, room: str): self.room_name=room
    def chat(self, *, chat_ctx: llm.ChatContext, **kwargs) -> llm.LLMStream:
        return self._create_response_stream(chat_ctx, "")
    def _create_response_stream(self, ctx, text: str):
        async def _gen():
            yield llm.ChatChunk(
                id=str(uuid.uuid4()),
                delta=llm.ChoiceDelta(content=text, role="assistant")
            )

        class Stream(llm.LLMStream):
            def __init__(self, generator):
                self._generator = generator
                self._ctx = ctx

            async def _run(self):
                async for chunk in self._generator:
                    yield chunk

            def __aiter__(self):
                return self

            async def __anext__(self):
                return await anext(self._generator)

            @property
            def chat_ctx(self):
                return self._ctx

            def execute_functions(self):
                return []

            async def aclose(self):
                return

        return Stream(_gen())

    async def call_backend(self, msg: str, history: list) -> Dict[str,Any]:
        payload={"room_name": self.room_name, "user_message": msg, "conversation_history": history[-10:]}
        try:
            async with httpx.AsyncClient(timeout=BACKEND_TIMEOUT) as c:
                r=await c.post(f"{self.backend_url}/api/process_flow_message",json=payload)
                if r.status_code==200: return r.json().get("flow_result",{})
        except Exception as e: logger.error(f"Backend error: {e}")
        return {"type":"error","response":"Sorry, I'm having trouble. Let me connect you with a human agent."}

# -----------------------------------------------------------------------------
# Assistant
# -----------------------------------------------------------------------------
class OrchestratorAssistant(Agent):
    def __init__(self, sid: str, proxy: BackendLLM):
        self.sid=sid; self.room_name=None
        self.selected_voice="7f423809-0011-4658-ba48-a411f5e516ba"
        self._speech_lock=asyncio.Lock(); self._buffer=""; self._task=None; self._window=1.0
        self.proxy=proxy
        super().__init__(instructions="Voice agent (orchestrator-first)", llm=proxy)
    async def on_enter(self):
        logger.info(f"🎤 Room entered {self.sid}")
        # Trigger orchestrator to start greeting
        await self._process_text("__start__")
    async def on_user_turn_completed(self, ctx, msg):
        text=(msg.text_content or "").strip(); 
        if not text: return
        self._buffer=(self._buffer+" "+text).strip()
        if self._task and not self._task.done(): self._task.cancel()
        self._task=asyncio.create_task(self._flush())
    async def _flush(self):
        await asyncio.sleep(self._window)
        text=self._buffer.strip(); self._buffer=""
        if text: await self._process_text(text)
    async def _process_text(self, user_text: str):
        logger.info(f"USER: {user_text}")
        history=[{"role":"user","content":user_text,"timestamp":datetime.now().isoformat()}]
        result=await self.proxy.call_backend(user_text, history)
        response=result.get("response") or "..."
        logger.info(f"ASSISTANT: {response}")
        async with self._speech_lock: await self.session.say(preprocess_text_for_tts(response))
        if result.get("type")=="conversation_end":
            await asyncio.sleep(1)
            if self.room: await self.room.disconnect()

# -----------------------------------------------------------------------------
# Entrypoint
# -----------------------------------------------------------------------------
def prewarm(proc): proc.userdata["vad"]=silero.VAD.load()
async def entrypoint(ctx: JobContext):
    sid,room=str(uuid.uuid4())[:8],ctx.room.name
    proxy=BackendLLM(BACKEND_URL); proxy.set_room_name(room)
    assistant=OrchestratorAssistant(sid, proxy); assistant.room_name=room
    await ctx.connect(auto_subscribe=AutoSubscribe.SUBSCRIBE_ALL)
    agent_session=AgentSession(
        stt=deepgram.STT(model="nova-2",language="en-US",api_key=os.getenv("DEEPGRAM_API_KEY")),
        llm=proxy,
        tts=cartesia.TTS(model="sonic-2",voice=assistant.selected_voice,api_key=os.getenv("CARTESIA_API_KEY")),
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
