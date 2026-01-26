import asyncio
import json
import os
import time

import httpx
import socketio


def derive_socket_base(a5_base_url: str) -> str:
    a5_base_url = (a5_base_url or "https://api-v2-stage.alive5.com").strip()
    host = a5_base_url.split("://", 1)[-1].split("/", 1)[0].strip()
    host = host.replace("api-v2-stage.", "api-stage.")
    host = host.replace("api-v2.", "api.")
    return f"wss://{host}"


async def main():
    backend = os.getenv("BACKEND_URL_INTERNAL", "http://127.0.0.1:8000").strip()
    a5_api_key = (os.getenv("A5_API_KEY") or "").strip()
    a5_base_url = os.getenv("A5_BASE_URL", "https://api-v2-stage.alive5.com")

    if not a5_api_key:
        raise SystemExit("A5_API_KEY is not set (expected in /home/ubuntu/alive5-voice-agent/.env)")

    socket_base = os.getenv("A5_SOCKET_URL", "").strip() or derive_socket_base(a5_base_url)

    room = f"test_socket_bridge_{int(time.time())}"
    print("Creating init_livechat for room:", room)

    async with httpx.AsyncClient(timeout=15.0) as client:
        resp = await client.post(
            f"{backend}/api/init_livechat",
            params={"room_name": room, "org_name": "alive5stage0", "botchain_name": "ai-voice-posh"},
        )
        resp.raise_for_status()
        data = resp.json()

    thread_id = data.get("thread_id")
    crm_id = data.get("crm_id")
    channel_id = data.get("channel_id")

    print(
        "init_livechat ok:",
        json.dumps({"thread_id": thread_id, "crm_id": crm_id, "channel_id": channel_id}, indent=2),
    )

    qs = {
        "type": "voice_agent",
        "x-a5-apikey": a5_api_key,
        "thread_id": thread_id,
        "crm_id": crm_id,
        "channel_id": channel_id,
    }

    url = socket_base + "?" + "&".join([f"{k}={v}" for k, v in qs.items()])
    print("Connecting socket to:", socket_base)

    ack = asyncio.Event()

    sio = socketio.AsyncClient(reconnection=False, logger=False, engineio_logger=False)

    @sio.event
    async def connect():
        print("socket connected")
        await sio.emit(
            "init_voice_agent",
            {"thread_id": thread_id, "crm_id": crm_id, "channel_id": channel_id},
        )
        print("init_voice_agent emitted")

    @sio.on("init_voice_agent_ack")
    async def on_ack(payload):
        print("init_voice_agent_ack:", payload)
        ack.set()

    @sio.event
    async def connect_error(e):
        print("connect_error:", e)

    await sio.connect(url, transports=["websocket"], socketio_path="socket.io", wait=True, wait_timeout=10)

    try:
        await asyncio.wait_for(ack.wait(), timeout=10)
    except asyncio.TimeoutError:
        print("WARNING: did not receive init_voice_agent_ack within timeout")

    await sio.emit(
        "post_message",
        {
            "thread_id": thread_id,
            "crm_id": crm_id,
            "message_content": "test message from server-side socket bridge",
            "message_type": "livechat",
        },
    )
    print("post_message emitted")

    await sio.emit(
        "save_crm_data",
        {
            "crm_id": crm_id,
            "thread_id": thread_id,
            "key": "notes",
            "value": "test crm update from server-side socket bridge",
        },
    )
    print("save_crm_data emitted")

    await sio.emit(
        "end_voice_chat",
        {
            "end_by": "voice_agent",
            "message_content": "ending test chat",
            "org_name": "alive5stage0",
            "thread_id": thread_id,
            "voice_agent_id": "",
        },
    )
    print("end_voice_chat emitted")

    await sio.disconnect()
    print("socket disconnected")


if __name__ == "__main__":
    asyncio.run(main())


