import asyncio
import inspect

from livekit.agents.llm import ChatContext, ChatMessage
from livekit.plugins import aws


async def main():
    llm = aws.LLM(model="anthropic.claude-3-5-sonnet-20240620-v1:0", region="us-east-1")
    ctx = ChatContext(items=[ChatMessage(role="user", content=["Say 'ok' and nothing else."])])
    stream = llm.chat(chat_ctx=ctx)
    print("stream type:", type(stream))
    print("stream attrs:", [a for a in dir(stream) if "context" in a.lower() or "final" in a.lower() or "collect" in a.lower()])
    # Try common patterns
    if hasattr(stream, "get_final_response"):
        print("has get_final_response:", inspect.signature(stream.get_final_response))
    if hasattr(stream, "get_final_chat_context"):
        print("has get_final_chat_context:", inspect.signature(stream.get_final_chat_context))
    if hasattr(stream, "aclose"):
        print("has aclose")

    # Consume minimal to completion if possible
    text = ""
    try:
        async for ev in stream:
            # Best-effort: accumulate any text deltas
            delta = getattr(ev, "delta", None) or getattr(ev, "text", None) or ""
            if isinstance(delta, str):
                text += delta
    except Exception as e:
        print("stream iteration failed:", repr(e))

    print("collected text:", text.strip()[:200])

    for method in ("get_final_chat_context", "final_chat_context", "chat_context", "context"):
        if hasattr(stream, method):
            try:
                val = getattr(stream, method)
                if callable(val):
                    out = await val()
                else:
                    out = val
                print(method, "->", type(out))
                # print small preview
                if hasattr(out, "items"):
                    print("items len:", len(out.items))
                elif hasattr(out, "messages"):
                    print("messages len:", len(out.messages))
            except Exception as e:
                print(method, "failed:", repr(e))


if __name__ == "__main__":
    asyncio.run(main())


