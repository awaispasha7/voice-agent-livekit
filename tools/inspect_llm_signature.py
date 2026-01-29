import inspect


def main():
    try:
        from livekit.plugins import aws

        llm = aws.LLM(model="anthropic.claude-3-5-sonnet-20240620-v1:0", region="us-east-1")
        print("aws.LLM.chat:", inspect.signature(llm.chat))
        methods = [m for m in dir(llm) if not m.startswith("_")]
        print("aws.LLM methods (filtered):", [m for m in methods if any(k in m.lower() for k in ["chat", "complete", "generate", "stream"])])
    except Exception as e:
        print("FAILED aws.LLM:", repr(e))

    try:
        from livekit.plugins import openai as lk_openai

        llm = lk_openai.LLM(model="gpt-4o-mini")
        print("openai.LLM.chat:", inspect.signature(llm.chat))
    except Exception as e:
        print("FAILED openai.LLM:", repr(e))


if __name__ == "__main__":
    main()


