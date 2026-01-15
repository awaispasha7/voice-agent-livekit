"""
AgentCore Runtime Agent Implementation
This agent handles LLM processing and function calling via AgentCore Runtime
Uses AgentCore Gateway for tools and AgentCore Memory for state
"""

import os
import json
import logging
from typing import Dict, Any, Optional, List
from pathlib import Path

try:
    from bedrock_agentcore.runtime import BedrockAgentCoreApp
    AGENTCORE_AVAILABLE = True
except ImportError:
    AGENTCORE_AVAILABLE = False
    logging.warning("bedrock-agentcore not installed. AgentCore features will be disabled.")

from dotenv import load_dotenv

# Load environment variables
env_path = Path(__file__).parent.parent.parent / ".env"
if env_path.exists():
    load_dotenv(env_path)

logger = logging.getLogger("agentcore-agent")

# Import AgentCore utilities
try:
    from agentcore.memory import AgentCoreMemory
    from agentcore.gateway_tools import AgentCoreGateway
    MEMORY_AVAILABLE = True
    GATEWAY_AVAILABLE = True
except ImportError:
    MEMORY_AVAILABLE = False
    GATEWAY_AVAILABLE = False
    logger.warning("AgentCore utilities not available")

# Initialize AgentCore app if available
if AGENTCORE_AVAILABLE:
    app = BedrockAgentCoreApp()
else:
    app = None

# Safe decorator that becomes a no-op if AgentCore runtime isn't installed.
def _agentcore_entrypoint(fn):
    if AGENTCORE_AVAILABLE and app:
        return app.entrypoint(fn)
    return fn

# Initialize Memory and Gateway
memory_client = AgentCoreMemory() if MEMORY_AVAILABLE else None
gateway_client = AgentCoreGateway() if GATEWAY_AVAILABLE else None


@_agentcore_entrypoint
async def handler(event: Dict[str, Any], context: Any) -> Dict[str, Any]:
    """
    AgentCore entrypoint handler for voice agent
    
    Expected input:
    {
        "transcription": "user's spoken text",
        "session_id": "room_name",
        "botchain_name": "voice-1",
        "org_name": "alive5stage0",
        "context": {
            "conversation_history": [...],
            "collected_data": {...},
            "flow_states": {...},
            "bot_template": {...}
        },
        "functions": ["faq_bot_request", "load_bot_flows", ...]
    }
    
    Returns:
    {
        "response": "agent's text response",
        "functions_called": [...],
        "updated_state": {...}
    }
    """
    if not AGENTCORE_AVAILABLE:
        return {
            "error": "AgentCore not available",
            "response": "I'm having trouble processing your request right now."
        }
    
    try:
        # Extract input
        transcription = event.get("transcription", "")
        session_id = event.get("session_id", "")
        botchain_name = event.get("botchain_name", "voice-1")
        org_name = event.get("org_name", "alive5stage0")
        context_data = event.get("context", {})
        
        logger.info(f"AgentCore processing: session={session_id}, transcription={transcription[:50]}...")
        
        # Get conversation history from AgentCore Memory
        if memory_client and memory_client.is_enabled():
            conversation_history = await memory_client.get_conversation_history(session_id, limit=20)
            # Get session data from memory
            session_data = await memory_client.get_session(session_id) or {}
            collected_data = session_data.get("collected_data", {})
            flow_states = session_data.get("flow_states", {})
            bot_template = session_data.get("bot_template")
        else:
            # Fallback to context data
            conversation_history = context_data.get("conversation_history", [])
            collected_data = context_data.get("collected_data", {})
            flow_states = context_data.get("flow_states", {})
            bot_template = context_data.get("bot_template")
        
        # Import LLM
        from livekit.plugins import aws, openai
        
        # Get LLM provider from env
        llm_provider = os.getenv("LLM_PROVIDER", "bedrock").lower()
        
        # Initialize LLM based on provider
        if llm_provider == "bedrock":
            bedrock_model = os.getenv("BEDROCK_MODEL", "anthropic.claude-3-5-sonnet-20240620-v1:0")
            bedrock_region = os.getenv("BEDROCK_REGION", "us-east-1")
            llm = aws.LLM(model=bedrock_model, region=bedrock_region, temperature=0.3)
        elif llm_provider == "nova":
            nova_region = os.getenv("NOVA_REGION", "us-east-1")
            llm = aws.realtime.RealtimeModel(region=nova_region)
        else:
            openai_model = os.getenv("OPENAI_MODEL", "gpt-4o")
            llm = openai.LLM(model=openai_model, temperature=0.3)
        
        # Build system prompt (load from system_prompt.py if available)
        try:
            import sys
            sys.path.insert(0, str(Path(__file__).parent.parent / "alive5-worker"))
            from system_prompt import get_system_prompt
            system_prompt = get_system_prompt(botchain_name, org_name, "")
        except ImportError:
            system_prompt = f"""You are a conversational voice agent for {org_name}.
Handle the user's request conversationally and naturally.
Use the provided functions when needed."""
        
        # Prepare messages for LLM
        messages = [{"role": "system", "content": system_prompt}]
        
        # Add conversation history
        for msg in conversation_history[-10:]:  # Last 10 messages for context
            messages.append(msg)
        
        # Add current user message
        messages.append({"role": "user", "content": transcription})
        
        # Call LLM (with function calling support via Gateway)
        # If Gateway is enabled, LLM can call tools through Gateway
        response = llm.chat(messages=messages)
        
        # Extract response text
        response_text = response.content if hasattr(response, 'content') else str(response)
        
        logger.info(f"AgentCore response generated: {response_text[:50]}...")
        
        # Store conversation turn in AgentCore Memory
        if memory_client and memory_client.is_enabled():
            await memory_client.store_conversation_turn(session_id, "user", transcription)
            await memory_client.store_conversation_turn(session_id, "assistant", response_text)
        
        # Update session state in Memory
        if memory_client and memory_client.is_enabled():
            await memory_client.store_session(session_id, {
                "collected_data": collected_data,
                "flow_states": flow_states,
                "bot_template": bot_template,
                "botchain_name": botchain_name,
                "org_name": org_name
            })
        
        return {
            "response": response_text,
            "functions_called": [],
            "updated_state": {
                "conversation_history": conversation_history + [
                    {"role": "user", "content": transcription},
                    {"role": "assistant", "content": response_text}
                ],
                "collected_data": collected_data,
                "flow_states": flow_states
            }
        }
        
    except Exception as e:
        logger.error(f"Error in AgentCore handler: {e}", exc_info=True)
        return {
            "error": str(e),
            "response": "I'm having trouble processing your request right now. Please try again."
        }


if __name__ == "__main__":
    """Run AgentCore agent locally for testing"""
    if app:
        logger.info("Starting AgentCore agent...")
        app.run()
    else:
        logger.error("AgentCore not available. Install with: pip install bedrock-agentcore")

