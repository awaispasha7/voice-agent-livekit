"""
AgentCore Runtime Client
Handles HTTP communication with deployed AgentCore Runtime agent
Integrates with AgentCore Memory and Gateway
"""

import os
import json
import logging
from typing import Dict, Any, Optional, List
import httpx
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables
env_path = Path(__file__).parent.parent.parent / ".env"
if env_path.exists():
    load_dotenv(env_path)

logger = logging.getLogger("agentcore-client")

# Import Memory client
try:
    from agentcore.memory import AgentCoreMemory
    MEMORY_AVAILABLE = True
except ImportError:
    MEMORY_AVAILABLE = False


class AgentCoreClient:
    """Client for invoking AgentCore Runtime agent"""
    
    def __init__(self):
        self.agent_arn = os.getenv("AGENTCORE_AGENT_ARN")
        self.agent_url = os.getenv("AGENTCORE_AGENT_URL")
        self.enabled = os.getenv("USE_AGENTCORE", "false").lower() == "true"
        
        # Initialize Memory client
        self.memory_client = AgentCoreMemory() if MEMORY_AVAILABLE else None
        
        if not self.enabled:
            logger.info("AgentCore is disabled (USE_AGENTCORE=false)")
        elif not self.agent_arn and not self.agent_url:
            logger.warning("AgentCore enabled but no AGENTCORE_AGENT_ARN or AGENTCORE_AGENT_URL configured")
            self.enabled = False
    
    async def invoke(
        self,
        transcription: str,
        session_id: str,
        botchain_name: str = "voice-1",
        org_name: str = "alive5stage0",
        conversation_history: Optional[List[Dict[str, Any]]] = None,
        collected_data: Optional[Dict[str, Any]] = None,
        flow_states: Optional[Dict[str, Any]] = None,
        bot_template: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Invoke AgentCore Runtime agent
        
        Args:
            transcription: User's spoken text
            session_id: Room/session identifier
            botchain_name: Botchain name
            org_name: Organization name
            conversation_history: Previous conversation messages
            collected_data: Collected CRM data
            flow_states: Current flow states
            bot_template: Bot template/flow definitions
            
        Returns:
            {
                "response": "agent response text",
                "functions_called": [...],
                "updated_state": {...}
            }
        """
        if not self.enabled:
            return {
                "error": "AgentCore not enabled",
                "response": None
            }
        
        try:
            # Prepare payload
            payload = {
                "transcription": transcription,
                "session_id": session_id,
                "botchain_name": botchain_name,
                "org_name": org_name,
                "context": {
                    "conversation_history": conversation_history or [],
                    "collected_data": collected_data or {},
                    "flow_states": flow_states or {},
                    "bot_template": bot_template
                }
            }
            
            # Invoke AgentCore
            if self.agent_url:
                # Direct HTTP call to AgentCore Runtime endpoint
                async with httpx.AsyncClient(timeout=30.0) as client:
                    response = await client.post(
                        f"{self.agent_url}/invoke",
                        json=payload,
                        headers={"Content-Type": "application/json"}
                    )
                    response.raise_for_status()
                    result = response.json()
            elif self.agent_arn:
                # Use AgentCore SDK (if available)
                try:
                    from bedrock_agentcore import AgentCoreClient as SDKClient
                    sdk_client = SDKClient()
                    result = await sdk_client.invoke_agent(
                        agent_arn=self.agent_arn,
                        input_data=payload
                    )
                except ImportError:
                    logger.error("bedrock-agentcore SDK not available. Use AGENTCORE_AGENT_URL instead.")
                    return {
                        "error": "AgentCore SDK not available",
                        "response": None
                    }
            else:
                return {
                    "error": "No AgentCore configuration",
                    "response": None
                }
            
            logger.debug(f"AgentCore response: {result.get('response', '')[:50]}...")
            return result
            
        except httpx.TimeoutException:
            logger.error("AgentCore request timed out")
            return {
                "error": "Request timeout",
                "response": None
            }
        except httpx.HTTPStatusError as e:
            logger.error(f"AgentCore HTTP error: {e.response.status_code} - {e.response.text}")
            return {
                "error": f"HTTP {e.response.status_code}",
                "response": None
            }
        except Exception as e:
            logger.error(f"Error invoking AgentCore: {e}", exc_info=True)
            return {
                "error": str(e),
                "response": None
            }
    
    def is_enabled(self) -> bool:
        """Check if AgentCore is enabled"""
        return self.enabled

