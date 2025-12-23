"""
AgentCore Integration for LiveKit Worker
This module provides integration between LiveKit worker and AgentCore Runtime
"""

import os
import logging
from typing import Dict, Any, Optional, List
from pathlib import Path
import sys

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

try:
    from agentcore.client import AgentCoreClient
    AGENTCORE_AVAILABLE = True
except ImportError:
    AGENTCORE_AVAILABLE = False
    logging.warning("AgentCore client not available. Will use direct LLM.")

logger = logging.getLogger("agentcore-integration")


class AgentCoreIntegration:
    """Integration layer between LiveKit worker and AgentCore Runtime"""
    
    def __init__(self):
        self.client = AgentCoreClient() if AGENTCORE_AVAILABLE else None
        self.enabled = self.client.is_enabled() if self.client else False
        
        if self.enabled:
            logger.info("✅ AgentCore integration enabled")
        else:
            logger.info("ℹ️ AgentCore integration disabled - using direct LLM")
    
    async def process_user_message(
        self,
        user_text: str,
        session_id: str,
        botchain_name: str,
        org_name: str,
        conversation_history: Optional[List[Dict[str, Any]]] = None,
        collected_data: Optional[Dict[str, Any]] = None,
        flow_states: Optional[Dict[str, Any]] = None,
        bot_template: Optional[Dict[str, Any]] = None
    ) -> Optional[str]:
        """
        Process user message through AgentCore or return None to use direct LLM
        
        Args:
            user_text: User's transcribed text
            session_id: Room/session identifier
            botchain_name: Botchain name
            org_name: Organization name
            conversation_history: Previous conversation messages
            collected_data: Collected CRM data
            flow_states: Current flow states
            bot_template: Bot template/flow definitions
            
        Returns:
            Agent response text if AgentCore processed it, None to use direct LLM
        """
        if not self.enabled or not self.client:
            return None  # Use direct LLM
        
        try:
            # Call AgentCore
            result = await self.client.invoke(
                transcription=user_text,
                session_id=session_id,
                botchain_name=botchain_name,
                org_name=org_name,
                conversation_history=conversation_history or [],
                collected_data=collected_data or {},
                flow_states=flow_states or {},
                bot_template=bot_template
            )
            
            # Check for errors
            if result.get("error"):
                logger.warning(f"AgentCore error: {result.get('error')}")
                # Fallback to direct LLM if configured
                if os.getenv("AGENTCORE_FALLBACK_TO_DIRECT_LLM", "true").lower() == "true":
                    logger.info("🔄 Falling back to direct LLM due to AgentCore error")
                    return None
                else:
                    # Return error message
                    return result.get("response", "I'm having trouble processing your request right now.")
            
            # Return AgentCore response
            response = result.get("response")
            if response:
                logger.debug(f"AgentCore response: {response[:50]}...")
                return response
            else:
                logger.warning("AgentCore returned no response")
                return None
                
        except Exception as e:
            logger.error(f"Error in AgentCore integration: {e}", exc_info=True)
            # Fallback to direct LLM
            if os.getenv("AGENTCORE_FALLBACK_TO_DIRECT_LLM", "true").lower() == "true":
                logger.info("🔄 Falling back to direct LLM due to exception")
                return None
            else:
                return "I'm having trouble processing your request right now."
    
    def is_enabled(self) -> bool:
        """Check if AgentCore is enabled"""
        return self.enabled

