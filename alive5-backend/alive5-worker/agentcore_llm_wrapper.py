"""
AgentCore LLM Wrapper
Wraps LiveKit LLM to intercept calls and route to AgentCore Runtime
"""

import logging
from typing import Optional, List, Dict, Any
from livekit.agents.llm import LLM, ChatContext, ChatRole, ChatMessage

logger = logging.getLogger("agentcore-llm-wrapper")


class AgentCoreLLMWrapper(LLM):
    """Wrapper that intercepts LLM calls and routes to AgentCore Runtime"""
    
    def __init__(self, base_llm: LLM, agentcore_integration, session_id: str, botchain_name: str, org_name: str):
        """
        Initialize wrapper
        
        Args:
            base_llm: Original LLM instance (fallback)
            agentcore_integration: AgentCoreIntegration instance
            session_id: Session identifier
            botchain_name: Botchain name
            org_name: Organization name
        """
        self.base_llm = base_llm
        self.agentcore_integration = agentcore_integration
        self.session_id = session_id
        self.botchain_name = botchain_name
        self.org_name = org_name
        self.use_agentcore = agentcore_integration and agentcore_integration.is_enabled()
        
        # Copy attributes from base LLM
        if hasattr(base_llm, 'model'):
            self.model = base_llm.model
        if hasattr(base_llm, '_opts'):
            self._opts = base_llm._opts
    
    async def chat(
        self,
        ctx: ChatContext,
        fnc_ctx: Optional[Any] = None
    ) -> ChatContext:
        """
        Intercept chat call and route to AgentCore if enabled
        """
        if self.use_agentcore:
            # Extract user message from context
            user_message = None
            conversation_history = []
            
            for msg in ctx.messages:
                if msg.role == ChatRole.USER:
                    user_message = msg.content
                # Build conversation history
                conversation_history.append({
                    "role": "user" if msg.role == ChatRole.USER else "assistant",
                    "content": msg.content
                })
            
            if user_message:
                # Call AgentCore
                response = await self.agentcore_integration.process_user_message(
                    user_text=user_message,
                    session_id=self.session_id,
                    botchain_name=self.botchain_name,
                    org_name=self.org_name,
                    conversation_history=conversation_history[:-1] if conversation_history else []  # Exclude current message
                )
                
                if response:
                    # AgentCore processed it - create response
                    ctx.messages.append(
                        ChatMessage(role=ChatRole.ASSISTANT, content=response)
                    )
                    return ctx
        
        # Fallback to base LLM
        return await self.base_llm.chat(ctx, fnc_ctx)

