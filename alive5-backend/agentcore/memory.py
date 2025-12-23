"""
AgentCore Memory Integration
Replaces in-memory session storage with AgentCore Memory
"""

import os
import logging
from typing import Dict, Any, Optional, List
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables
env_path = Path(__file__).parent.parent.parent / ".env"
if env_path.exists():
    load_dotenv(env_path)

logger = logging.getLogger("agentcore-memory")

try:
    from bedrock_agentcore.memory import MemoryClient
    AGENTCORE_MEMORY_AVAILABLE = True
except ImportError:
    AGENTCORE_MEMORY_AVAILABLE = False
    logger.warning("bedrock-agentcore memory not available. Will use fallback storage.")


class AgentCoreMemory:
    """AgentCore Memory client for session and conversation state"""
    
    def __init__(self):
        self.enabled = os.getenv("USE_AGENTCORE_MEMORY", "true").lower() == "true"
        self.memory_client = None
        
        if self.enabled and AGENTCORE_MEMORY_AVAILABLE:
            try:
                self.memory_client = MemoryClient()
                logger.info("✅ AgentCore Memory enabled")
            except Exception as e:
                logger.warning(f"⚠️ Failed to initialize AgentCore Memory: {e}")
                self.enabled = False
        else:
            logger.info("ℹ️ AgentCore Memory disabled - using fallback storage")
    
    async def store_session(
        self,
        session_id: str,
        session_data: Dict[str, Any]
    ) -> bool:
        """
        Store session data in AgentCore Memory
        
        Args:
            session_id: Session/room identifier
            session_data: Session data to store
            
        Returns:
            True if successful
        """
        if not self.enabled or not self.memory_client:
            return False
        
        try:
            # Store as short-term memory event
            await self.memory_client.store_event(
                session_id=session_id,
                event_type="session_update",
                content=session_data
            )
            return True
        except Exception as e:
            logger.error(f"Error storing session in AgentCore Memory: {e}")
            return False
    
    async def get_session(self, session_id: str) -> Optional[Dict[str, Any]]:
        """
        Retrieve session data from AgentCore Memory
        
        Args:
            session_id: Session/room identifier
            
        Returns:
            Session data or None
        """
        if not self.enabled or not self.memory_client:
            return None
        
        try:
            # Retrieve recent events for this session
            events = await self.memory_client.retrieve_events(
                session_id=session_id,
                limit=1
            )
            
            if events:
                # Get the most recent session update
                for event in events:
                    if event.get("event_type") == "session_update":
                        return event.get("content")
            
            return None
        except Exception as e:
            logger.error(f"Error retrieving session from AgentCore Memory: {e}")
            return None
    
    async def store_conversation_turn(
        self,
        session_id: str,
        role: str,
        content: str,
        metadata: Optional[Dict[str, Any]] = None
    ) -> bool:
        """
        Store a conversation turn in AgentCore Memory
        
        Args:
            session_id: Session identifier
            role: 'user' or 'assistant'
            content: Message content
            metadata: Optional metadata
            
        Returns:
            True if successful
        """
        if not self.enabled or not self.memory_client:
            return False
        
        try:
            await self.memory_client.store_event(
                session_id=session_id,
                event_type="conversation_turn",
                content={
                    "role": role,
                    "content": content,
                    "metadata": metadata or {}
                }
            )
            return True
        except Exception as e:
            logger.error(f"Error storing conversation turn: {e}")
            return False
    
    async def get_conversation_history(
        self,
        session_id: str,
        limit: int = 20
    ) -> List[Dict[str, Any]]:
        """
        Retrieve conversation history from AgentCore Memory
        
        Args:
            session_id: Session identifier
            limit: Maximum number of turns to retrieve
            
        Returns:
            List of conversation messages
        """
        if not self.enabled or not self.memory_client:
            return []
        
        try:
            events = await self.memory_client.retrieve_events(
                session_id=session_id,
                event_type="conversation_turn",
                limit=limit
            )
            
            # Convert events to conversation format
            history = []
            for event in events:
                content = event.get("content", {})
                history.append({
                    "role": content.get("role"),
                    "content": content.get("content")
                })
            
            return history
        except Exception as e:
            logger.error(f"Error retrieving conversation history: {e}")
            return []
    
    async def store_long_term_memory(
        self,
        session_id: str,
        memory_type: str,
        content: Dict[str, Any]
    ) -> bool:
        """
        Store long-term memory (persists across sessions)
        
        Args:
            session_id: Session identifier
            memory_type: Type of memory (e.g., "user_preferences", "collected_data")
            content: Memory content
            
        Returns:
            True if successful
        """
        if not self.enabled or not self.memory_client:
            return False
        
        try:
            await self.memory_client.store_memory(
                session_id=session_id,
                memory_type=memory_type,
                content=content
            )
            return True
        except Exception as e:
            logger.error(f"Error storing long-term memory: {e}")
            return False
    
    async def retrieve_long_term_memory(
        self,
        session_id: str,
        memory_type: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """
        Retrieve long-term memories
        
        Args:
            session_id: Session identifier
            memory_type: Optional filter by memory type
            
        Returns:
            List of memories
        """
        if not self.enabled or not self.memory_client:
            return []
        
        try:
            memories = await self.memory_client.retrieve_memories(
                session_id=session_id,
                memory_type=memory_type
            )
            return memories
        except Exception as e:
            logger.error(f"Error retrieving long-term memory: {e}")
            return []
    
    def is_enabled(self) -> bool:
        """Check if AgentCore Memory is enabled"""
        return self.enabled

