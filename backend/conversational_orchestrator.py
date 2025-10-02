"""
Conversational Orchestrator - The Intelligent Brain of the Voice Agent

This module provides an intelligent orchestration layer that sits above the flow system,
FAQ bot, and general conversation. It makes smart decisions about:
- When to use structured flows vs. conversational AI
- How to handle user refusals and preferences
- Context preservation and smart conversation resumption
- Seamless integration with client's FAQ knowledge base

Architecture:
    User Input → Orchestrator → {Flow System, FAQ Bot, General Conversation}
    
The orchestrator maintains rich user state, conversation context, and makes
intelligent routing decisions to provide a natural, human-like experience.
"""

import json
import time
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime
from dataclasses import dataclass, field, asdict
from enum import Enum
import logging

logger = logging.getLogger(__name__)


# ============================================================================
# ENUMS & CONSTANTS
# ============================================================================

class OrchestratorAction(str, Enum):
    """Actions the orchestrator can take"""
    USE_FAQ = "use_faq"
    EXECUTE_FLOW = "execute_flow"
    HANDLE_CONVERSATIONALLY = "handle_conversationally"
    HANDLE_REFUSAL = "handle_refusal"
    HANDLE_UNCERTAINTY = "handle_uncertainty"
    NAVIGATE_BACK = "navigate_back"
    SUMMARIZE_CONTEXT = "summarize_context"


class ConversationObjective(str, Enum):
    """High-level conversation objectives"""
    LEARN_ABOUT_ALIVE5 = "learn_about_alive5"
    GET_SALES_INFO = "get_sales_info"
    GET_MARKETING_INFO = "get_marketing_info"
    SPEAK_WITH_PERSON = "speak_with_person"
    GENERAL_INQUIRY = "general_inquiry"
    UNCLEAR = "unclear"


# ============================================================================
# USER PROFILE & STATE MANAGEMENT
# ============================================================================

@dataclass
class UserProfile:
    """
    Rich user profile that tracks everything we know about the user.
    This enables intelligent, context-aware conversations.
    """
    # Basic collected information
    collected_info: Dict[str, Any] = field(default_factory=dict)
    # e.g., {"name": "John", "email": "john@example.com", "campaign_count": 26}
    
    # User preferences and behaviors
    preferences: List[str] = field(default_factory=list)
    # e.g., ["prefers_text_updates", "wants_quick_responses", "informal_tone"]
    
    # Fields user explicitly refused to provide
    refused_fields: List[str] = field(default_factory=list)
    # e.g., ["name", "phone_number"]
    
    # Fields user skipped or was uncertain about
    skipped_fields: List[str] = field(default_factory=list)
    # e.g., ["budget_amount"]
    
    # Conversation objectives (what user wants)
    objectives: List[str] = field(default_factory=list)
    # e.g., ["learn_about_pricing", "get_marketing_info"]
    
    # Conversation summary (rolling context)
    conversation_summary: str = ""
    
    # Metadata
    first_seen: float = field(default_factory=time.time)
    last_updated: float = field(default_factory=time.time)
    interaction_count: int = 0
    
    def update(self, **kwargs):
        """Update profile fields"""
        for key, value in kwargs.items():
            if hasattr(self, key):
                setattr(self, key, value)
        self.last_updated = time.time()
    
    def add_collected_info(self, key: str, value: Any):
        """Add a piece of collected information"""
        self.collected_info[key] = value
        self.last_updated = time.time()
        logger.info(f"👤 USER PROFILE: Added {key} = {value}")
    
    def mark_refused(self, field: str):
        """Mark a field as refused by the user"""
        if field not in self.refused_fields:
            self.refused_fields.append(field)
            logger.info(f"👤 USER PROFILE: User refused field: {field}")
    
    def mark_skipped(self, field: str):
        """Mark a field as skipped"""
        if field not in self.skipped_fields:
            self.skipped_fields.append(field)
            logger.info(f"👤 USER PROFILE: User skipped field: {field}")
    
    def add_objective(self, objective: str):
        """Add a conversation objective"""
        if objective not in self.objectives:
            self.objectives.append(objective)
            logger.info(f"👤 USER PROFILE: Added objective: {objective}")
    
    def to_dict(self) -> dict:
        """Convert to dictionary for serialization"""
        return asdict(self)
    
    @classmethod
    def from_dict(cls, data: dict) -> 'UserProfile':
        """Create from dictionary"""
        return cls(**data)


@dataclass
class OrchestratorDecision:
    """
    A decision made by the orchestrator about how to handle user input.
    This is the output of the intelligent decision-making process.
    """
    action: OrchestratorAction
    reasoning: str
    response: Optional[str] = None
    flow_to_execute: Optional[str] = None
    skip_fields: List[str] = field(default_factory=list)
    profile_updates: Dict[str, Any] = field(default_factory=dict)
    next_objective: Optional[str] = None
    confidence: float = 1.0
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> dict:
        """Convert to dictionary"""
        return {
            "action": self.action.value,
            "reasoning": self.reasoning,
            "response": self.response,
            "flow_to_execute": self.flow_to_execute,
            "skip_fields": self.skip_fields,
            "profile_updates": self.profile_updates,
            "next_objective": self.next_objective,
            "confidence": self.confidence,
            "metadata": self.metadata
        }


# ============================================================================
# CONVERSATIONAL ORCHESTRATOR
# ============================================================================

class ConversationalOrchestrator:
    """
    The intelligent brain that orchestrates all conversations.
    
    This class sits above the flow system and makes intelligent decisions about:
    - Routing to FAQ bot, flow system, or general conversation
    - Handling user refusals, preferences, and context
    - Maintaining conversation state and user profile
    - Providing natural, human-like interactions
    
    The orchestrator uses LLM-powered decision making to understand user intent,
    context, and conversation dynamics in real-time.
    """
    
    def __init__(self, available_flows: Dict[str, Any], faq_bot_available: bool = True):
        """
        Initialize the orchestrator.
        
        Args:
            available_flows: Dictionary of available flow configurations
            faq_bot_available: Whether the FAQ bot is available
        """
        self.available_flows = available_flows
        self.faq_bot_available = faq_bot_available
        self.user_profiles: Dict[str, UserProfile] = {}
        
        logger.info("🧠 ORCHESTRATOR: Initialized with intelligent decision-making")
    
    def get_or_create_profile(self, room_name: str) -> UserProfile:
        """Get or create a user profile for a room"""
        if room_name not in self.user_profiles:
            self.user_profiles[room_name] = UserProfile()
            logger.info(f"👤 ORCHESTRATOR: Created new user profile for {room_name}")
        return self.user_profiles[room_name]
    
    async def process_message(
        self,
        user_message: str,
        room_name: str,
        conversation_history: List[Dict],
        current_flow_state: Optional[Dict] = None,
        current_step_data: Optional[Dict] = None
    ) -> OrchestratorDecision:
        """
        Process a user message and make an intelligent decision.
        
        This is the main entry point for the orchestrator. It analyzes the user's
        message in context and decides the best action to take.
        
        Args:
            user_message: The user's message
            room_name: The room/session identifier
            conversation_history: Full conversation history
            current_flow_state: Current flow state (if in a flow)
            current_step_data: Current step data (if in a flow)
            
        Returns:
            OrchestratorDecision with the recommended action
        """
        profile = self.get_or_create_profile(room_name)
        profile.interaction_count += 1
        
        logger.info(f"🧠 ORCHESTRATOR: Processing message for {room_name}")
        logger.info(f"🧠 ORCHESTRATOR: User message: '{user_message}'")
        logger.info(f"🧠 ORCHESTRATOR: Current objectives: {profile.objectives}")
        logger.info(f"🧠 ORCHESTRATOR: Collected info: {profile.collected_info}")
        logger.info(f"🧠 ORCHESTRATOR: Refused fields: {profile.refused_fields}")
        
        # Build context for decision making
        context = self._build_context(
            user_message=user_message,
            profile=profile,
            conversation_history=conversation_history,
            current_flow_state=current_flow_state,
            current_step_data=current_step_data
        )
        
        # Import here to avoid circular dependency
        from backend.llm_utils import make_orchestrator_decision
        
        # Get decision from LLM
        decision = await make_orchestrator_decision(context)
        
        # Update user profile based on decision
        if decision.profile_updates:
            for key, value in decision.profile_updates.items():
                profile.add_collected_info(key, value)
        
        if decision.skip_fields:
            for field in decision.skip_fields:
                profile.mark_refused(field)
        
        if decision.next_objective:
            profile.add_objective(decision.next_objective)
        
        logger.info(f"🧠 ORCHESTRATOR: Decision - Action: {decision.action}, Confidence: {decision.confidence}")
        logger.info(f"🧠 ORCHESTRATOR: Reasoning: {decision.reasoning}")
        
        return decision
    
    def _build_context(
        self,
        user_message: str,
        profile: UserProfile,
        conversation_history: List[Dict],
        current_flow_state: Optional[Dict],
        current_step_data: Optional[Dict]
    ) -> Dict[str, Any]:
        """
        Build a rich context object for decision making.
        
        This context contains everything the LLM needs to make an intelligent decision.
        """
        # Get recent conversation (last 10 messages for context)
        recent_history = conversation_history[-10:] if len(conversation_history) > 10 else conversation_history
        
        # Format conversation history for LLM
        formatted_history = []
        for msg in recent_history:
            role = msg.get("role", "unknown")
            content = msg.get("content", "")
            formatted_history.append(f"{role.upper()}: {content}")
        
        context = {
            "user_message": user_message,
            "profile": {
                "collected_info": profile.collected_info,
                "preferences": profile.preferences,
                "refused_fields": profile.refused_fields,
                "skipped_fields": profile.skipped_fields,
                "objectives": profile.objectives,
                "interaction_count": profile.interaction_count
            },
            "conversation_history": formatted_history,
            "available_flows": list(self.available_flows.keys()) if self.available_flows else [],
            "faq_available": self.faq_bot_available,
            "current_flow": None,
            "current_step": None,
            "current_question": None
        }
        
        # Add current flow context if available
        if current_flow_state:
            context["current_flow"] = current_flow_state.get("current_flow")
            context["current_step"] = current_flow_state.get("current_step")
        
        if current_step_data:
            context["current_question"] = current_step_data.get("text")
            context["current_step_type"] = current_step_data.get("type")
            context["expected_answers"] = list(current_step_data.get("answers", {}).keys()) if current_step_data.get("answers") else None
        
        return context
    
    def update_profile_from_flow_response(
        self,
        room_name: str,
        step_name: str,
        user_response: str,
        extracted_value: Any = None
    ):
        """
        Update user profile when we collect information through flows.
        
        This ensures the profile stays in sync with flow progression.
        """
        profile = self.get_or_create_profile(room_name)
        
        # Store the response
        key = step_name.replace("question_", "").replace("_", " ").title()
        value = extracted_value if extracted_value is not None else user_response
        
        profile.add_collected_info(key, value)
        
        logger.info(f"👤 ORCHESTRATOR: Updated profile from flow - {key}: {value}")
    
    def get_profile_summary(self, room_name: str) -> str:
        """
        Get a human-readable summary of what we know about the user.
        
        This can be used to provide context or to show the user what we've collected.
        """
        profile = self.get_or_create_profile(room_name)
        
        summary_parts = []
        
        if profile.objectives:
            summary_parts.append(f"Objectives: {', '.join(profile.objectives)}")
        
        if profile.collected_info:
            info_str = ", ".join([f"{k}: {v}" for k, v in profile.collected_info.items()])
            summary_parts.append(f"Collected: {info_str}")
        
        if profile.refused_fields:
            summary_parts.append(f"Refused: {', '.join(profile.refused_fields)}")
        
        return " | ".join(summary_parts) if summary_parts else "New conversation"
    
    def should_skip_field(self, room_name: str, field_name: str) -> bool:
        """
        Check if a field should be skipped based on user preferences.
        
        This allows flows to adapt based on what users have refused.
        """
        profile = self.get_or_create_profile(room_name)
        
        # Check if user explicitly refused this field
        if field_name in profile.refused_fields:
            logger.info(f"🔒 ORCHESTRATOR: Skipping field '{field_name}' - user refused")
            return True
        
        # Check if we already collected this
        if field_name in profile.collected_info:
            logger.info(f"✅ ORCHESTRATOR: Skipping field '{field_name}' - already collected")
            return True
        
        return False
    
    def get_collected_value(self, room_name: str, field_name: str) -> Optional[Any]:
        """
        Get a previously collected value for a field.
        
        This enables smart resumption without re-asking questions.
        """
        profile = self.get_or_create_profile(room_name)
        return profile.collected_info.get(field_name)


# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def create_orchestrator_from_template(bot_template: Dict) -> ConversationalOrchestrator:
    """
    Create an orchestrator instance from a bot template.
    
    This extracts available flows from the template and initializes the orchestrator.
    """
    flows = {}
    
    # Extract flows from template's "data" structure
    if bot_template and "data" in bot_template:
        template_data = bot_template["data"]
        
        for flow_key, flow_data in template_data.items():
            if isinstance(flow_data, dict):
                flow_type = flow_data.get("type")
                flow_text = flow_data.get("text", "")
                
                # Only include intent_bot types as flows (greeting is handled separately)
                if flow_type == "intent_bot":
                    flows[flow_text] = {
                        "type": flow_type,
                        "name": flow_text,
                        "key": flow_key,
                        "data": flow_data
                    }
                    logger.info(f"🧠 ORCHESTRATOR: Found flow - {flow_text} ({flow_type})")
    
    orchestrator = ConversationalOrchestrator(
        available_flows=flows,
        faq_bot_available=True  # Assume FAQ bot is always available
    )
    
    logger.info(f"🧠 ORCHESTRATOR: Created from template with {len(flows)} flows")
    return orchestrator


# ============================================================================
# MODULE EXPORTS
# ============================================================================

__all__ = [
    'ConversationalOrchestrator',
    'UserProfile',
    'OrchestratorDecision',
    'OrchestratorAction',
    'ConversationObjective',
    'create_orchestrator_from_template'
]

