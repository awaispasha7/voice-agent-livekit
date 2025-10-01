"""
Orchestrator Integration Example

This file shows how to integrate the Conversational Orchestrator
into your existing flow processing logic.

Copy these patterns into your process_flow_message() function.
"""

from backend.conversational_orchestrator import OrchestratorAction

# ============================================================================
# INTEGRATION PATTERN 1: Main Flow Processing
# ============================================================================

async def process_flow_message_with_orchestrator(
    room_name: str,
    user_message: str,
    flow_state: FlowState,
    bot_template: dict
):
    """
    Enhanced flow processing with orchestrator intelligence.
    
    This replaces the rigid flow logic with intelligent decision-making.
    """
    global conversational_orchestrator
    
    # ============================================================================
    # STEP 1: Get Orchestrator Decision
    # ============================================================================
    
    if conversational_orchestrator:
        logger.info("🧠 ORCHESTRATOR: Analyzing message for intelligent routing...")
        
        decision = await conversational_orchestrator.process_message(
            user_message=user_message,
            room_name=room_name,
            conversation_history=flow_state.conversation_history,
            current_flow_state={
                "current_flow": flow_state.current_flow,
                "current_step": flow_state.current_step
            },
            current_step_data=flow_state.flow_data
        )
        
        logger.info(f"🧠 ORCHESTRATOR: Decision → {decision.action}")
        logger.info(f"🧠 ORCHESTRATOR: Reasoning → {decision.reasoning}")
        logger.info(f"🧠 ORCHESTRATOR: Confidence → {decision.confidence}")
        
        # ========================================================================
        # STEP 2: Act on Decision
        # ========================================================================
        
        # ────────────────────────────────────────────────────────────────────
        # ACTION 1: Route to FAQ Bot
        # ────────────────────────────────────────────────────────────────────
        if decision.action == OrchestratorAction.USE_FAQ:
            logger.info("🔀 ORCHESTRATOR: Routing to FAQ bot (Bedrock knowledge base)")
            
            # Call your existing FAQ bot function
            faq_response = await call_faq_bot(
                bot_id=FAQ_BOT_ID,
                question=user_message
            )
            
            if faq_response and faq_response.get("response"):
                response_text = faq_response["response"]
                
                # Add to conversation history
                add_agent_response_to_history(flow_state, response_text)
                
                return {
                    "type": "faq_response",
                    "response": response_text,
                    "flow_state": flow_state,
                    "orchestrator_decision": decision.to_dict()
                }
            else:
                # FAQ bot failed, fallback to conversational
                logger.warning("⚠️ FAQ bot returned no response, using fallback")
                response_text = "I'm here to help! Could you please rephrase your question?"
                
                return {
                    "type": "conversational",
                    "response": response_text,
                    "flow_state": flow_state
                }
        
        # ────────────────────────────────────────────────────────────────────
        # ACTION 2: Execute a Structured Flow
        # ────────────────────────────────────────────────────────────────────
        elif decision.action == OrchestratorAction.EXECUTE_FLOW:
            flow_name = decision.flow_to_execute
            logger.info(f"🔀 ORCHESTRATOR: Starting flow → {flow_name}")
            
            # Find the intent in bot template
            intent_data = find_intent_in_template(bot_template, flow_name)
            
            if intent_data:
                # Update flow state
                flow_state.current_flow = intent_data["flow_key"]
                flow_state.current_step = intent_data["name"]
                flow_state.flow_data = intent_data
                
                # Check if intent has next_flow (first question)
                if intent_data.get("next_flow"):
                    next_flow = intent_data["next_flow"]
                    flow_state.current_step = next_flow.get("name")
                    flow_state.flow_data = next_flow
                    
                    response_text = next_flow.get("text", "How can I help you?")
                else:
                    response_text = intent_data.get("text", "How can I help you?")
                
                # Add to conversation history
                add_agent_response_to_history(flow_state, response_text)
                
                # Update user profile with objective
                if decision.next_objective:
                    conversational_orchestrator.get_or_create_profile(room_name).add_objective(
                        decision.next_objective
                    )
                
                return {
                    "type": "flow_started",
                    "flow_name": flow_name,
                    "response": response_text,
                    "flow_state": flow_state,
                    "orchestrator_decision": decision.to_dict()
                }
            else:
                logger.error(f"❌ Flow '{flow_name}' not found in template")
                return {
                    "type": "error",
                    "response": "I'm having trouble finding that information. Could you try rephrasing?",
                    "flow_state": flow_state
                }
        
        # ────────────────────────────────────────────────────────────────────
        # ACTION 3: Handle Conversationally
        # ────────────────────────────────────────────────────────────────────
        elif decision.action == OrchestratorAction.HANDLE_CONVERSATIONALLY:
            logger.info("🔀 ORCHESTRATOR: Handling conversationally")
            
            response_text = decision.response or "I'm here to help! What would you like to know?"
            
            # Apply any profile updates
            if decision.profile_updates:
                profile = conversational_orchestrator.get_or_create_profile(room_name)
                for key, value in decision.profile_updates.items():
                    profile.add_collected_info(key, value)
            
            # Mark any refused fields
            if decision.skip_fields:
                profile = conversational_orchestrator.get_or_create_profile(room_name)
                for field in decision.skip_fields:
                    profile.mark_refused(field)
            
            # Add to conversation history
            add_agent_response_to_history(flow_state, response_text)
            
            return {
                "type": "conversational",
                "response": response_text,
                "flow_state": flow_state,
                "orchestrator_decision": decision.to_dict()
            }
        
        # ────────────────────────────────────────────────────────────────────
        # ACTION 4: Handle Refusal
        # ────────────────────────────────────────────────────────────────────
        elif decision.action == OrchestratorAction.HANDLE_REFUSAL:
            logger.info("🔀 ORCHESTRATOR: User refused to provide information")
            
            response_text = decision.response or "That's perfectly fine! We can continue without it."
            
            # Mark refused fields in profile
            if decision.skip_fields:
                profile = conversational_orchestrator.get_or_create_profile(room_name)
                for field in decision.skip_fields:
                    profile.mark_refused(field)
                    logger.info(f"🔒 ORCHESTRATOR: Marked field '{field}' as refused")
            
            # Continue to next question in flow (skip current one)
            if flow_state.flow_data and flow_state.flow_data.get("next_flow"):
                next_flow = flow_state.flow_data["next_flow"]
                flow_state.current_step = next_flow.get("name")
                flow_state.flow_data = next_flow
                
                # Combine refusal acknowledgment with next question
                response_text += " " + next_flow.get("text", "")
            
            # Add to conversation history
            add_agent_response_to_history(flow_state, response_text)
            
            return {
                "type": "refusal_handled",
                "response": response_text,
                "flow_state": flow_state,
                "orchestrator_decision": decision.to_dict()
            }
        
        # ────────────────────────────────────────────────────────────────────
        # ACTION 5: Handle Uncertainty
        # ────────────────────────────────────────────────────────────────────
        elif decision.action == OrchestratorAction.HANDLE_UNCERTAINTY:
            logger.info("🔀 ORCHESTRATOR: User expressed uncertainty")
            
            response_text = decision.response or "That's okay! If you're not sure, we can move on."
            
            # Offer to skip or continue with approximation
            if flow_state.flow_data and flow_state.flow_data.get("next_flow"):
                next_flow = flow_state.flow_data["next_flow"]
                flow_state.current_step = next_flow.get("name")
                flow_state.flow_data = next_flow
                
                response_text += " Let's continue. " + next_flow.get("text", "")
            
            # Add to conversation history
            add_agent_response_to_history(flow_state, response_text)
            
            return {
                "type": "uncertainty_handled",
                "response": response_text,
                "flow_state": flow_state,
                "orchestrator_decision": decision.to_dict()
            }
    
    # ============================================================================
    # FALLBACK: No Orchestrator Available
    # ============================================================================
    else:
        logger.warning("⚠️ Orchestrator not available, using legacy flow logic")
        # Fall back to existing flow processing logic
        return await process_flow_message_legacy(room_name, user_message, flow_state)


# ============================================================================
# INTEGRATION PATTERN 2: Smart Field Skipping
# ============================================================================

async def ask_question_with_orchestrator_check(
    room_name: str,
    field_name: str,
    question_text: str,
    flow_state: FlowState
):
    """
    Smart question asking that respects user preferences.
    
    Checks if field was previously refused or already collected.
    """
    global conversational_orchestrator
    
    if conversational_orchestrator:
        # Check if we should skip this field
        if conversational_orchestrator.should_skip_field(room_name, field_name):
            logger.info(f"⏭️ ORCHESTRATOR: Skipping field '{field_name}' (refused or already collected)")
            
            # Check if we have a value already
            existing_value = conversational_orchestrator.get_collected_value(room_name, field_name)
            
            if existing_value:
                logger.info(f"✅ ORCHESTRATOR: Using existing value for '{field_name}': {existing_value}")
                return {
                    "skipped": True,
                    "reason": "already_collected",
                    "value": existing_value
                }
            else:
                logger.info(f"🔒 ORCHESTRATOR: Field '{field_name}' was refused by user")
                return {
                    "skipped": True,
                    "reason": "user_refused",
                    "value": None
                }
    
    # Field not skipped, ask the question
    return {
        "skipped": False,
        "question": question_text
    }


# ============================================================================
# INTEGRATION PATTERN 3: Profile-Aware Responses
# ============================================================================

async def generate_response_with_profile_context(
    room_name: str,
    base_response: str
):
    """
    Enhance responses with user profile context.
    
    Makes conversations feel more personal and aware.
    """
    global conversational_orchestrator
    
    if conversational_orchestrator:
        profile = conversational_orchestrator.get_or_create_profile(room_name)
        
        # Personalize based on collected info
        if profile.collected_info.get("name"):
            base_response = base_response.replace(
                "Hello!",
                f"Hello {profile.collected_info['name']}!"
            )
        
        # Add context if we know their objective
        if profile.objectives:
            latest_objective = profile.objectives[-1]
            logger.info(f"👤 ORCHESTRATOR: User's objective → {latest_objective}")
        
        # Get full profile summary
        summary = conversational_orchestrator.get_profile_summary(room_name)
        logger.info(f"👤 ORCHESTRATOR: Profile → {summary}")
    
    return base_response


# ============================================================================
# INTEGRATION PATTERN 4: Session Resume
# ============================================================================

async def resume_session_with_orchestrator(room_name: str):
    """
    Resume a conversation intelligently using stored profile.
    
    Perfect for returning users.
    """
    global conversational_orchestrator
    
    if conversational_orchestrator:
        profile = conversational_orchestrator.get_or_create_profile(room_name)
        
        if profile.interaction_count > 0:
            # Returning user
            logger.info(f"👤 ORCHESTRATOR: Returning user (interaction #{profile.interaction_count})")
            
            # Get what we know
            collected = profile.collected_info
            objectives = profile.objectives
            
            if objectives:
                return {
                    "greeting": f"Welcome back! We were discussing {objectives[-1]}. Ready to continue?",
                    "resume_flow": True
                }
            elif collected:
                return {
                    "greeting": "Welcome back! How can I help you today?",
                    "resume_flow": False
                }
        
        # New user
        logger.info(f"👤 ORCHESTRATOR: New user")
        return {
            "greeting": "Welcome! How can I help you today?",
            "resume_flow": False
        }
    
    return {"greeting": "Hello!", "resume_flow": False}


# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def find_intent_in_template(bot_template: dict, intent_name: str) -> Optional[dict]:
    """Find intent in bot template by name"""
    if not bot_template or "data" not in bot_template:
        return None
    
    for flow_key, flow_data in bot_template["data"].items():
        if flow_data.get("type") == "intent_bot":
            # Check if name matches
            if flow_data.get("text", "").lower() == intent_name.lower():
                return {
                    "flow_key": flow_key,
                    **flow_data
                }
    
    return None


def add_agent_response_to_history(flow_state: FlowState, response: str):
    """Add agent response to conversation history"""
    if not flow_state.conversation_history:
        flow_state.conversation_history = []
    
    flow_state.conversation_history.append({
        "role": "assistant",
        "content": response,
        "timestamp": datetime.now().isoformat()
    })


# ============================================================================
# USAGE EXAMPLE
# ============================================================================

"""
# In your process_flow_message function:

async def process_flow_message(room_name, user_message, ...):
    # ... template loading, state management ...
    
    # 🧠 Use orchestrator for intelligent processing
    result = await process_flow_message_with_orchestrator(
        room_name=room_name,
        user_message=user_message,
        flow_state=flow_state,
        bot_template=bot_template
    )
    
    return result
"""

