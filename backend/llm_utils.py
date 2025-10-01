"""
LLM Utilities - Centralized LLM API calls for the voice agent system

This module contains all LLM-related functions to make the codebase more maintainable
and easier to fine-tune. All OpenAI API calls are centralized here.

Functions:
- analyze_transcription_quality: Check if transcription is complete and meaningful
- extract_answer_with_llm: Extract structured answers from user responses
- analyze_message_with_smart_processor: Smart contextual analysis of user messages
- detect_intent_with_llm: Detect user intent from available options
"""

import os
import json
import logging
from typing import Dict, Any, Optional, List
import openai
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Configure logging
logger = logging.getLogger(__name__)

# Get OpenAI API key
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

def get_openai_client():
    """Get configured OpenAI client"""
    return openai.OpenAI(api_key=OPENAI_API_KEY)


async def analyze_transcription_quality(transcribed_text: str) -> Dict[str, Any]:
    """
    Analyze if a transcribed text is complete, meaningful, and not garbled.
    
    Args:
        transcribed_text: The transcribed text to analyze
        
    Returns:
        Dict with 'is_complete' (bool) and 'confidence' (float)
    """
    try:
        client = get_openai_client()
        
        prompt = f"""You are a speech transcription quality analyzer. Your job is to determine if a transcribed text is complete, meaningful, and not garbled.

TRANSCRIBED TEXT: "{transcribed_text}"

Analyze this text and determine:
1. Is it a complete thought/sentence?
2. Is it meaningful (not just random words or sounds)?
3. Is it not heavily garbled or corrupted?

Consider these examples:
- "Hello" → Complete and meaningful
- "I need help with" → Incomplete (cut off)
- "Um, uh, I was" → Incomplete and unclear
- "asdfghjkl" → Garbled/nonsensical
- "Can you help me with my account?" → Complete and meaningful

Respond with ONLY a JSON object:
{{
    "is_complete": true/false,
    "confidence": 0.0-1.0,
    "reasoning": "brief explanation"
}}"""

        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.1,
            max_tokens=200
        )
        
        response_text = response.choices[0].message.content.strip()
        logger.info(f"🔍 TRANSCRIPTION QUALITY: LLM response: {response_text}")
        
        try:
            result = json.loads(response_text)
            return result
        except json.JSONDecodeError:
            logger.warning(f"🔍 TRANSCRIPTION QUALITY: Invalid JSON response: {response_text}")
            return {"is_complete": False, "confidence": 0.0, "reasoning": "Invalid LLM response"}
            
    except Exception as e:
        logger.error(f"🔍 TRANSCRIPTION QUALITY: Error analyzing transcription: {e}")
        return {"is_complete": False, "confidence": 0.0, "reasoning": f"Error: {str(e)}"}


def extract_answer_with_llm(question_text: str, user_text: str) -> Dict[str, Any]:
    """
    Enhanced LLM-only answer extraction with comprehensive prompting.
    
    This replaces the hybrid approach with a single, well-tuned LLM call.
    
    Args:
        question_text: The question being asked
        user_text: The user's natural language response
        
    Returns:
        Dict with status, kind, value, confidence
    """
    try:
        client = get_openai_client()
        
        system = """You are an expert at extracting structured answers from natural language responses.

Your job is to analyze user responses and extract the intended answer with high accuracy.

EXTRACTION RULES:
1. NUMBERS: Convert words to digits ("zero" → 0, "fifty one" → 51, "twenty five" → 25)
2. YES/NO: Convert to boolean ("yes" → true, "no" → false, "I need it" → true)
3. ZIP CODES: Extract 5-digit codes ("two five nine six three" → "25963")
4. QUANTITIES: Extract numbers from phrases ("around fifteen" → 15, "maybe ten" → 10)
5. TEXT: Clean up responses, remove filler words
6. CONFIDENCE: Rate your confidence in the extraction (0.0-1.0)

EDGE CASES TO HANDLE:
- Incomplete responses: "the", "uh can i" → status: "unclear"
- Garbled speech: "some some", "through phone lines" → status: "unclear"
- Ambiguous numbers: "around fifteen" → extract 15 with medium confidence
- Context-dependent answers: "I need it" → true (for yes/no questions)

Return ONLY a JSON object with these keys:
- status: "extracted" | "unclear"
- kind: "number" | "boolean" | "zip" | "text" | "ambiguous"
- value: the extracted value (number, boolean, string)
- confidence: 0.0-1.0 (how certain you are)

EXAMPLES:
Q: "How many campaigns?" A: "zero" → {"status": "extracted", "kind": "number", "value": 0, "confidence": 0.95}
Q: "How many campaigns?" A: "fifty one" → {"status": "extracted", "kind": "number", "value": 51, "confidence": 0.95}
Q: "Do you need special needs?" A: "yes" → {"status": "extracted", "kind": "boolean", "value": true, "confidence": 0.95}
Q: "Do you need special needs?" A: "I need it" → {"status": "extracted", "kind": "boolean", "value": true, "confidence": 0.9}
Q: "What's your ZIP?" A: "two five nine six three" → {"status": "extracted", "kind": "zip", "value": "25963", "confidence": 0.95}
Q: "How many campaigns?" A: "the" → {"status": "unclear", "kind": "ambiguous", "value": "the", "confidence": 0.0}
Q: "How many campaigns?" A: "around fifteen" → {"status": "extracted", "kind": "number", "value": 15, "confidence": 0.8}"""

        user_prompt = f"""QUESTION: "{question_text}"
USER RESPONSE: "{user_text}"

Extract the answer following the rules above. Be especially careful with:
- Number conversion (words to digits)
- Yes/no detection (including implied answers)
- ZIP code extraction (5 digits)
- Confidence scoring (be honest about uncertainty)

Respond with JSON only."""

        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user_prompt},
            ],
            temperature=0.1,  # Lower temperature for more consistent results
            max_tokens=150
        )
        
        response_text = response.choices[0].message.content.strip()
        logger.info(f"🔍 ENHANCED LLM: Response: {response_text}")
        
        try:
            result = json.loads(response_text)
            
            # Validate the result structure
            if not isinstance(result, dict):
                raise ValueError("Response is not a dictionary")
            
            required_keys = ["status", "kind", "value", "confidence"]
            if not all(key in result for key in required_keys):
                raise ValueError(f"Missing required keys: {required_keys}")
            
            # Validate status
            if result["status"] not in ["extracted", "unclear"]:
                result["status"] = "unclear"
            
            # Validate confidence
            if not isinstance(result["confidence"], (int, float)) or not 0 <= result["confidence"] <= 1:
                result["confidence"] = 0.0
            
            return result
            
        except (json.JSONDecodeError, ValueError) as e:
            logger.warning(f"🔍 ENHANCED LLM: Invalid response: {response_text}, Error: {e}")
            return {
                "status": "unclear",
                "kind": "ambiguous",
                "value": user_text,
                "confidence": 0.0
            }
            
    except Exception as e:
        logger.error(f"🔍 ENHANCED LLM: Error: {e}")
        return {
            "status": "unclear",
            "kind": "ambiguous",
            "value": user_text,
            "confidence": 0.0
        }


async def analyze_message_with_smart_processor(user_message: str, context_info: str, intents_list: str) -> Dict[str, Any]:
    """
    Analyze user message with smart contextual understanding.
    
    Args:
        user_message: The user's message to analyze
        context_info: Context about current flow state
        intents_list: Available intents for the system
        
    Returns:
        Dict with intent_detected, message_type, confidence, action, reasoning
    """
    try:
        client = get_openai_client()
        
        prompt = f"""You are a smart conversation analyzer. Analyze the user's message and determine the best response strategy.

{context_info}

USER MESSAGE: "{user_message}"

AVAILABLE INTENTS: {intents_list}

ANALYSIS TASKS:
1. INTENT DETECTION: Does this message indicate a clear intent from the available list?
2. CONTEXT UNDERSTANDING: Is this a response to a question, filler/stuttering, simple greeting, or a new topic?
3. RESPONSE STRATEGY: What should the agent do next?

CRITICAL CONTEXT RULES:
- If user is already in a flow (Flow_1, Flow_2, Flow_3, etc.) and responding to a question, treat as "question_response" with "continue_flow"
- If user mentions menu items (pasta, biryani, qorma, fried rice) while in menu flow, treat as "question_response" with intent_detected="none"
- If user is answering a question about their choice, treat as "question_response" with "continue_flow" and intent_detected="none"
- INTENT DETECTION vs RESPONSE DISTINCTION:
  * "What's in the menu?" → intent_detected="menu" (requesting menu information)
  * "I'll take biryani" → intent_detected="none" (selecting from menu)
  * "Give me pasta" → intent_detected="none" (selecting from menu)
  * "Can I see the menu?" → intent_detected="menu" (requesting menu information)
- Only treat as "new_topic" if user is clearly starting a completely different conversation
- Simple greetings like "Hi", "Hi there", "Hello" should be treated as natural conversation flow, not filtered out. Let the flow continue naturally.

RESPONSE FORMAT (JSON):
{{
    "intent_detected": "intent_name|none",
    "message_type": "intent_request|question_response|filler|unclear|new_topic|greeting",
    "confidence": "high|medium|low",
    "action": "continue_flow|switch_intent|ask_clarification|ignore|respond_naturally",
    "reasoning": "brief explanation of the analysis"
}}

EXAMPLES:
- "Yeah, I'm looking for someone to help" → {{"intent_detected": "agent", "message_type": "intent_request", "confidence": "high", "action": "switch_intent", "reasoning": "Clear request for human help"}}
- "Yeah" → {{"intent_detected": "none", "message_type": "question_response", "confidence": "medium", "action": "continue_flow", "reasoning": "Simple affirmation to current question"}}
- "What's in the menu?" → {{"intent_detected": "menu", "message_type": "intent_request", "confidence": "high", "action": "switch_intent", "reasoning": "User is requesting menu information"}}
- "Can I see the menu?" → {{"intent_detected": "menu", "message_type": "intent_request", "confidence": "high", "action": "switch_intent", "reasoning": "User is requesting menu information"}}
- "Pasta" (while in menu flow) → {{"intent_detected": "none", "message_type": "question_response", "confidence": "high", "action": "continue_flow", "reasoning": "User is selecting menu item, continue with order flow"}}
- "I'll take the biryani" (while in menu flow) → {{"intent_detected": "none", "message_type": "question_response", "confidence": "high", "action": "continue_flow", "reasoning": "User is making menu selection, continue with order flow"}}
- "Give me pasta" (while in menu flow) → {{"intent_detected": "none", "message_type": "question_response", "confidence": "high", "action": "continue_flow", "reasoning": "User is selecting menu item, continue with order flow"}}
- "Hi" → {{"intent_detected": "none", "message_type": "greeting", "confidence": "high", "action": "continue_flow", "reasoning": "Simple greeting, let flow continue naturally"}}
- "Hi there" → {{"intent_detected": "none", "message_type": "greeting", "confidence": "high", "action": "continue_flow", "reasoning": "Simple greeting, let flow continue naturally"}}
- "Uh, I, uh, I was asking" → {{"intent_detected": "none", "message_type": "filler", "confidence": "high", "action": "ignore", "reasoning": "Stuttering/filler, not meaningful content"}}
- "Can I speak with someone?" → {{"intent_detected": "agent", "message_type": "intent_request", "confidence": "high", "action": "switch_intent", "reasoning": "Direct request for human agent"}}
- "Can I speak with someone over the phone?" → {{"intent_detected": "agent", "message_type": "intent_request", "confidence": "high", "action": "switch_intent", "reasoning": "Clear request to speak with human agent"}}
- "Connect me to an agent" → {{"intent_detected": "agent", "message_type": "intent_request", "confidence": "high", "action": "switch_intent", "reasoning": "Direct request for agent connection"}}
- "I'm looking for someone to speak" → {{"intent_detected": "agent", "message_type": "intent_request", "confidence": "high", "action": "switch_intent", "reasoning": "Request to speak with someone"}}

Respond with ONLY the JSON object, no other text."""

        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[{"role": "user", "content": prompt}],
            max_tokens=200,
            temperature=0.1
        )
        
        response_text = response.choices[0].message.content.strip()
        logger.info(f"🧠 SMART PROCESSOR: LLM response: {response_text}")
        
        try:
            result = json.loads(response_text)
            return result
        except json.JSONDecodeError:
            logger.warning(f"🧠 SMART PROCESSOR: Invalid JSON response: {response_text}")
            return {
                "intent_detected": "none",
                "message_type": "unclear",
                "confidence": "low",
                "action": "ask_clarification",
                "reasoning": "Invalid LLM response"
            }
            
    except Exception as e:
        logger.error(f"🧠 SMART PROCESSOR: Error analyzing message: {e}")
        return {
            "intent_detected": "none",
            "message_type": "unclear",
            "confidence": "low",
            "action": "ask_clarification",
            "reasoning": f"Error: {str(e)}"
        }


async def detect_intent_with_llm(user_message: str, intent_mapping: Dict[str, str]) -> Optional[Dict[str, Any]]:
    """
    Detect user intent from available options using LLM.
    
    Args:
        user_message: The user's message to analyze
        intent_mapping: Mapping of intent names to their descriptions
        
    Returns:
        Dict with intent information or None if no intent detected
    """
    try:
        client = get_openai_client()
        
        intent_list = ", ".join(intent_mapping.keys())
        
        prompt = f"""You are an intent detection system. Analyze the user's message and determine which intent they want.

AVAILABLE INTENTS: {intent_list}

INTENT DESCRIPTIONS:
{chr(10).join([f"- {intent}: {desc}" for intent, desc in intent_mapping.items()])}

RULES:
1. Return the EXACT intent name from the list above: {intent_list}
2. If no intent matches, return "none"
3. If it's just a greeting, return "greeting"
4. Do NOT return "agent" or any other intent not in the list above

IMPORTANT: The user said: "{user_message}"
Think about what they really want. Are they asking to speak to a person? Do they want sales information? Do they want marketing information?

Respond with ONLY the exact intent name from the list above (case-insensitive), "greeting", or "none" if no intent matches.

Examples:
- "Can I speak with Affan?" → Speak with Affan (if available, they want to talk to Affan specifically)
- "Can I speak with someone over the phone?" → agent (if available, they want to talk to a human)
- "Can I speak with someone, please?" → agent (if available, they want human help)
- "Connect me to an agent" → agent (if available, they want human help)
- "I'm looking for someone to speak" → agent (if available, they want human help)
- "Can you connect me with someone?" → agent (if available, they want human help)
- "Sales information, please" → sales (if available)
- "Can I get the marketing information?" → marketing (if available)
- "Hello there" → greeting
- "I need help with billing" → agent (if available, they want human help)
"""

        logger.info(f"🔍 INTENT DETECTION: Analyzing message '{user_message}' for intents: {intent_list}")
        logger.info(f"🔍 INTENT DETECTION: Available intents mapping: {list(intent_mapping.keys())}")
        print(f"🔍 INTENT DETECTION: Available intents: {intent_list}")
        print(f"🔍 INTENT DETECTION: User message: '{user_message}'")

        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[{"role": "user", "content": prompt}],
            max_tokens=50,
            temperature=0.0
        )
        
        detected_intent = response.choices[0].message.content.strip()
        logger.info(f"🔍 INTENT DETECTION: LLM response: '{detected_intent}'")
        print(f"🔍 INTENT DETECTION: LLM returned: '{detected_intent}'")
        
        # Handle special "greeting" response
        if detected_intent.lower() == "greeting":
            return {"type": "greeting", "intent": "greeting"}
        
        # Check if detected intent is in our mapping
        if detected_intent.lower() in [intent.lower() for intent in intent_mapping.keys()]:
            # Find the exact case match
            for intent_key in intent_mapping.keys():
                if intent_key.lower() == detected_intent.lower():
                    logger.info(f"🔍 INTENT DETECTION: ✅ Intent found: '{intent_key}'")
                    return {"type": "intent", "intent": intent_key}
        
        # Check for "none" response
        if detected_intent.lower() == "none":
            logger.info(f"🔍 INTENT DETECTION: ❌ No intent found")
            return None
        
        # If we get here, the LLM returned something unexpected
        logger.warning(f"🔍 INTENT DETECTION: Unexpected response: '{detected_intent}'")
        return None
        
    except Exception as e:
        logger.error(f"🔍 INTENT DETECTION: Error detecting intent: {e}")
        return None


# All LLM functions are now centralized here for simplicity
