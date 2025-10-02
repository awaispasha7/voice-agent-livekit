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


def match_answer_with_llm(question_text: str, user_response: str, available_answers: Dict[str, Any]) -> Optional[str]:
    """
    Use LLM to match user response with available answer options.
    
    Args:
        question_text: The question being asked
        user_response: The user's natural language response
        available_answers: Dict of answer keys and their data
        
    Returns:
        The matching answer key, or None if no match found
    """
    try:
        client = get_openai_client()
        
        # Extract just the answer keys for the LLM
        answer_keys = list(available_answers.keys())
        
        system = """You are an expert at matching user responses to predefined answer options.

Your job is to analyze a user's response and determine which predefined answer option it matches.

MATCHING RULES:
1. EXACT MATCHES: "zero" matches "0", "five" matches "5"
2. RANGE MATCHES: "around fifteen" matches "11-20" (if 15 is in that range), "ten" matches "1-10" (if 10 is in that range)
3. THRESHOLD MATCHES: "about thirty" matches "More than 21" (if 30 > 21), "26" matches "More than 21" (if 26 > 21)
4. CONTEXT MATCHES: "I'm not running any" matches "0"
5. CONFIDENCE: Only return a match if you're confident (confidence > 0.7)

CRITICAL: When matching numbers to ranges:
- If the number is EXACTLY at the boundary (e.g., 10 for range "1-10"), match it to that range
- If the number exceeds all ranges (e.g., 26 > 21), match to "More than X" option
- Return ONLY the answer key EXACTLY as it appears in the options list

EXAMPLES:
Q: "How many campaigns?" A: "zero" Options: ["0", "1-10", "11-20", "More than 21"] → "0"
Q: "How many campaigns?" A: "five" Options: ["0", "1-10", "11-20", "More than 21"] → "1-10"
Q: "How many campaigns?" A: "ten" Options: ["0", "1-10", "11-20", "More than 21"] → "1-10"
Q: "How many campaigns?" A: "around fifteen" Options: ["0", "1-10", "11-20", "More than 21"] → "11-20"
Q: "How many campaigns?" A: "26" Options: ["0", "1-10", "11-20", "More than 21"] → "More than 21"
Q: "How many campaigns?" A: "about thirty" Options: ["0", "1-10", "11-20", "More than 21"] → "More than 21"
Q: "How many campaigns?" A: "I'm not running any" Options: ["0", "1-10", "11-20", "More than 21"] → "0"
Q: "How many campaigns?" A: "the" Options: ["0", "1-10", "11-20", "More than 21"] → "none"
"""

        user_prompt = f"""QUESTION: "{question_text}"
USER RESPONSE: "{user_response}"
AVAILABLE ANSWER OPTIONS: {answer_keys}

Which answer option does the user's response match? Return only the answer key or "none"."""

        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user_prompt}
            ],
            temperature=0.1,
            max_tokens=50
        )
        
        result = response.choices[0].message.content.strip()
        
        # Strip surrounding quotes if present (LLM sometimes adds them)
        result = result.strip('"').strip("'")
        
        # Debug logging
        logger.info(f"🔍 ANSWER MATCHING: LLM returned: '{result}' for response: '{user_response}'")
        logger.info(f"🔍 ANSWER MATCHING: Available keys: {answer_keys}")
        logger.info(f"🔍 ANSWER MATCHING: Result in keys: {result in answer_keys}")
        
        # Validate the result
        if result == "none" or result not in answer_keys:
            logger.info(f"🔍 ANSWER MATCHING: ❌ No valid match found (result: '{result}')")
            return None
            
        logger.info(f"🔍 ANSWER MATCHING: ✅ Matched to: '{result}'")
        return result
        
    except Exception as e:
        logger.error(f"🔍 ANSWER MATCHING: Error matching answer: {e}")
        return None


def detect_uncertainty_with_llm(user_message: str, question_text: str) -> bool:
    """
    Use LLM to detect if user is expressing uncertainty or inability to answer.
    
    Args:
        user_message: The user's response
        question_text: The question being asked
        
    Returns:
        True if user is uncertain/unable to answer, False otherwise
    """
    try:
        client = get_openai_client()
        
        system = """You are an expert at detecting when users express uncertainty or inability to answer a question.

Your job is to determine if the user is expressing that they:
- Don't know the answer
- Are unsure or uncertain
- Can't provide the information
- Need help figuring it out

EXAMPLES OF UNCERTAINTY:
- "I don't know"
- "I'm not sure"
- "I have no idea"
- "Can't say"
- "Uncertain"
- "Not certain"
- "I'm unsure"
- "I don't have that information"
- "I need to check"
- "I'm not really sure about that"
- "No clue"
- "Beats me"

EXAMPLES OF CLEAR ANSWERS (NOT UNCERTAINTY):
- "Five"
- "About ten"
- "Around 100 dollars"
- "Yes"
- "No"
- "Maybe later" (this is a decision, not uncertainty about information)

Return ONLY "uncertain" or "certain" (no other text)."""

        user_prompt = f"""QUESTION: "{question_text}"
USER RESPONSE: "{user_message}"

Is the user expressing uncertainty or inability to answer? Return only "uncertain" or "certain"."""

        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user_prompt}
            ],
            temperature=0.1,
            max_tokens=10
        )
        
        result = response.choices[0].message.content.strip().lower()
        
        is_uncertain = result == "uncertain"
        
        logger.info(f"🔍 UNCERTAINTY DETECTION: User message: '{user_message}' → Result: '{result}' → Is uncertain: {is_uncertain}")
        
        return is_uncertain
        
    except Exception as e:
        logger.error(f"🔍 UNCERTAINTY DETECTION: Error: {e}")
        # Fallback to simple phrase detection if LLM fails
        uncertainty_phrases = ["don't know", "not sure", "unsure", "no idea"]
        return any(phrase in user_message.lower() for phrase in uncertainty_phrases)


# ============================================================================
# ORCHESTRATOR LLM FUNCTIONS
# ============================================================================

def extract_user_data_with_llm(user_message: str) -> Dict[str, Any]:
    """
    Extract comprehensive user information from natural language using LLM.
    
    This replaces hardcoded regex patterns with intelligent LLM-based extraction
    that can understand context and extract various types of user information.
    
    Args:
        user_message: The user's message to analyze
        
    Returns:
        Dict with extracted user information (name, email, phone, company, etc.)
    """
    try:
        client = get_openai_client()
        
        system = """You are an expert at extracting user information from natural language conversations.

Your job is to identify and extract any personal or business information mentioned by the user.

EXTRACTION TARGETS:
1. **Personal Information:**
   - Name: "My name is John", "I'm Sarah", "Call me Mike"
   - Email: "john@example.com", "reach me at sarah@company.org"
   - Phone: "(555) 123-4567", "call me at 555-987-6543", "+1-800-555-0123"
   - Location: "I'm from New York", "ZIP code 10001", "I live in California"

2. **Business Information:**
   - Company: "I work at Microsoft", "My company is Acme Corp", "We are Google"
   - Role/Title: "I'm a marketing manager", "I work as a software engineer"
   - Website: "www.acme.com", "our site is company.org"
   - Industry: "We're in healthcare", "I'm in the tech industry"

3. **Quantitative Information:**
   - Budget: "$50,000", "our budget is 25k", "we have 10k to spend"
   - Quantities: "25 campaigns", "we have 100 employees", "50 customers"
   - Percentages: "25% increase", "we're at 80% capacity"
   - Timeframes: "ASAP", "urgent", "this week", "next month"

4. **Preferences/Needs:**
   - Requirements: "I need help with", "I want to", "I'm looking for"
   - Preferences: "I prefer", "I would like", "we need"

EXTRACTION RULES:
1. Extract ONLY information explicitly mentioned or clearly implied
2. Clean up extracted values (remove filler words, standardize format)
3. For names, extract first name only unless full name is clearly stated
4. For phone numbers, preserve original format but ensure it's valid
5. For companies, extract the main company name (first 2-3 words max)
6. For roles, extract the main job title (first 2-3 words max)
7. For budgets, include currency symbol if mentioned
8. For quantities, include the unit if mentioned
9. Be conservative - only extract what you're confident about

Return ONLY a JSON object with extracted information. Use these exact keys:
- name: First name only
- email: Email address
- phone: Phone number (any format)
- company: Company name
- role: Job title/role
- website: Website URL
- zip_code: ZIP/postal code
- budget: Budget amount with currency
- quantity: Number with unit (e.g., "25 campaigns")
- percentage: Percentage value
- timeline: Time-related urgency/timeline
- preference: What they need/want

Only include keys for information that was actually found. If no information is found, return an empty object {}.

EXAMPLES:
Input: "Hi, my name is John Smith and I work at Microsoft as a software engineer"
Output: {"name": "John", "company": "Microsoft", "role": "Software Engineer"}

Input: "You can reach me at john@example.com or call (555) 123-4567"
Output: {"email": "john@example.com", "phone": "(555) 123-4567"}

Input: "Our budget is $50,000 and we have 25 campaigns running"
Output: {"budget": "$50,000", "quantity": "25 campaigns"}

Input: "I need this done ASAP for our company Acme Corp"
Output: {"company": "Acme Corp", "timeline": "ASAP"}

Input: "Hello there, how are you?"
Output: {}"""

        user_prompt = f"""Extract user information from this message:

"{user_message}"

Analyze the message and extract any personal, business, or quantitative information mentioned. Be precise and only extract information that is clearly stated or strongly implied.

Respond with JSON only."""

        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user_prompt},
            ],
            temperature=0.1,  # Low temperature for consistent extraction
            max_tokens=300
        )
        
        response_text = response.choices[0].message.content.strip()
        logger.info(f"🔍 USER DATA EXTRACTION: LLM response: {response_text}")
        
        try:
            result = json.loads(response_text)
            logger.info(f"🔍 USER DATA EXTRACTION: Extracted data: {result}")
            return result
        except json.JSONDecodeError as e:
            logger.error(f"🔍 USER DATA EXTRACTION: JSON decode error: {e}")
            logger.error(f"🔍 USER DATA EXTRACTION: Raw response: {response_text}")
            return {}
            
    except Exception as e:
        logger.error(f"🔍 USER DATA EXTRACTION: Error using LLM: {e}")
        return {}


async def make_orchestrator_decision(context: Dict[str, Any]) -> Any:
    """
    Make an intelligent orchestration decision using LLM.
    
    This is the brain of the conversational orchestrator. It analyzes the full
    context and decides the best action to take.
    
    Args:
        context: Rich context including user message, profile, history, flow state
        
    Returns:
        OrchestratorDecision with recommended action and reasoning
    """
    # Import here to avoid circular dependency
    from backend.conversational_orchestrator import OrchestratorDecision, OrchestratorAction
    
    try:
        client = get_openai_client()
        
        # Build comprehensive system prompt
        system = """You are an intelligent conversation orchestrator for Alive5, a company that provides AI-powered solutions.

**YOUR ROLE:**
You sit above all conversation systems and make intelligent routing decisions to provide
a natural, human-like conversation experience.

**YOUR CAPABILITIES:**

1. **FAQ Bot (Bedrock Knowledge Base)**: Client's knowledge base about Alive5
   - Company information, products, features, pricing
   - Use when user asks about: "What does Alive5 do?", "pricing", "features", etc.

2. **Flow System**: Structured data collection flows
   - Available flows: sales, marketing, speak with person
   - Use when user wants: sales info, marketing info, to speak with someone
   - BUT: Be intelligent about it!
     * Never re-ask already collected information
     * If user refuses a field, skip it gracefully
     * Adapt based on user preferences and context

3. **General Conversation**: Natural dialogue handling
   - Clarifications, preferences, small talk
   - Refusals ("I don't want to give my name")
   - Uncertainty ("I'm not sure", "I don't know")
   - Context questions ("What were we talking about?")
   - Navigation ("Can we go back?", "Skip this")

**DECISION CRITERIA:**

**Route to FAQ Bot when:**
- User asks about Alive5: products, features, pricing, company info
- Questions like: "What is Alive5?", "How much?", "What do you offer?"
- Any knowledge-base answerable question

**Execute Flow when:**
- User expresses intent: "I want sales info", "Tell me about marketing", "Speak with manager"
- Clear objective that maps to a structured flow
- User ready to provide information

**Handle Conversationally when:**
- User refuses: "I don't want to", "No thanks", "Skip that"
- User uncertain: "I'm not sure", "I don't know", "Maybe"
- User navigating: "Go back", "What did you ask?", "Can you repeat?"
- Small talk or clarifications
- Context questions

**INTELLIGENT BEHAVIORS (CRITICAL):**

1. **Memory**: NEVER re-ask for information already in user profile
2. **Respect**: If user refused a field, acknowledge and move on
3. **Context**: Remember conversation from 10+ messages ago
4. **Adaptation**: Adjust flow based on user preferences
5. **Natural**: Respond like a human, not a robot
6. **Smooth Transitions**: Handle topic changes gracefully

**RESPONSE FORMAT (JSON):**
{{
    "action": "use_faq | execute_flow | handle_conversationally | handle_refusal | handle_uncertainty",
    "reasoning": "Why this action? (1-2 sentences)",
    "response": "What to say to user (if handle_conversationally)",
    "flow_to_execute": "sales | marketing | speak_with_person | null",
    "skip_fields": ["field_name"],
    "profile_updates": {{"key": "value"}},
    "next_objective": "What user wants to accomplish",
    "confidence": 0.95
}}

**EXAMPLES:**

📌 **Example 1: FAQ Request**
User: "What does Alive5 do?"
Profile: {{}}
→ {{"action": "use_faq", "reasoning": "Direct question about Alive5 company information", "confidence": 0.99}}

📌 **Example 2: Flow Request**
User: "I want marketing information"
Profile: {{}}
→ {{"action": "execute_flow", "reasoning": "Clear intent for marketing info", "flow_to_execute": "marketing", "next_objective": "get_marketing_info", "confidence": 0.95}}

📌 **Example 3: Refusal Handling**
User: "I'd rather not share my name"
Current Question: "May I have your name?"
→ {{"action": "handle_conversationally", "reasoning": "User refusing to provide name", "response": "That's perfectly fine! We can continue without it.", "skip_fields": ["name"], "profile_updates": {{"prefers_privacy": true}}, "confidence": 0.98}}

📌 **Example 4: Uncertainty**
User: "I'm not sure how many campaigns"
Current Question: "How many campaigns are you running?"
→ {{"action": "handle_conversationally", "reasoning": "User uncertain about answer", "response": "That's okay! If you don't have an exact number, a rough estimate works too. Or we can move on to the next question.", "confidence": 0.92}}

📌 **Example 5: Smart Resume (Already Collected)**
User: "What's my budget?"
Profile: {{"collected_info": {{"budget": "5000"}}}}
→ {{"action": "handle_conversationally", "reasoning": "User asking about already provided info", "response": "You mentioned your budget is $5000. Would you like to update that?", "confidence": 0.95}}

📌 **Example 6: Context Switch**
User: "Actually, tell me about your pricing first"
Current Flow: "marketing" (in middle of questions)
→ {{"action": "use_faq", "reasoning": "User wants to learn about pricing before continuing", "confidence": 0.90, "metadata": {{"resume_flow_after": "marketing"}}}}

**CRITICAL RULES:**
1. ALWAYS check profile.collected_info before deciding to execute flow
2. ALWAYS respect profile.refused_fields
3. ALWAYS provide reasoning
4. BE NATURAL - avoid robotic responses
5. If uncertain, prefer handle_conversationally over forcing a flow"""

        # Build user prompt with full context
        user_prompt = f"""**CURRENT CONTEXT:**

**User Message:** "{context.get('user_message')}"

**User Profile:**
- Collected Info: {context['profile'].get('collected_info', {})}
- Preferences: {context['profile'].get('preferences', [])}
- Refused Fields: {context['profile'].get('refused_fields', [])}
- Skipped Fields: {context['profile'].get('skipped_fields', [])}
- Current Objectives: {context['profile'].get('objectives', [])}
- Interaction Count: {context['profile'].get('interaction_count', 0)}

**Available Systems:**
- FAQ Bot: {"Available" if context.get('faq_available') else "Not Available"}
- Available Flows: {context.get('available_flows', [])}

**Current Conversation State:**
- Current Flow: {context.get('current_flow', 'None')}
- Current Step: {context.get('current_step', 'None')}
- Current Question: {context.get('current_question', 'None')}
- Expected Answers: {context.get('expected_answers', 'None')}

**Recent Conversation History (Last 10 messages):**
{chr(10).join(context.get('conversation_history', [])[-10:])}

---

**YOUR TASK:**
Analyze this context and decide the best action. Return ONLY valid JSON (no markdown, no extra text).

**Return format:**
{{
    "action": "use_faq | execute_flow | handle_conversationally | handle_refusal | handle_uncertainty",
    "reasoning": "...",
    "response": "..." (if applicable),
    "flow_to_execute": "..." (if applicable),
    "skip_fields": [...],
    "profile_updates": {{}},
    "next_objective": "...",
    "confidence": 0.0-1.0
}}"""

        response = client.chat.completions.create(
            model="gpt-4o",  # Using GPT-4 for best decision-making
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user_prompt}
            ],
            temperature=0.3,  # Lower temperature for more consistent decisions
            max_tokens=500
        )
        
        result_text = response.choices[0].message.content.strip()
        
        # Remove markdown code blocks if present
        if result_text.startswith("```"):
            result_text = result_text.split("```")[1]
            if result_text.startswith("json"):
                result_text = result_text[4:]
            result_text = result_text.strip()
        
        logger.info(f"🧠 ORCHESTRATOR LLM: Raw response: {result_text[:200]}...")
        
        # Parse JSON response
        decision_data = json.loads(result_text)
        
        # Create OrchestratorDecision object
        decision = OrchestratorDecision(
            action=OrchestratorAction(decision_data.get("action", "handle_conversationally")),
            reasoning=decision_data.get("reasoning", "No reasoning provided"),
            response=decision_data.get("response"),
            flow_to_execute=decision_data.get("flow_to_execute"),
            skip_fields=decision_data.get("skip_fields", []),
            profile_updates=decision_data.get("profile_updates", {}),
            next_objective=decision_data.get("next_objective"),
            confidence=decision_data.get("confidence", 0.8),
            metadata=decision_data.get("metadata", {})
        )
        
        logger.info(f"🧠 ORCHESTRATOR LLM: Decision - {decision.action} (confidence: {decision.confidence})")
        logger.info(f"🧠 ORCHESTRATOR LLM: Reasoning - {decision.reasoning}")
        
        return decision
        
    except json.JSONDecodeError as e:
        logger.error(f"🧠 ORCHESTRATOR LLM: JSON decode error: {e}")
        logger.error(f"🧠 ORCHESTRATOR LLM: Raw response was: {result_text}")
        
        # Fallback decision
        return OrchestratorDecision(
            action=OrchestratorAction.HANDLE_CONVERSATIONALLY,
            reasoning="Failed to parse LLM response, using safe fallback",
            response="I'm here to help! Could you please rephrase that?",
            confidence=0.5
        )
        
    except Exception as e:
        logger.error(f"🧠 ORCHESTRATOR LLM: Error: {e}")
        
        # Fallback decision
        return OrchestratorDecision(
            action=OrchestratorAction.HANDLE_CONVERSATIONALLY,
            reasoning=f"Error in orchestrator: {str(e)}",
            response="I'm having trouble processing that. Could you try again?",
            confidence=0.3
        )


# All LLM functions are now centralized here for simplicity
