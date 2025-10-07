"""
LLM Utilities - Centralized LLM API calls for the voice agent system

This module centralizes ALL OpenAI calls so prompts can be tuned in one place.
Every function uses verbose, rule-driven system prompts (with edge cases and examples)
to minimize ambiguity and ensure stable JSON outputs.

Covered functions:
- analyze_transcription_quality (async)
- extract_answer_with_llm (sync)
- match_answer_with_llm (sync)
- detect_intent_with_llm (async)
- detect_uncertainty_with_llm (sync)
- extract_user_data_with_llm (sync)
- generate_conversational_response (async)
- make_orchestrator_decision (async)  ← includes speak_with_person action
"""

import os
import json
import logging
from typing import Dict, Any, Optional
import openai
from dotenv import load_dotenv

# ──────────────────────────────────────────────────────────────────────────────
# Setup
# ──────────────────────────────────────────────────────────────────────────────
load_dotenv()
logger = logging.getLogger(__name__)
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

def get_openai_client():
    """Return configured OpenAI client"""
    return openai.OpenAI(api_key=OPENAI_API_KEY)


# ──────────────────────────────────────────────────────────────────────────────
# 1) Transcription Quality (async)
# ──────────────────────────────────────────────────────────────────────────────
async def analyze_transcription_quality(transcribed_text: str) -> Dict[str, Any]:
    """
    Evaluate whether a transcript is complete, meaningful, and not garbled.

    Returns JSON:
    {
      "is_complete": bool,
      "confidence": float (0.0-1.0),
      "reasoning": "short explanation"
    }
    """
    try:
        client = get_openai_client()
        system = """You are a speech transcription quality analyst.

GOAL:
- Judge if a transcription is (a) complete and (b) meaningful (not just fragments/fillers),
  and (c) not garbled.

DEFINITIONS:
- Complete: A standalone, coherent phrase/sentence (even if short).
- Meaningful: Carries intent or semantic content (not just fillers or random chars).
- Garbled: Corrupted, repeated nonsense, keyboard mash, or heavy phonetic debris.

RULES:
1) Single-word greetings like "Hello" or "Hi" are complete & meaningful.
2) Cut-off phrases like "I need help with" are incomplete.
3) Fillers ("um", "uh", "erm") without content are incomplete/low confidence.
4) Nonsense strings ("asdfghjkl", "sdf sdf sdf") are garbled.
5) Clear questions/requests ("Can you reset my password?") are complete & meaningful.
6) If the text seems clipped mid-thought → is_complete=false.
7) Confidence reflects how certain you are in your classification.

OUTPUT:
Return ONLY valid JSON:
{
  "is_complete": true/false,
  "confidence": 0.0-1.0,
  "reasoning": "short explanation"
}

EXAMPLES:
- "Hi" → {"is_complete": true, "confidence": 0.9, "reasoning": "Short greeting"}
- "I need to reset" → {"is_complete": false, "confidence": 0.7, "reasoning": "Cut off"}
- "asdfghj" → {"is_complete": false, "confidence": 0.95, "reasoning": "Garbled"}"""

        user = f'TRANSCRIBED TEXT: "{transcribed_text}"\nRespond in JSON only.'
        resp = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "system", "content": system},
                      {"role": "user", "content": user}],
            temperature=0.1,
            max_tokens=200
        )
        text = resp.choices[0].message.content.strip()
        # Strip code fences if any
        if text.startswith("```"):
            text = text.strip("`").replace("json", "").strip()
        return json.loads(text)
    except Exception as e:
        logger.error(f"Transcription quality error: {e}")
        return {"is_complete": False, "confidence": 0.0, "reasoning": str(e)}


# ──────────────────────────────────────────────────────────────────────────────
# 2) Answer Extraction (sync)
# ──────────────────────────────────────────────────────────────────────────────
def extract_answer_with_llm(question_text: str, user_text: str) -> Dict[str, Any]:
    """
    Extracts structured answers (number/boolean/zip/text) from natural responses.

    Returns JSON:
    {
      "status": "extracted"|"unclear",
      "kind": "number"|"boolean"|"zip"|"text"|"ambiguous",
      "value": number|boolean|string,
      "confidence": float
    }
    """
    try:
        client = get_openai_client()
        system = """You are an expert at extracting structured answers from natural language.

TASK:
Given a QUESTION and a USER RESPONSE, extract the intended answer precisely.

EXTRACTION RULES:
A) NUMBERS:
   - Convert words to digits: "twenty five"→25, "zero"→0, "ten"→10
   - Ranges like "around fifteen"→15 with lower confidence (e.g., 0.7-0.85)
B) YES/NO:
   - Map "yes", "yeah", "yup", "I need it", "I do" → true
   - Map "no", "nope", "don't need it", "I don't" → false
C) ZIP CODES:
   - Extract 5-digit patterns even when spoken: "two five nine six three"→"25963"
D) TEXT:
   - Clean filler words, keep concise
E) UNCLEAR:
   - Incomplete/garbled: "uh the", "asdf", "ummm" → status="unclear"
F) CONFIDENCE:
   - 0.9+ for direct, 0.7–0.85 for approximations, <0.5 if shaky

OUTPUT JSON ONLY:
{
  "status": "extracted"|"unclear",
  "kind": "number"|"boolean"|"zip"|"text"|"ambiguous",
  "value": ...,
  "confidence": 0.0-1.0
}

EXAMPLES:
Q: "How many campaigns?" A: "zero"
→ {"status":"extracted","kind":"number","value":0,"confidence":0.95}

Q: "Do you need SSO?" A:"I need it"
→ {"status":"extracted","kind":"boolean","value":true,"confidence":0.9}

Q: "ZIP?" A:"two five nine six three"
→ {"status":"extracted","kind":"zip","value":"25963","confidence":0.95}

Q: "How many lines?" A:"around fifteen"
→ {"status":"extracted","kind":"number","value":15,"confidence":0.8}

Q: "How many?" A:"uh the"
→ {"status":"unclear","kind":"ambiguous","value":"uh the","confidence":0.0}"""

        user = f"""QUESTION: "{question_text}"
USER RESPONSE: "{user_text}"
Return ONLY JSON per the schema."""
        resp = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[{"role": "system", "content": system},
                      {"role": "user", "content": user}],
            temperature=0.1,
            max_tokens=200
        )
        text = resp.choices[0].message.content.strip()
        if text.startswith("```"):
            text = text.strip("`").replace("json", "").strip()
        return json.loads(text)
    except Exception as e:
        logger.error(f"Answer extraction error: {e}")
        return {"status": "unclear", "kind": "ambiguous", "value": user_text, "confidence": 0.0}


# ──────────────────────────────────────────────────────────────────────────────
# 3) Answer Matching (sync)
# ──────────────────────────────────────────────────────────────────────────────
def match_answer_with_llm(question_text: str, user_response: str, available_answers: Dict[str, Any]) -> Optional[str]:
    """
    Matches user free-text to one of the predefined option keys.

    Returns:
      - matching key from available_answers, or
      - None if no confident match
    """
    try:
        client = get_openai_client()
        keys = list(available_answers.keys())
        system = f"""You map a user's response to a single option from a predefined list.

OPTIONS: {keys}

MATCHING RULES:
1) Exact numbers: "five"→"5", "0"→"0".
2) Range mapping: 15→"11-20"; 10→"1-10".
3) Threshold: 26 or "about thirty"→"More than 21".
4) Zero intent: "none", "not running any"→"0".
5) If not confident, return "none".

OUTPUT:
Return only the option key (verbatim) or "none".

EXAMPLES:
- User:"ten" Options:["0","1-10","11-20","More than 21"] → "1-10"
- User:"26"  → "More than 21"
- User:"we're not running any" → "0"
- User:"the" → "none\""""

        user = f"""Q: {question_text}
User: {user_response}
Options: {keys}
Return only the exact option key or "none"."""
        resp = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "system", "content": system},
                      {"role": "user", "content": user}],
            temperature=0.0,
            max_tokens=50
        )
        result = resp.choices[0].message.content.strip().strip('"').strip("'")
        return result if result in keys else None
    except Exception as e:
        logger.error(f"Answer matching error: {e}")
        return None


# ──────────────────────────────────────────────────────────────────────────────
# 4) Intent Detection (async)
# ──────────────────────────────────────────────────────────────────────────────
async def detect_intent_with_llm(user_message: str, intent_mapping: Dict[str, str]) -> Optional[Dict[str, Any]]:
    """
    Detects which intent (from provided list) the user expressed.

    Returns:
      {"type":"intent","intent":"<key>"} or {"type":"greeting","intent":"greeting"} or None
    """
    try:
        client = get_openai_client()
        system = """You are an intent detector constrained to a known list.

RULES:
- Return EXACTLY one of the provided intents when matched (case-insensitive ok; output must match key).
- If it's a greeting ("hi", "hello", "good morning", etc.) → return "greeting".
- If the user asks for a human/agent/representative, or to "talk to someone", "connect me", "speak with person" → return "speak_with_person".
- If no match, return "none".
- Output plain text only (no JSON).

EXAMPLES:
- "Can I speak to a person?" → speak_with_person
- "I want sales info" → sales   (if "sales" exists)
- "Tell me about marketing" → marketing (if present)
- "hello" → greeting
- "price?" (if you have explicit "pricing" intent) → pricing else none"""

        user = f"""AVAILABLE INTENTS: {list(intent_mapping.keys())}
USER: "{user_message}"
Return exactly one: an intent key, or "greeting", or "speak_with_person", or "none"."""
        resp = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[{"role": "system", "content": system},
                      {"role": "user", "content": user}],
            temperature=0.0,
            max_tokens=30
        )
        detected = resp.choices[0].message.content.strip().lower()
        if detected in [i.lower() for i in intent_mapping.keys()]:
            return {"type": "intent", "intent": detected}
        if detected == "greeting":
            return {"type": "greeting", "intent": "greeting"}
        if detected in ["speak_with_person", "agent", "human", "representative", "talk_to_someone", "connect_me"]:
            return {"type": "intent", "intent": "speak_with_person"}
        return None
    except Exception as e:
        logger.error(f"Intent detection error: {e}")
        return None


# ──────────────────────────────────────────────────────────────────────────────
# 5) Uncertainty Detection (sync)
# ──────────────────────────────────────────────────────────────────────────────
def detect_uncertainty_with_llm(user_message: str, question_text: str) -> bool:
    """
    Returns True if the user expresses uncertainty/inability to answer.
    """
    try:
        client = get_openai_client()
        system = """You detect if a response expresses uncertainty/inability.

UNCERTAINTY EXAMPLES:
- "I don't know", "not sure", "no idea", "unsure", "can't say", "beats me", "I need to check"

CERTAIN EXAMPLES:
- Clear number ("five"), boolean ("yes"/"no"), or concrete detail.

OUTPUT:
Return only "uncertain" or "certain"."""
        user = f'Question: "{question_text}"\nUser: "{user_message}"'
        resp = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "system", "content": system},
                      {"role": "user", "content": user}],
            temperature=0.0,
            max_tokens=5
        )
        verdict = resp.choices[0].message.content.strip().lower()
        return verdict == "uncertain"
    except Exception as e:
        logger.error(f"Uncertainty detection error: {e}")
        # Heuristic fallback
        return any(p in user_message.lower() for p in ["don't know", "not sure", "unsure", "no idea", "cant say", "can't say"])


# ──────────────────────────────────────────────────────────────────────────────
# 6) User Data Extraction (sync)
# ──────────────────────────────────────────────────────────────────────────────
def extract_user_data_with_llm(user_message: str) -> Dict[str, Any]:
    """
    Extract personal/business attributes from free-text.

    Keys (include only when found):
      name, email, phone, company, role, website, zip_code,
      budget, quantity, percentage, timeline, preference
    """
    try:
        client = get_openai_client()
        system = """You are an expert in extracting structured user information from natural language.

TARGET KEYS:
- name (first name unless full name is explicit),
- email, phone, company (2-3 words), role (main job title),
- website, zip_code, budget (keep currency), quantity (e.g., "25 campaigns"),
- percentage (e.g., "25%"), timeline ("ASAP", "next week"), preference (short phrase of need/want).

RULES:
- Only extract what is explicitly present or strongly implied.
- Clean filler words; keep formats intact (email/phone).
- Be conservative: if unsure, omit the key.
- Return JSON only with found keys (empty {} if nothing).

EXAMPLES:
- "Hi, I'm John Smith at Acme Corp. I'm a marketing manager." →
  {"name":"John","company":"Acme Corp","role":"Marketing Manager"}
- "Reach me at sara@org.org or +1 555 123 4567" →
  {"email":"sara@org.org","phone":"+1 555 123 4567"}
- "Budget is $50k; 25 campaigns running" →
  {"budget":"$50k","quantity":"25 campaigns"}"""
        user = f'MESSAGE: "{user_message}"\nReturn JSON with found keys only.'
        resp = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[{"role": "system", "content": system},
                      {"role": "user", "content": user}],
            temperature=0.1,
            max_tokens=250
        )
        text = resp.choices[0].message.content.strip()
        if text.startswith("```"):
            text = text.strip("`").replace("json", "").strip()
        return json.loads(text)
    except Exception as e:
        logger.error(f"User data extraction error: {e}")
        return {}


# ──────────────────────────────────────────────────────────────────────────────
# 7) Conversational Response (async)
# ──────────────────────────────────────────────────────────────────────────────
async def generate_conversational_response(user_message: str, context: Dict[str, Any]) -> str:
    """
    Produce a natural, human-sounding reply, respecting refusals & uncertainty.

    context expects:
      - conversation_history: list[{role, content}], last ~6 used
      - current_flow, current_step
      - profile (collected_info, objectives)
      - flags: refusal_context, uncertainty_context, skipped_fields
    """
    try:
        client = get_openai_client()
        system = """You are a warm, professional Alive5 Agent - an AI assistant for Alive5's voice support line.
STYLE:
- Friendly, concise, and natural (use contractions).
- Context-aware: reference prior turns when helpful.
- Respect privacy & preferences.
- Always identify as "Alive5 Agent" when introducing yourself or explaining capabilities.

SPECIAL HANDLING:
- Refusal (refusal_context=true):
  Acknowledge kindly ("No problem at all") and transition naturally to the next step.
  If next_step_text is provided, incorporate it smoothly into your response.
- Uncertainty (uncertainty_context=true):
  Reassure the user; simplify the next step or offer an alternative.
- Self-identification: When asked "who are you", "what's this", etc., respond:
  "I'm an Alive5 Agent here to help you with [relevant capabilities]"
- End call detection: If user says goodbye, thanks, "that's all", "bye", etc., respond with warm farewell:
  "You're welcome! Have a great day!" or "Thanks for calling! Take care!"
- Flow progression: If next_step_text is provided, acknowledge the user's input naturally and then ask the next question directly. For example: "Thanks for that information! [Next question here]"
- Flow message enhancement (is_flow_start=true): When enhancing flow messages, make them sound natural and conversational. Transform rigid flow text into warm, engaging dialogue while preserving the core intent and information. NEVER add meta-commentary like "Here's a more conversational version" or "Sure!".

AVOID:
- Robotic phrasing, long monologues, repeating the user verbatim.
- Generic "AI assistant" - always specify "Alive5 Agent"
- Adding meta-commentary like "Here's a more conversational version" or "Sure!" to normal responses

OUTPUT:
- A single, helpful response (1-2 sentences is ideal)."""

        history_lines = []
        for m in context.get("conversation_history", [])[-6:]:
            role = m.get("role", "unknown")
            content = m.get("content", "")
            history_lines.append(f"{role}: {content}")
        history = "\n".join(history_lines)

        flags = []
        if context.get("refusal_context"):
            flags.append("refusal_context=true")
        if context.get("uncertainty_context"):
            flags.append("uncertainty_context=true")
        flags_str = ", ".join(flags) if flags else "none"

        next_step_info = ""
        if context.get("next_step_text"):
            next_step_info = f"- Next step: {context.get('next_step_text')}"
        
        # Handle FAQ cleanup only
        if context.get("faq_cleanup_only"):
            user = f"""FAQ RESPONSE CLEANUP:
- Original FAQ response: "{user_message}"

CRITICAL INSTRUCTIONS:
- ONLY remove meta-commentary like "Based on the search results", "I can see that", "From what I can see", etc.
- DO NOT change, summarize, or rephrase the actual content
- DO NOT add any new information
- DO NOT add meta-commentary like "Here's a cleaned version"
- Simply remove the meta-commentary and return the clean response
- If no meta-commentary is present, return the response exactly as provided
- Output ONLY the cleaned response, nothing else"""
        # Handle flow message enhancement
        elif context.get("is_flow_start") and context.get("intent_detected"):
            user = f"""FLOW MESSAGE ENHANCEMENT:
- Intent detected: {context.get('intent_detected')}
- Original flow message: "{user_message}"
- Conversation history: {history}

CRITICAL INSTRUCTIONS:
- DO NOT add any meta-commentary like "Here's a more conversational version" or "Sure!"
- DO NOT rewrite or rephrase the message unless it's extremely rigid/technical
- If the message is already conversational, return it EXACTLY as provided
- Only make minor natural language improvements if the message sounds robotic
- Output ONLY the enhanced message, nothing else"""
        else:
            user = f"""CONTEXT:
- Flow: {context.get('current_flow')}
- Step: {context.get('current_step')}
- Profile.collected: {context.get('profile', {}).get('collected_info', {})}
- Flags: {flags_str}
{next_step_info}

Recent history:
{history}

USER MESSAGE: "{user_message}"

IMPORTANT: This is a normal conversational response request. Do NOT add meta-commentary like "Here's a more conversational version" or "Sure!". Simply respond naturally to the user's message."""
        resp = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "system", "content": system},
                      {"role": "user", "content": user}],
            temperature=0.7,
            max_tokens=160
        )
        return resp.choices[0].message.content.strip()
    except Exception as e:
        logger.error(f"Conversational response error: {e}")
        return "Got it. How would you like to proceed?"


# ──────────────────────────────────────────────────────────────────────────────
# 8) Orchestrator Decision (async)  — includes speak_with_person
# ──────────────────────────────────────────────────────────────────────────────

async def make_orchestrator_decision(context: Dict[str, Any]) -> Any:
    """
    Orchestrator decision maker.
    Decides one of: use_faq, execute_flow, handle_conversationally, handle_refusal,
                    handle_uncertainty, speak_with_person

    Returns OrchestratorDecision instance.
    """
    from backend.conversational_orchestrator import OrchestratorDecision, OrchestratorAction
    try:
        client = get_openai_client()
        system = """You are the CONVERSATIONAL ORCHESTRATOR above all components.

GOAL:
Choose exactly ONE action for the next step.

VALID ACTIONS:
- use_faq            → Use knowledge base for factual company/product/pricing questions.
- execute_flow       → Start/switch to a structured flow when user expresses a compatible intent.
- handle_conversationally
                     → Natural dialogue: greetings, clarifications, follow-ups, answers to current flow questions.
- handle_refusal     → User refuses a requested field ("I prefer not", "I won't share").
- handle_uncertainty → User expresses not knowing/being unsure.
- speak_with_person  → User asks to talk to a human or says "agent", "representative", "connect me", etc. (ONLY when no specific intent flow exists for the request).
- end_call           → User says goodbye, thanks, "that's all", "bye", "have a good day", etc. (conversation ending).

END CALL DETECTION:
- If user says goodbye, thanks, "that's all", "bye", "have a good day", "thank you", "that's it", "all done", etc. → end_call with farewell response.
- If user says "end call", "hang up", "disconnect", "terminate call" → speak_with_person (agent handoff).

GLOBAL RULES:
1) Check current_flow/current_step: if user is responding to a flow question, prefer handle_conversationally.
2) Never re-ask for info already in profile.collected_info; respect refused_fields.
3) If user refuses a field → handle_refusal and include that field in skip_fields.
4) If uncertain → handle_uncertainty.
5) If human handoff requested AND no specific intent flow exists for the request → speak_with_person.
6) If user asks factual questions about Alive5 (features/pricing/etc) → use_faq.
7) If user expresses an intent that maps to an available flow → execute_flow (preferred over speak_with_person).
8) VALIDATE FLOW RESPONSES: If user is responding to a flow question, validate their response dynamically:
   - If response is appropriate for the flow context → handle_conversationally (allow flow progression)
   - If response is inappropriate or unclear → handle_conversationally with helpful correction
9) Always include brief reasoning and a reasonable confidence (0.0–1.0).
10) Output STRICT JSON only.

OUTPUT JSON SCHEMA:
{
  "action": "use_faq|execute_flow|handle_conversationally|handle_refusal|handle_uncertainty|speak_with_person|end_call",
  "reasoning": "1-2 sentences why",
  "response": "<optional natural language reply or leave null>",
  "flow_to_execute": "<flow-name-or-null>",
  "skip_fields": ["field_a","field_b"],
  "profile_updates": {"k":"v"},
  "next_objective": "<optional objective string>",
  "confidence": 0.0-1.0
}

EXAMPLES:
A) FAQ
User: "What does Alive5 do?"
→ {"action":"use_faq","reasoning":"Direct company info question","confidence":0.98}

B) Execute Flow
User: "I want marketing information"
→ {"action":"execute_flow","reasoning":"Clear intent for marketing flow","flow_to_execute":"marketing","next_objective":"get_marketing_info","confidence":0.95}

H) Execute Specific Intent Flow (preferred over speak_with_person)
User: "speak with manager"
→ {"action":"execute_flow","reasoning":"Specific intent flow exists for manager request","flow_to_execute":"speak_with_manager","confidence":0.98}

C) Conversational (flow answer - valid)
Current Flow: "sales" asking name
User: "My name is Rebecca"
→ {"action":"handle_conversationally","reasoning":"User answered current flow question","confidence":0.95}

C2) Conversational (flow answer - inappropriate response)
Current Flow: "menu" asking for order
User: "I want to book a hotel"
→ {"action":"handle_conversationally","reasoning":"User response is not appropriate for menu ordering context","response":"I'm here to help with your food order. What would you like from our menu?","confidence":0.95}

C3) Conversational (flow answer - appropriate response)
Current Flow: "menu" asking for order
User: "I'd like pasta"
→ {"action":"handle_conversationally","reasoning":"User provided appropriate response for menu ordering","confidence":0.95}

D) Refusal (in flow)
User: "I'd rather not share my name"
→ {"action":"handle_refusal","reasoning":"Refusal detected for 'name'","skip_fields":["name"],"confidence":0.97}

E) Uncertainty
User: "I'm not sure how many campaigns"
→ {"action":"handle_uncertainty","reasoning":"User unsure","confidence":0.9}

F) Speak to a Person (only when no specific flow exists)
User: "Can I talk to a human?" (no specific intent flow for this)
→ {"action":"speak_with_person","reasoning":"User requested human handoff with no specific flow","confidence":0.99}

G) Conversational (small talk/greeting)
User: "Hi there"
→ {"action":"handle_conversationally","reasoning":"Greeting small talk","confidence":0.9}

EDGE CASES:
- If both FAQ and flow seem plausible, prefer use_faq only if it is a direct factual question.
- If user mid-flow asks a different intent, you MAY switch flows via execute_flow (with reasoning).
- If user requests to speak with a specific person/role (manager, supervisor, etc.) and a specific intent flow exists, prefer execute_flow over speak_with_person."""

        # Pass the entire context as JSON (so the model sees structured data)
        user = f"""CONTEXT:
{json.dumps(context, indent=2)}

CRITICAL VALIDATION INSTRUCTIONS:
- If current_question exists, validate user response contextually and dynamically
- Consider the flow context and provide appropriate responses
- If user response is inappropriate for the current flow context, provide helpful correction
- Use conversational LLM intelligence to understand user intent and provide natural responses

Return ONLY the decision JSON (no markdown)."""

        resp = client.chat.completions.create(
            model="gpt-4o",
            messages=[{"role": "system", "content": system},
                      {"role": "user", "content": user}],
            temperature=0.3,
            max_tokens=700
        )

        text = resp.choices[0].message.content.strip()
        # Strip markdown fences if present
        if text.startswith("```"):
            text = text.strip("`").replace("json", "").strip()

        data = json.loads(text)

        return OrchestratorDecision(
            action=OrchestratorAction(data.get("action", "handle_conversationally")),
            reasoning=data.get("reasoning", ""),
            response=data.get("response"),
            flow_to_execute=data.get("flow_to_execute"),
            skip_fields=data.get("skip_fields", []),
            profile_updates=data.get("profile_updates", {}),
            next_objective=data.get("next_objective"),
            confidence=data.get("confidence", 0.8),
            metadata=data.get("metadata", {})
        )
    except Exception as e:
        logger.error(f"Decision error: {e}")
        # Safe fallback
        from backend.conversational_orchestrator import OrchestratorDecision, OrchestratorAction
        return OrchestratorDecision(
            action=OrchestratorAction.HANDLE_CONVERSATIONALLY,
            reasoning="Fallback due to error",
            response="Sorry, could you rephrase that?",
            confidence=0.3
        )

