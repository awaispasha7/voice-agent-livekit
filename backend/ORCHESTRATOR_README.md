# 🧠 Conversational Orchestrator

## **The Intelligent Brain of Your Voice Agent**

The Conversational Orchestrator is an intelligent layer designed to transform your voice agent from a rigid flow-based system into a natural, context-aware conversational AI.

> **⚠️ CURRENT STATUS:** The orchestrator is **partially integrated**. See `ORCHESTRATOR_STATUS.md` for accurate current status.

---

## 🎯 **What Problem Does It Solve?**

### **Before Orchestrator:**
- ❌ Rigid flow following - can't adapt to user needs
- ❌ No memory - re-asks for information
- ❌ Can't handle refusals gracefully
- ❌ FAQ bot and flows operate in silos
- ❌ Poor context awareness
- ❌ Robotic, script-like conversations

### **After Orchestrator (When Fully Integrated):**
- ✅ Intelligent routing between FAQ, flows, and conversation
- ✅ Remembers everything about the user
- ✅ Gracefully handles refusals and uncertainty
- ✅ Seamlessly integrates FAQ bot with flows
- ✅ Rich context awareness across entire conversation
- ✅ Natural, human-like interactions

> **Note:** These features require full integration into the main conversation flow.

---

## 📊 **Current Status**

**See `ORCHESTRATOR_STATUS.md` for detailed current status.**

### **What Works Now:**
- ✅ Orchestrator initializes successfully
- ✅ Intent detection after greeting (N/A step fix)
- ✅ Basic FAQ and flow functionality

### **What Doesn't Work Yet:**
- ❌ Intelligent conversation routing
- ❌ User profile-based decisions
- ❌ Refusal handling
- ❌ Conversation memory

### **Current Behavior:**
- After greeting: Users can say "I want sales info" → Routes to sales flow ✅
- General questions: "What does Alive5 do?" → FAQ (but not intelligently routed) ⚠️
- Refusals: "I don't want to share that" → Still asks again ❌

---

## 🏗️ **Architecture**

```
┌─────────────────────────────────────────────────────┐
│                                                     │
│        Conversational Orchestrator (GPT-4)         │
│                                                     │
│  - Analyzes user intent and context                │
│  - Makes intelligent routing decisions             │
│  - Maintains rich user profile                     │
│  - Handles refusals, uncertainty, navigation       │
│                                                     │
└─────────────────────────────────────────────────────┘
                         ↓
        ┌────────────────┼────────────────┐
        ↓                ↓                ↓
┌──────────────┐  ┌──────────────┐  ┌──────────────┐
│              │  │              │  │              │
│  FAQ Bot     │  │  Flow System │  │  General     │
│  (Bedrock)   │  │  (Structured)│  │  Conversation│
│              │  │              │  │              │
└──────────────┘  └──────────────┘  └──────────────┘
```

---

## 🎭 **Key Components**

### **1. UserProfile**
Tracks everything we know about each user:
- **Collected Information**: Name, email, campaign count, budget, etc.
- **Preferences**: Communication style, privacy concerns
- **Refused Fields**: What they don't want to share
- **Objectives**: What they're trying to accomplish
- **Conversation Summary**: Rolling context of the conversation

### **2. OrchestratorDecision**
The output of the intelligent decision-making process:
- **Action**: What to do (`use_faq`, `execute_flow`, `handle_conversationally`, etc.)
- **Reasoning**: Why this action was chosen
- **Response**: What to say to the user (if conversational)
- **Flow to Execute**: Which flow to start (if applicable)
- **Skip Fields**: What to skip based on user preferences
- **Profile Updates**: New information to store
- **Confidence**: How confident the system is in this decision

### **3. ConversationalOrchestrator**
The main brain that:
- Analyzes every user message in full context
- Makes routing decisions using LLM intelligence
- Maintains user profiles across sessions
- Provides smooth, natural conversation flow

---

## 🚀 **How It Works**

### **Decision-Making Process**

```python
User: "I'd rather not give my name"
                ↓
    Orchestrator analyzes:
    - Current context: "Agent just asked for name"
    - User profile: {}
    - Intent: "Refusal to provide information"
    - Available actions: FAQ, Flow, Conversational
                ↓
    Decision: handle_conversationally
    - Response: "That's perfectly fine! We can continue without it."
    - Skip fields: ["name"]
    - Profile update: {" prefers_privacy": true}
    - Next action: Continue flow, skip name collection
```

---

## 💡 **Intelligent Behaviors**

### **1. Memory & Context**
```
Agent: "How many campaigns are you running?"
User: "Around 26"
Agent: "How much budget per campaign?"
User: "Wait, remind me what I said earlier?"

Orchestrator:
- Checks conversation history
- Finds: "User said '26 campaigns'"
- Response: "You mentioned you're running around 26 campaigns. Would you like to continue with the budget question?"
```

### **2. Refusal Handling**
```
Agent: "May I have your phone number?"
User: "I don't feel comfortable sharing that."

Orchestrator:
- Detects refusal
- Marks "phone_number" as refused
- Response: "No problem at all! We can reach you via email instead."
- Adapts future flow to skip phone collection
```

### **3. Context Switching**
```
[In middle of marketing flow]
User: "Actually, what does Alive5 do?"

Orchestrator:
- Recognizes FAQ question
- Routes to Bedrock knowledge base
- Gets Alive5 information
- Saves current flow state
- After FAQ response: "Does that help? Now, back to your marketing campaigns..."
```

### **4. Uncertainty Handling**
```
Agent: "How many campaigns?"
User: "I'm not really sure, maybe around 20 something?"

Orchestrator:
- Detects uncertainty
- Extracts approximate answer ("20 something" → ~20-25)
- Continues flow gracefully
- OR offers to skip: "That's okay! We can move to the next question."
```

---

## 🎨 **Usage Examples**

### **Basic Integration**

```python
# In process_flow_message function

# Get orchestrator decision
if conversational_orchestrator:
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
    
    # Act based on decision
    if decision.action == OrchestratorAction.USE_FAQ:
        # Route to FAQ bot
        return await call_faq_bot(user_message)
    
    elif decision.action == OrchestratorAction.EXECUTE_FLOW:
        # Start a structured flow
        return await start_flow(decision.flow_to_execute)
    
    elif decision.action == OrchestratorAction.HANDLE_CONVERSATIONALLY:
        # Use orchestrator's response
        return {"response": decision.response, "type": "conversational"}
```

### **Checking User Preferences**

```python
# Before asking for a field
if orchestrator.should_skip_field(room_name, "phone_number"):
    # User refused this before, skip it
    continue_to_next_step()
else:
    # Ask for phone number
    ask_for_phone()
```

### **Retrieving Collected Information**

```python
# Get previously collected value
campaign_count = orchestrator.get_collected_value(room_name, "campaign_count")

if campaign_count:
    # We already have this, don't re-ask
    response = f"You mentioned you have {campaign_count} campaigns..."
else:
    # Need to collect
    response = "How many campaigns are you running?"
```

---

## 📊 **Conversation Scenarios**

### **Scenario 1: New User - Sales Intent**
```
User: "Hi, I want to learn about your pricing"

Orchestrator Decision:
→ Action: use_faq
→ Reasoning: "Clear request for pricing information from knowledge base"
→ Confidence: 0.95

[FAQ Bot provides pricing info]

User: "Great! I'd like to get started"

Orchestrator Decision:
→ Action: execute_flow
→ Flow: "sales"
→ Reasoning: "User ready to proceed with sales process"
→ Confidence: 0.90
```

### **Scenario 2: Returning User - Context Awareness**
```
[Previous session: User gave email but refused phone number]

User: "Hi, I'm back"

Orchestrator:
- Loads user profile
- Sees: email collected, phone refused, objective: "get_sales_info"
- Checks: Sales flow incomplete

Decision:
→ Action: handle_conversationally
→ Response: "Welcome back! We were just about to discuss your campaign needs. Ready to continue?"
→ Skip fields: ["email", "phone_number"] // Already have email, user refused phone
```

### **Scenario 3: Refusal with Adaptation**
```
Agent: "May I have your name?"
User: "I'd rather stay anonymous"

Orchestrator Decision:
→ Action: handle_conversationally
→ Response: "That's perfectly fine! I don't need your name to help you."
→ Profile updates: {"prefers_privacy": true}
→ Skip fields: ["name"]

Agent: "How many campaigns are you running?"
[Flow continues seamlessly without name]
```

---

## 🔧 **Configuration**

### **Environment Variables**
```bash
# OpenAI API for orchestrator decision-making
OPENAI_API_KEY=your-key-here

# Alive5 API for FAQ bot integration
ALIVE5_BASE_URL=https://api-v2-stage.alive5.com
FAQ_BOT_ID=your-faq-bot-id
```

### **Tuning Parameters**

The orchestrator uses GPT-4 with:
- **Temperature**: 0.3 (more consistent decisions)
- **Max Tokens**: 500 (detailed reasoning)
- **Context Window**: Last 10 messages

---

## 🧪 **Testing the Orchestrator**

### **Test 1: FAQ Routing**
```python
User: "What is Alive5?"
Expected: Route to FAQ bot
```

### **Test 2: Flow Execution**
```python
User: "I want marketing information"
Expected: Start marketing flow
```

### **Test 3: Refusal Handling**
```python
Agent: "What's your name?"
User: "I don't want to say"
Expected: Acknowledge refusal, continue without name
```

### **Test 4: Context Recall**
```python
Agent: "How many campaigns?"
User: "About 20"
Agent: "What's your budget?"
User: "Wait, how many campaigns did I say?"
Expected: "You mentioned about 20 campaigns."
```

---

## 📈 **Performance & Scalability**

### **LLM Call Optimization**
- **Single decision per message**: One LLM call makes all routing decisions
- **Cached profiles**: User state persisted across sessions
- **Smart context pruning**: Only last 10 messages sent to LLM

### **Fallback Mechanisms**
- If LLM fails: Safe conversational fallback
- If confidence low: Ask clarifying questions
- If context unclear: Continue current flow

---

## 🎓 **Best Practices**

### **1. Profile Management**
- ✅ Always update profiles after collecting information
- ✅ Check for existing values before re-asking
- ✅ Respect refused fields throughout conversation

### **2. Decision Trust**
- ✅ Trust high-confidence decisions (> 0.8)
- ⚠️  Validate medium-confidence decisions (0.5-0.8)
- ❌ Fallback on low-confidence decisions (< 0.5)

### **3. Context Preservation**
- ✅ Include conversation history in every decision
- ✅ Maintain flow state across interruptions
- ✅ Allow users to navigate back/forward

---

## 🚧 **Future Enhancements**

### **Phase 2** (Coming Soon)
- [ ] Multi-turn FAQ conversations
- [ ] Proactive suggestions based on profile
- [ ] Sentiment analysis for tone adjustment
- [ ] A/B testing different orchestration strategies

### **Phase 3** (Future)
- [ ] Voice tone adaptation
- [ ] Multilingual orchestration
- [ ] Integration with CRM systems
- [ ] Advanced analytics dashboard

---

## 🆘 **Troubleshooting**

### **Issue: Orchestrator not routing correctly**
**Solution**: Check logs for decision reasoning:
```python
logger.info(f"🧠 ORCHESTRATOR: Decision - {decision.action}")
logger.info(f"🧠 ORCHESTRATOR: Reasoning - {decision.reasoning}")
```

### **Issue: User profile not persisting**
**Solution**: Verify room_name consistency across calls

### **Issue: LLM timeout or errors**
**Solution**: Check OPENAI_API_KEY and network connectivity. System will fallback to safe conversational mode.

---

## 📚 **Related Documentation**

- [LLM Utils README](./llm_utils.py) - Centralized LLM functions
- [Main Dynamic README](./README.md) - Backend architecture
- [Flow System Guide](../README.md) - Flow-based conversation design

---

## 🎉 **Success Metrics**

Track these KPIs to measure orchestrator effectiveness:

- **User Satisfaction**: Fewer frustrations, more natural conversations
- **Completion Rate**: Higher flow completion rates
- **Re-ask Reduction**: Fewer repeated questions
- **Refusal Handling**: Graceful handling without conversation breaks
- **Context Accuracy**: Correct recall of previous information

---

**Built with ❤️ for intelligent, human-like conversations**

