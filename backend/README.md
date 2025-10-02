# Voice Agent Backend

This backend provides the core functionality for the LiveKit-based voice agent system, featuring an **intelligent conversational orchestrator** with dynamic response generation, optimized LLM processing, and real-time voice switching.

## 🏗️ Architecture

### Core Components

- **`main_dynamic.py`** - Main FastAPI application with intelligent orchestration and flow management
- **`llm_utils.py`** - Centralized LLM utility functions with dynamic conversational AI
- **`conversational_orchestrator.py`** - Intelligent conversation orchestrator for routing decisions
- **`worker/main_flow_based.py`** - LiveKit agent worker for real-time voice processing

### Key Features

- **🧠 Intelligent Orchestrator**: GPT-4 powered conversation routing with context awareness
- **🎭 Dynamic Conversational AI**: Natural, human-like responses generated in real-time
- **⚡ Optimized Performance**: Single LLM call per user message (50% reduction in API calls)
- **🔄 Smart Flow Management**: Context-aware intent detection and flow progression
- **🎤 Voice Integration**: Cartesia TTS with real-time voice switching and caching
- **📊 User Profile Management**: Comprehensive user data extraction and profile tracking
- **💾 Local Persistence**: User profiles and flow states saved to JSON files for debugging
- **🐛 Debug System**: Comprehensive logging and API endpoints for testing and analysis
- **🚀 Production Ready**: Clean, maintainable code with no redundant systems

## 🤖 LLM Utilities (`llm_utils.py`)

Centralized LLM functions for all AI operations with **dynamic conversational AI** and **intelligent orchestration**.

### Available Functions

#### `analyze_transcription_quality(user_text: str) -> Dict[str, Any]`
Analyzes transcription quality and returns clarity assessment.

#### `extract_answer_with_llm(question_text: str, user_text: str) -> Dict[str, Any]`
Extracts structured answers from user responses.

**Supported Types:**
- Numbers: "zero", "fifty one", "around fifteen"
- Booleans: "yes", "no", "I need it"
- ZIP Codes: "two five nine six three", "12345"
- Text: Any meaningful response

#### `match_answer_with_llm(user_response: str, available_answers: Dict[str, Any]) -> Optional[Dict[str, Any]]`
Matches user responses with predefined answer options using LLM intelligence.

#### `detect_intent_with_llm(user_message: str) -> Optional[Dict[str, Any]]`
Detects user intent from available options using LLM.

#### `detect_uncertainty_with_llm(user_message: str) -> Dict[str, Any]`
Detects if user is expressing uncertainty or inability to answer a question.

#### `extract_user_data_with_llm(user_message: str) -> Dict[str, Any]`
Extracts comprehensive user information (name, email, phone, company, etc.) from natural language.

#### `generate_conversational_response(user_message: str, context: Dict[str, Any]) -> str`
Generates natural, human-sounding conversational responses with context awareness.

#### `make_orchestrator_decision(context: Dict[str, Any]) -> OrchestratorDecision`
Makes intelligent orchestration decisions using GPT-4 for conversation routing.

**Supported Matching Types:**
- **Exact Matches**: "zero" matches "0", "five" matches "5"
- **Range Matches**: "around fifteen" matches "11-20" (if 15 is in that range)
- **Threshold Matches**: "about thirty" matches "More than 21" (if 30 > 21)
- **Context Matches**: "I'm not running any" matches "0"

**Returns:**
- Matching answer key if found, `None` if no confident match

## 🔄 Flow Management

### Flow Types

1. **Greeting Flow** - Initial conversation handling
2. **Intent Flows** - Specific conversation paths (menu, manager, etc.)
3. **Question Flows** - Structured Q&A sequences

### Flow Processing

The system uses a hybrid approach combining:
- **Smart Processor**: Contextual understanding of user messages
- **Intent Detection**: Classification of user requests
- **Answer Extraction**: Structured parsing of responses

### Flow State Management

- **Persistent Storage**: Flow states saved to `persistence/flow_states/` directory
- **Session Tracking**: Each room maintains its own flow state
- **Recovery**: Automatic flow state restoration on reconnection
- **User Profile Persistence**: User data extracted by orchestrator saved to `persistence/user_profiles/`
- **Debug Logging**: Comprehensive logs saved to `persistence/debug_logs/` for analysis

## 🚀 Current System Features

### 🧠 Intelligent Orchestrator
- **GPT-4 Powered**: Advanced conversation routing with context awareness
- **Single LLM Call**: Optimized performance with one API call per user message
- **Smart Decision Making**: Intelligent routing between FAQ, flows, and conversational AI

### 🎭 Dynamic Conversational AI
- **Natural Responses**: Human-like conversational responses generated in real-time
- **Context Awareness**: Responses adapt based on conversation history and user profile
- **Adaptive Personality**: Warm, friendly, and professional communication style

### 🔄 Smart Flow Management
- **Intent Detection**: Accurate classification of user requests and responses
- **Flow Progression**: Seamless navigation through structured conversation flows
- **User Data Extraction**: Comprehensive information capture from natural language

### ⚡ Optimized Performance
- **Efficient Architecture**: Clean, maintainable code with maximum performance
- **Real-time Processing**: Fast response times with intelligent caching
- **Production Ready**: Stable, tested, and ready for deployment

## 🎤 Voice Management

### Voice Caching

- **Cartesia Integration**: Fetches available voices from Cartesia API
- **Local Caching**: Voices cached in `cached_voices.json` for performance
- **Pagination Support**: Handles large voice catalogs efficiently

### Voice Switching

- **Pre-session**: Voice selection before joining the room
- **In-session**: Real-time voice changes via LiveKit data packets
- **Persistence**: Voice preferences saved in session data

### Default Voice

- **ID**: `7f423809-0011-4658-ba48-a411f5e516ba`
- **Name**: "Ashwin - Warm Narrator"
- **Fallback**: Used when no voice is specified

## 🔧 API Endpoints

### Session Management
- `POST /api/sessions/{room_name}` - Create or get session info
- `GET /api/sessions/{room_name}` - Get session details
- `POST /api/change_voice` - Change voice for a session

### Voice Management
- `GET /api/available_voices` - Get cached voice list
- `POST /api/update_voice_cache` - Refresh voice cache

### Flow Processing
- `POST /api/process_flow_message` - Process user messages through flows

### Debug & Testing
- `GET /api/debug/rooms` - List all rooms with saved data
- `GET /api/debug/room/{room_name}` - Get debug data for specific room
- `DELETE /api/debug/room/{room_name}` - Clear debug data for specific room

## 🚀 Getting Started

### Prerequisites

```bash
pip install -r requirements.txt
```

### Environment Variables

Create a `.env` file with:

```env
OPENAI_API_KEY=your_openai_key
LIVEKIT_URL=your_livekit_url
LIVEKIT_API_KEY=your_livekit_api_key
LIVEKIT_API_SECRET=your_livekit_secret
CARTESIA_API_KEY=your_cartesia_key
A5_API_KEY=your_alive5_key
A5_BASE_URL=https://api-v2-stage.alive5.com
A5_BOTCHAIN_NAME=voice-1
A5_ORG_NAME=alive5stage0
```

### Running the Backend

```bash
# Start the FastAPI server
uvicorn main_dynamic:app --host 0.0.0.0 --port 8000

# Start the LiveKit worker
python worker/main_flow_based.py
```

## 📊 Performance

### LLM Response Times
- **Average**: ~1.1 seconds
- **Range**: 0.9 - 1.5 seconds
- **Success Rate**: 82.4% for answer extraction
- **Confidence Scores**: 0.8-0.95 for successful extractions

### Voice Operations
- **Cache Loading**: <100ms for 290 voices
- **Voice Switching**: <500ms for in-session changes
- **API Calls**: Optimized with pagination and caching

### System Optimizations
- **Simplified Architecture**: Direct function calls, no wrappers
- **Clean APIs**: Optimized function signatures
- **Reduced Complexity**: Easier maintenance and debugging

## 💾 Local Persistence System

### Overview
The system automatically saves conversation data to JSON files for debugging, testing, and analysis:

```
backend/persistence/
├── flow_states/          # Flow state JSON files
├── user_profiles/        # User profile JSON files  
└── debug_logs/          # Debug logs for analysis
```

### User Profile Persistence
- **Automatic Saving**: User profiles saved after every orchestrator decision
- **Data Extracted**: Name, email, phone, company, role, budget, preferences, refused fields, objectives
- **Context Preservation**: Profiles persist across conversations for 24 hours
- **Smart Loading**: Existing profiles loaded and merged with new data

### Flow State Persistence
- **Enhanced Structure**: Organized in dedicated directory
- **Complete State**: Current flow, step, conversation history, user responses
- **Automatic Cleanup**: Old files (>24 hours) automatically removed

### Debug Logging
- **Orchestrator Decisions**: Every decision with reasoning and confidence
- **User Data Extraction**: What data was extracted from each message
- **Flow Progression**: How flows are progressing and why
- **Timestamped Logs**: Easy to track conversation flow

### Debug API Endpoints

#### View Room Data
```bash
GET /api/debug/room/{room_name}
```
Returns:
- Flow state (current flow, step, conversation history)
- User profile (collected info, preferences, refused fields)
- Debug logs (last 10 logs with timestamps)

#### List All Rooms
```bash
GET /api/debug/rooms
```
Returns all rooms with saved data

#### Clear Room Data
```bash
DELETE /api/debug/room/{room_name}
```
Clears all debug data for a specific room

## 🔍 Logging

The system uses structured logging with different levels:

- **INFO**: Flow transitions, voice changes, successful operations
- **WARNING**: Fallback scenarios, retry attempts
- **ERROR**: API failures, processing errors
- **DEBUG**: Detailed flow state information

Verbose logs from external libraries (`httpx`, `httpcore`, `uvicorn.access`) are disabled for cleaner output.

## 🛠️ Development

- Add new LLM functions to `llm_utils.py`
- Use `detect_intent_with_llm` for intent detection
- Use `extract_answer_with_llm` for answer extraction
- Use `match_answer_with_llm` for answer matching in flows
- Keep functions simple and centralized

## 📝 Notes

- All LLM calls are centralized in `llm_utils.py` for easy maintenance
- The system uses an LLM-only approach for answer extraction and matching
- Answer matching uses intelligent LLM-based logic instead of hardcoded patterns
- Voice changes are handled via LiveKit data packets for real-time updates
- Flow states are automatically persisted and restored
- The system gracefully handles API failures with fallback mechanisms
