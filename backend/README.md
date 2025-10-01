# Voice Agent Backend

This backend provides the core functionality for the LiveKit-based voice agent system, featuring a streamlined architecture with centralized LLM processing, dynamic flow management, and real-time voice switching.

## 🏗️ Architecture

### Core Components

- **`main_dynamic.py`** - Main FastAPI application with flow management and voice handling
- **`llm_utils.py`** - Centralized LLM utility functions for all AI operations (simplified and optimized)
- **`worker/main_flow_based.py`** - LiveKit agent worker for real-time voice processing

### Key Features

- **Simplified LLM Processing**: Single, well-tuned LLM functions with no unnecessary wrappers
- **Flow Management**: Dynamic conversation flows with smart intent detection
- **Voice Integration**: Cartesia TTS with real-time voice switching and caching
- **Session Management**: Persistent session state with voice preferences
- **Clean Architecture**: Removed legacy code and unnecessary complexity

## 🤖 LLM Utilities (`llm_utils.py`)

Centralized LLM functions for all AI operations.

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

#### `analyze_message_with_smart_processor(user_message: str, conversation_history: List[Dict], current_flow: str = None, current_step: str = None) -> Dict[str, Any]`
Smart contextual analysis of user messages for intent detection.

#### `detect_intent_with_llm(user_message: str, available_intents: List[str]) -> Optional[str]`
Detects user intent from available options.

#### `match_answer_with_llm(question_text: str, user_response: str, available_answers: Dict[str, Any]) -> Optional[str]`
Matches user responses with predefined answer options using LLM intelligence.

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

- **Persistent Storage**: Flow states saved to `flow_states/` directory
- **Session Tracking**: Each room maintains its own flow state
- **Recovery**: Automatic flow state restoration on reconnection

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
