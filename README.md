# 🎙️ Alive5 Voice Agent - LiveKit Integration

A sophisticated AI-powered voice agent system that provides intelligent conversation flows, intent detection, and seamless human agent transfers. Built with LiveKit for real-time voice communication and powered by OpenAI GPT for intelligent responses.

## 🌟 Key Features

- **🎯 Intelligent Intent Detection**: AI-powered recognition of user intents (Pricing, Weather, Agent requests)
- **🔄 Dynamic Flow Management**: Context-aware conversation flows with state persistence
- **👥 Seamless Agent Transfer**: Automatic escalation to human agents when needed
- **🎤 Real-time Voice Processing**: LiveKit integration for high-quality voice communication
- **📊 Session Management**: Complete conversation tracking and state management
- **🔧 Environment-based Configuration**: Flexible deployment across different environments

## 🏗️ Architecture Overview

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   Frontend      │    │   Backend API   │    │   Worker        │
│   (Vercel)      │◄──►│   (Server)      │◄──►│   (Server)      │
│                 │    │                 │    │                 │
│ • Voice UI      │    │ • Flow Logic    │    │ • LiveKit       │
│ • Real-time     │    │ • Intent        │    │ • Voice         │
│   Communication │    │   Detection     │    │   Processing    │
│ • Session Mgmt  │    │ • API Endpoints │    │ • Agent Logic   │
└─────────────────┘    └─────────────────┘    └─────────────────┘
```

## 🚀 Quick Start

### Prerequisites
- Python 3.11+
- Node.js (for frontend)
- LiveKit account
- OpenAI API key
- Alive5 API credentials

### Local Development

1. **Clone and Setup**
   ```bash
   git clone <repository-url>
   cd voice-agent-livekit-affan
   pip install -r requirements.txt
   ```

2. **Environment Configuration**
   ```bash
   cp .env.example .env
   # Edit .env with your credentials
   ```

3. **Start Backend**
   ```bash
   uvicorn backend.main_dynamic:app --host=0.0.0.0 --port=8000
   ```

4. **Start Worker**
   ```bash
   python backend/worker/main_flow_based.py
   ```

5. **Open Frontend**
   ```bash
   # Open frontend/index.html in browser
   # Or serve with a local server
   ```

## 🎯 Intent Detection System

### Supported Intents

| Intent | Trigger Phrases | Response |
|--------|----------------|----------|
| **Agent Transfer** | "speak with someone", "talk to an agent", "over the phone" | "Connecting you to a human agent. Please wait." |
| **Pricing** | "pricing", "cost", "how much", "plans" | Dynamic pricing flow with questions |
| **Weather** | "weather", "forecast", "temperature" | Weather information flow |
| **Greeting** | "hello", "hi", "hey", "good morning" | Friendly welcome message |

### Intent Detection Flow

```
User Input → Escalation Check → Agent Transfer (if detected)
     ↓ (if no escalation)
LLM Intent Detection → Flow Processing → Response
```

## 🔄 Conversation Flow Examples

### 1. Agent Transfer Flow
```
User: "Can I speak with someone over the phone?"
System: "Connecting you to a human agent. Please wait."
Result: Transfer initiated
```

### 2. Pricing Flow
```
User: "I need pricing information"
System: "How many phone lines do you need?"
User: "5 lines"
System: "How many texts do you send a month?"
User: "2000 texts"
System: "Do you have any special needs like SSO or CRM integration?"
User: "Yes, we need Salesforce integration"
System: "Thanks for providing all your details. Please hold while we generate the best plan for you..."
```

### 3. Weather Flow
```
User: "What's the weather like?"
System: "What is your zip code?"
User: "90210"
System: [Weather information provided]
```

## 🛠️ API Endpoints

### Core Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/api/connection_details` | POST | Get LiveKit connection details |
| `/api/process_flow_message` | POST | Process user message through flow system |
| `/api/sessions/update` | POST | Update session information |
| `/api/rooms/{room_name}` | DELETE | Clean up room resources |
| `/health` | GET | Health check endpoint |

### Request/Response Examples

#### Connection Details
```json
POST /api/connection_details
{
  "participant_name": "user123",
  "room_name": "session_123"
}

Response:
{
  "serverUrl": "wss://alive5-x7iklos9.livekit.cloud",
  "roomName": "session_123",
  "participantToken": "eyJhbGciOiJIUzI1NiIs...",
  "participantName": "user123",
  "sessionId": "session_123"
}
```

#### Flow Message Processing
```json
POST /api/process_flow_message
{
  "room_name": "session_123",
  "user_message": "I need pricing information",
  "conversation_history": [...]
}

Response:
{
  "status": "processed",
  "room_name": "session_123",
  "user_message": "I need pricing information",
  "flow_result": {
    "type": "flow_started",
    "flow_name": "Pricing",
    "response": "How many phone lines do you need?",
    "next_step": {...}
  }
}
```

## 🔧 Configuration

### Environment Variables

```bash
# LiveKit Configuration
LIVEKIT_URL=wss://your-livekit-url
LIVEKIT_API_KEY=your-api-key
LIVEKIT_API_SECRET=your-api-secret

# OpenAI Configuration
OPENAI_API_KEY=your-openai-key

# Alive5 Configuration
A5_BASE_URL=https://api-v2-stage.alive5.com
A5_API_KEY=your-a5-api-key
A5_FAQ_BOT_ID=your-faq-bot-id

# Application URLs
FRONTEND_URL=https://your-frontend-url
BACKEND_URL=http://localhost:8000
```

### Frontend Configuration

The frontend automatically detects the environment and configures API endpoints:

```javascript
// Automatic configuration
const CONFIG = {
    API_BASE_URL: window.API_BASE_URL || 'http://localhost:8000',
    FRONTEND_URL: window.FRONTEND_URL || 'http://localhost:3000',
    // ... other settings
};
```

## 🚀 Deployment

### Production Deployment

1. **Frontend (Vercel)**
   - Deploy `frontend/` directory to Vercel
   - Set environment variables in Vercel dashboard
   - Automatic HTTPS and CDN

2. **Backend & Worker (Client Server)**
   - Deploy both backend and worker on client's server
   - Configure environment variables on server
   - Set up process management (systemd, PM2, or Docker)
   - Configure reverse proxy (nginx) for backend API

### Deployment Commands

```bash
# Backend API
uvicorn backend.main_dynamic:app --host=0.0.0.0 --port=8000

# Worker
python backend/worker/main_flow_based.py
```

### Cost Breakdown

| Service | Plan | Monthly Cost |
|---------|------|--------------|
| Frontend (Vercel) | Free | $0 |
| Backend & Worker (Client Server) | Client Provided | $0 |
| **Total** | | **$0/month** |

## 📊 Monitoring & Logging

### Log Levels
- **INFO**: Normal operation logs
- **WARNING**: Non-critical issues
- **ERROR**: Critical errors requiring attention
- **DEBUG**: Detailed debugging information

### Key Metrics
- Session duration
- Intent detection accuracy
- Response times
- Error rates
- Transfer success rates

## 🔍 Troubleshooting

### Common Issues

1. **Intent Detection Not Working**
   - Check OpenAI API key
   - Verify bot template is loaded
   - Review LLM response logs

2. **Voice Connection Issues**
   - Verify LiveKit credentials
   - Check network connectivity
   - Review browser permissions

3. **Flow State Issues**
   - Check flow_states directory permissions
   - Verify session persistence
   - Review conversation history

### Debug Mode

Enable detailed logging by setting:
```bash
export LOG_LEVEL=DEBUG
```

## 🤝 Support

For technical support or questions:
- Check the troubleshooting section
- Review logs for error details
- Contact the development team

## 📄 License

This project is proprietary software. All rights reserved.

---

**Built with ❤️ for Alive5**