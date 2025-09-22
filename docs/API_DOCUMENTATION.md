# 🛠️ Alive5 Voice Agent - API Documentation

## Overview

The Alive5 Voice Agent API provides intelligent conversation flows, intent detection, and seamless human agent transfers. This document covers all available endpoints, request/response formats, and integration examples.

## Base URL

- **Development**: `http://localhost:8000`
- **Production**: `https://your-backend-url.onrender.com`

## Authentication

All API endpoints use JWT tokens for authentication. Tokens are automatically generated and managed by the system.

## Core Endpoints

### 1. Health Check

**GET** `/health`

Check the health status of the API.

#### Response
```json
{
  "status": "ok",
  "active_sessions": 5,
  "timestamp": 1640995200.123
}
```

---

### 2. Connection Details

**POST** `/api/connection_details`

Get LiveKit connection details for establishing a voice session.

#### Request Body
```json
{
  "participant_name": "user123",
  "room_name": "session_123"
}
```

#### Response
```json
{
  "serverUrl": "wss://alive5-x7iklos9.livekit.cloud",
  "roomName": "session_123",
  "participantToken": "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9...",
  "participantName": "user123",
  "sessionId": "session_123",
  "expiresAt": 1640998800
}
```

#### Error Responses
- `400 Bad Request`: Missing required fields
- `500 Internal Server Error`: Server configuration error

---

### 3. Process Flow Message

**POST** `/api/process_flow_message`

Process user messages through the intelligent flow system.

#### Request Body
```json
{
  "room_name": "session_123",
  "user_message": "I need pricing information",
  "conversation_history": [
    {
      "role": "user",
      "content": "Hello",
      "timestamp": "2025-01-18T10:00:00Z"
    },
    {
      "role": "assistant", 
      "content": "Hi there! How can I help you today?",
      "timestamp": "2025-01-18T10:00:01Z"
    }
  ]
}
```

#### Response Types

##### Flow Started
```json
{
  "status": "processed",
  "room_name": "session_123",
  "user_message": "I need pricing information",
  "flow_result": {
    "type": "flow_started",
    "flow_name": "Pricing",
    "response": "How many phone lines do you need?",
    "next_step": {
      "type": "question",
      "text": "How many texts do you send a month?",
      "name": "question_ae051e94-daf6-4d3d-b4fa-296647392d64"
    }
  }
}
```

##### Transfer Initiated
```json
{
  "status": "processed",
  "room_name": "session_123", 
  "user_message": "Can I speak with someone over the phone?",
  "flow_result": {
    "type": "transfer_initiated",
    "response": "Connecting you to a human agent. Please wait.",
    "flow_state": {
      "current_flow": null,
      "current_step": null,
      "conversation_history": [...]
    }
  }
}
```

##### FAQ Response
```json
{
  "status": "processed",
  "room_name": "session_123",
  "user_message": "What are your business hours?",
  "flow_result": {
    "type": "faq_response",
    "response": "Our business hours are Monday-Friday 9AM-6PM EST.",
    "urls": ["https://alive5.com/contact"],
    "bot_id": "faq_b9952a56-fc7b-41c9-b0a0-5c662ddb039e"
  }
}
```

##### Error Response
```json
{
  "status": "processed",
  "room_name": "session_123",
  "user_message": "Invalid input",
  "flow_result": {
    "type": "error",
    "response": "I'm sorry, I didn't understand that. Could you please rephrase?"
  }
}
```

---

### 4. Update Session

**POST** `/api/sessions/update`

Update session information and state.

#### Request Body
```json
{
  "room_name": "session_123",
  "intent": "pricing",
  "status": "active",
  "flow_state": {
    "current_flow": "pricing_flow",
    "current_step": "phone_lines_question",
    "user_responses": {
      "phone_lines": "5",
      "texts_per_month": "2000"
    }
  }
}
```

#### Response
```json
{
  "message": "Session updated successfully",
  "session_id": "session_123",
  "current_intent": "pricing",
  "status": "active"
}
```

---

### 5. Get Session Info

**GET** `/api/sessions/{room_name}`

Get current session information.

#### Response
```json
{
  "session_id": "session_123",
  "participant_name": "user123",
  "intent": "pricing",
  "status": "active",
  "duration_seconds": 120,
  "flow_state": {
    "current_flow": "pricing_flow",
    "current_step": "phone_lines_question",
    "user_responses": {
      "phone_lines": "5"
    }
  }
}
```

---

### 6. List All Sessions

**GET** `/api/sessions`

Get information about all active sessions.

#### Response
```json
{
  "total_sessions": 3,
  "sessions": [
    {
      "session_id": "session_123",
      "participant_name": "user123",
      "intent": "pricing",
      "status": "active",
      "duration_seconds": 120
    },
    {
      "session_id": "session_456", 
      "participant_name": "user456",
      "intent": "agent_transfer",
      "status": "transferring",
      "duration_seconds": 45
    }
  ]
}
```

---

### 7. Cleanup Room

**DELETE** `/api/rooms/{room_name}`

Clean up room resources and session data.

#### Response
```json
{
  "message": "Room session_123 cleaned up successfully",
  "session_summary": {
    "duration_seconds": 300,
    "intent": "pricing",
    "final_status": "completed"
  }
}
```

---

### 8. Initiate Transfer

**POST** `/api/sessions/{room_name}/transfer`

Initiate transfer to human agent.

#### Request Body
```json
{
  "department": "sales"
}
```

#### Response
```json
{
  "message": "Transfer to sales initiated",
  "session_id": "session_123",
  "transfer_status": "initiated",
  "estimated_wait_time": "2-3 minutes"
}
```

---

## Flow Types

### Supported Flow Result Types

| Type | Description | Use Case |
|------|-------------|----------|
| `flow_started` | New flow initiated | Starting a new conversation flow |
| `flow_response` | Flow step response | Continuing an existing flow |
| `question` | Asking user a question | Gathering information |
| `message` | General message | Providing information |
| `faq_response` | FAQ bot response | Answering common questions |
| `transfer_initiated` | Agent transfer started | Escalating to human agent |
| `conversation_end` | Conversation completed | Ending the session |
| `error` | Error occurred | Handling errors gracefully |

### Intent Types

| Intent | Description | Trigger Examples |
|--------|-------------|------------------|
| `greeting` | Welcome message | "Hello", "Hi", "Good morning" |
| `pricing` | Pricing information | "How much", "Pricing", "Cost" |
| `weather` | Weather information | "Weather", "Forecast", "Temperature" |
| `agent` | Human agent request | "Speak with someone", "Talk to agent" |

## Error Handling

### HTTP Status Codes

- `200 OK`: Request successful
- `400 Bad Request`: Invalid request data
- `404 Not Found`: Resource not found
- `500 Internal Server Error`: Server error

### Error Response Format

```json
{
  "error": "Error message",
  "details": "Additional error details",
  "timestamp": "2025-01-18T10:00:00Z"
}
```

## Rate Limiting

- **Default**: 100 requests per minute per IP
- **Headers**: 
  - `X-RateLimit-Limit`: Request limit
  - `X-RateLimit-Remaining`: Remaining requests
  - `X-RateLimit-Reset`: Reset timestamp

## Webhooks

### Session Events

The system can send webhooks for important session events:

#### Transfer Initiated
```json
{
  "event": "transfer_initiated",
  "session_id": "session_123",
  "timestamp": "2025-01-18T10:00:00Z",
  "data": {
    "department": "sales",
    "estimated_wait_time": "2-3 minutes"
  }
}
```

#### Session Completed
```json
{
  "event": "session_completed",
  "session_id": "session_123",
  "timestamp": "2025-01-18T10:05:00Z",
  "data": {
    "duration_seconds": 300,
    "intent": "pricing",
    "final_status": "completed"
  }
}
```

## Integration Examples

### JavaScript/Node.js

```javascript
// Get connection details
const connectionResponse = await fetch('/api/connection_details', {
  method: 'POST',
  headers: {
    'Content-Type': 'application/json',
  },
  body: JSON.stringify({
    participant_name: 'user123',
    room_name: 'session_123'
  })
});

const connectionData = await connectionResponse.json();

// Process message
const messageResponse = await fetch('/api/process_flow_message', {
  method: 'POST',
  headers: {
    'Content-Type': 'application/json',
  },
  body: JSON.stringify({
    room_name: 'session_123',
    user_message: 'I need pricing information',
    conversation_history: []
  })
});

const messageData = await messageResponse.json();
console.log('Flow result:', messageData.flow_result);
```

### Python

```python
import requests

# Get connection details
connection_response = requests.post('/api/connection_details', json={
    'participant_name': 'user123',
    'room_name': 'session_123'
})

connection_data = connection_response.json()

# Process message
message_response = requests.post('/api/process_flow_message', json={
    'room_name': 'session_123',
    'user_message': 'I need pricing information',
    'conversation_history': []
})

message_data = message_response.json()
print('Flow result:', message_data['flow_result'])
```

### cURL

```bash
# Get connection details
curl -X POST http://localhost:8000/api/connection_details \
  -H "Content-Type: application/json" \
  -d '{
    "participant_name": "user123",
    "room_name": "session_123"
  }'

# Process message
curl -X POST http://localhost:8000/api/process_flow_message \
  -H "Content-Type: application/json" \
  -d '{
    "room_name": "session_123",
    "user_message": "I need pricing information",
    "conversation_history": []
  }'
```

## Testing

### Test Endpoints

#### Test Intent Detection
**POST** `/api/test_intent_detection`

```json
{
  "user_message": "I need pricing information",
  "conversation_history": []
}
```

#### Get Template Info
**GET** `/api/template_info`

Returns current bot template information and available intents.

#### Get Flow State
**GET** `/api/flow_state/{room_name}`

Returns current flow state for a specific room.

## Monitoring

### Health Check
- **Endpoint**: `/health`
- **Frequency**: Every 30 seconds
- **Alerts**: Email/SMS on failure

### Metrics
- Response times
- Error rates
- Session durations
- Intent detection accuracy
- Transfer success rates

## Security

### Authentication
- JWT tokens with expiration
- Secure token generation
- Automatic token refresh

### Data Protection
- All data encrypted in transit (HTTPS)
- Sensitive data encrypted at rest
- GDPR compliant data handling

### Rate Limiting
- IP-based rate limiting
- Per-user rate limiting
- DDoS protection

---

**For additional support or questions, please contact the development team.**
