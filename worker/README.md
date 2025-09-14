# Voice Agent Workers

This directory contains different worker implementations for the Alive5 Voice Agent.

## Available Workers

### 🚀 `main_flow_based.py`
- **Purpose**: Flow-based conversation system using Alive5 templates
- **Features**: 
  - Dynamic template loading from Alive5 API
  - Structured conversation flows (pricing, support, billing, agent transfer)
  - Fallback to FAQ bot for general questions
  - Real-time template updates
- **Use Case**: Production use with client-defined conversation flows

 

## Usage

To run a specific worker:

```bash
# Flow-based worker
python main_flow_based.py

 
```

## Environment Variables

All workers require these environment variables:
- `OPENAI_API_KEY`
- `DEEPGRAM_API_KEY` 
- `CARTESIA_API_KEY`
- `LIVEKIT_URL`
- `LIVEKIT_API_KEY`
- `LIVEKIT_API_SECRET`

Additional requirements:
- **Flow-based**: `A5_BASE_URL`, `A5_API_KEY`
