# Voice Agent Workers

This directory contains different worker implementations for the Alive5 Voice Agent.

## Available Workers

### 🚀 `main_flow_based.py` (RECOMMENDED)
- **Purpose**: Flow-based conversation system using Alive5 templates
- **Features**: 
  - Dynamic template loading from Alive5 API
  - Structured conversation flows (pricing, support, billing, agent transfer)
  - Fallback to FAQ bot for general questions
  - Real-time template updates
- **Use Case**: Production use with client-defined conversation flows

### 🔄 `main_dynamic.py`
- **Purpose**: Intent-based conversation system using LLM classification
- **Features**:
  - OpenAI-based intent detection (sales, support, billing)
  - Dynamic conversation adaptation
  - User data extraction
  - Session tracking and analytics
- **Use Case**: Open-ended conversations with AI-generated responses

## Usage

To run a specific worker:

```bash
# Flow-based worker (recommended)
python main_flow_based.py

# Dynamic intent-based worker
python main_dynamic.py
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
