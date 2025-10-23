# Simple Agent Deployment Guide

## 🚀 Quick Deployment

The simple-agent is your **robust and complete** voice agent that's ready for production deployment.

### Prerequisites
- SSH key: `alive5-voice-ai-agent.pem` in the project root
- Server access: `ubuntu@18.210.238.67`

### Deploy Simple Agent

```powershell
# Run the deployment script
.\deploy-simple-agent.ps1
```

This will:
1. ✅ Deploy the robust `backend/simple-agent/` 
2. ✅ Deploy the existing working `frontend/`
3. ✅ Update service configurations
4. ✅ Start the new services
5. ✅ Verify deployment

### What Gets Deployed

**Backend (`backend/simple-agent/`):**
- `main_simple.py` - Main agent with LiveKit Agent class
- `functions.py` - Alive5 API integration (flows + FAQ)
- `system_prompt.py` - LLM guidance and instructions
- `llm_utils.py` - LLM configuration and utilities

**Frontend (`frontend/`):**
- Existing working frontend (proven and stable)
- Voice selection functionality
- Clean UI/UX

**Configuration:**
- `requirements.txt` - All dependencies
- `.env` - Environment variables
- Service files for systemd

### Services Created

- `alive5-simple-backend` - FastAPI backend (port 8000)
- `alive5-simple-worker` - LiveKit worker

### Access Points

- **Main App**: http://18.210.238.67
- **Health Check**: http://18.210.238.67/health
- **API Docs**: http://18.210.238.67/docs

## 🎯 Simple Agent Features

Your deployed simple-agent includes:

### Core Functionality
- ✅ **Dynamic Flow Loading** - Loads bot flows from Alive5 API
- ✅ **FAQ Bot Integration** - Handles FAQ requests with periodic updates
- ✅ **Proactive Greeting** - Agent speaks first with greeting from flows
- ✅ **Consecutive Messages** - Handles multiple message nodes in sequence
- ✅ **Question Detection** - Only asks questions when it's a question node

### Technical Features
- ✅ **Robust Function Calling** - Uses LiveKit `@function_tool` decorators
- ✅ **Clean Logging** - No timestamps, essential info only
- ✅ **VAD & Noise Cancellation** - Silero VAD + LiveKit BVC
- ✅ **TTS with Fallback** - Cartesia primary, ElevenLabs fallback
- ✅ **STT** - Deepgram speech-to-text
- ✅ **Error Handling** - Comprehensive error handling and recovery

### User Experience
- ✅ **Voice Selection** - Choose from available voices
- ✅ **Real-time Updates** - Live transcription and responses
- ✅ **Connection Management** - Robust connection handling
- ✅ **UI Feedback** - Status updates and notifications

## 🔧 Post-Deployment

### Check Services
```bash
# SSH into server
ssh -i alive5-voice-ai-agent.pem ubuntu@18.210.238.67

# Check service status
sudo systemctl status alive5-simple-backend alive5-simple-worker

# View logs
sudo journalctl -u alive5-simple-backend -f
sudo journalctl -u alive5-simple-worker -f
```

### Test the Deployment
1. **Open**: http://18.210.238.67
2. **Select**: Bot name (e.g., "voice-1")
3. **Choose**: Voice from dropdown
4. **Click**: "Join Voice Chat"
5. **Verify**: Agent greets you and responds to voice

### Environment Variables
Ensure these are set in `.env`:
```env
# Alive5 API
ALIVE5_API_BASE_URL=https://api.alive5.com
ALIVE5_API_KEY=your_api_key

# LiveKit
LIVEKIT_URL=your_livekit_url
LIVEKIT_API_KEY=your_api_key
LIVEKIT_API_SECRET=your_api_secret

# TTS
CARTESIA_API_KEY=your_cartesia_key
ELEVENLABS_API_KEY=your_elevenlabs_key

# STT
DEEPGRAM_API_KEY=your_deepgram_key
```

## 🎉 Success!

Your robust simple-agent is now deployed and ready to handle voice conversations with:
- Dynamic flow loading
- FAQ bot integration  
- Clean, professional logging
- Robust error handling
- Excellent user experience

The simple-agent represents the culmination of all your development work - it's production-ready and feature-complete!
