# 🎙️ Alive5 Voice Agent

AI-powered voice agent with intelligent conversation flows, intent detection, and seamless FAQ integration.

## 🚀 Quick Start

### Prerequisites
- Python 3.12+
- LiveKit, OpenAI, and Alive5 API credentials
- SSH access to deployment server

### Local Development
```bash
git clone <repository-url>
cd voice-agent-livekit
python -m venv venv
venv\Scripts\activate  # Windows
pip install -r requirements.txt
# Create .env file with your credentials
```

### Run Services Locally
```bash
# Backend API (runs on port 8000)
uvicorn alive5-backend.main:app --host=0.0.0.0 --port=8000

# Worker (in separate terminal)
python alive5-backend/alive5-worker/worker.py dev
```

## 🌐 Production Deployment

### Server: 18.210.238.67 (Ubuntu 24)

#### Smart Deployment (PowerShell)
```powershell
# Deploy with interactive menu
.\deploy.ps1

# Check service status
.\check-services.ps1

# View logs
.\logs-backend.ps1    # Backend logs
.\logs-worker.ps1     # Worker logs

# Restart services
.\restart-services.ps1
```

### Service URLs
- **Frontend**: https://voice-agent-livekit.vercel.app
- **Backend API**: https://18.210.238.67.nip.io
- **Health Check**: https://18.210.238.67.nip.io/health
- **Available Voices**: https://18.210.238.67.nip.io/api/available_voices

## 🔧 Management

### Service Control
```bash
# Check status
ssh -i alive5-voice-ai-agent.pem ubuntu@18.210.238.67 'sudo systemctl status alive5-backend alive5-worker'

# Restart services
ssh -i alive5-voice-ai-agent.pem ubuntu@18.210.238.67 'sudo systemctl restart alive5-backend alive5-worker'
```

### Log Monitoring
```powershell
# Real-time logs (PowerShell)
.\logs-backend.ps1    # Backend service logs
.\logs-worker.ps1     # Worker service logs
```

## 📁 Project Structure
```
voice-agent-livekit/
├── deploy.ps1              # 🚀 Interactive deployment script
├── check-services.ps1      # 📊 Service status checker
├── logs-backend.ps1        # 📊 Backend logs
├── logs-worker.ps1         # 📊 Worker logs
├── restart-services.ps1    # 🔄 Service restart script
├── alive5-backend/         # Backend API application
│   ├── main.py            # FastAPI backend (simplified)
│   ├── cached_voices.json # Voice configuration
│   └── alive5-worker/     # LiveKit worker
│       ├── worker.py      # Main worker entry point
│       ├── functions.py   # Business logic (flows, FAQ)
│       └── system_prompt.py # LLM system prompt
├── alive5-frontend/        # Frontend UI
│   ├── index.html         # Main HTML
│   ├── style.css          # Styling
│   └── main_dynamic.js    # JavaScript client
├── .env                   # Environment variables
├── .gitignore            # Git ignore rules
├── README.md             # This file
└── requirements.txt      # Python dependencies
```

## 🔑 Environment Variables
```bash
OPENAI_API_KEY=your-openai-key
LIVEKIT_URL=wss://your-livekit-url
LIVEKIT_API_KEY=your-livekit-key
LIVEKIT_API_SECRET=your-livekit-secret
A5_API_KEY=your-alive5-key
A5_BASE_URL=https://api-v2-stage.alive5.com
BACKEND_URL=https://18.210.238.67.nip.io
DEEPGRAM_API_KEY=your-deepgram-key
CARTESIA_API_KEY=your-cartesia-key
```

## 🎯 Key Features

- **Simplified Architecture**: Clean, maintainable codebase
- **Smart Deployment**: Interactive deployment with file synchronization
- **HTTPS Support**: Secure communication with SSL certificates
- **Service Management**: Systemd services with auto-restart
- **Real-time Logs**: Easy log monitoring with PowerShell scripts
- **Health Monitoring**: Built-in health checks and status endpoints
- **🧠 Intelligent Agent**: GPT-4 powered conversation with function calling
- **📊 Voice Management**: 291+ available voices with dynamic switching
- **🔄 FAQ Integration**: Seamless FAQ bot integration with verbose mode
- **🎤 Voice Activity Detection**: Smart VAD with noise cancellation

## 🎤 Voice Features

### Available Voices
The system includes 291+ voices from Cartesia, accessible via:
- **API Endpoint**: `/api/available_voices`
- **Dynamic Switching**: Change voices during conversation
- **Voice Categories**: Different voice types and languages

### Voice Configuration
```bash
# Default voice (can be changed via API)
DEFAULT_VOICE_ID=f114a467-c40a-4db8-964d-aaba89cd08fa  # Miles - Yogi
```

## 🔧 API Endpoints

### Core Endpoints
- `GET /health` - Health check
- `GET /api/available_voices` - List all available voices
- `GET /api/connection_details` - LiveKit connection details
- `POST /api/change_voice` - Change agent voice
- `GET /api/sessions/{room_name}` - Get session data
- `POST /api/sessions/update` - Update session data


## 📞 Troubleshooting

### Common Issues
1. **Service Won't Start**: Check logs with `.\logs-backend.ps1`
2. **Missing Files**: Re-run `.\deploy.ps1` to sync files
3. **Voice Issues**: Check voice ID in logs and API responses
4. **Connection Issues**: Verify LiveKit credentials in `.env`

### Log Commands
```powershell
# View service logs
.\logs-backend.ps1
.\logs-worker.ps1

# Check service status
.\check-services.ps1

# Restart services
.\restart-services.ps1
```

### Debug Commands
```bash
# Check health
curl https://18.210.238.67.nip.io/health

# List available voices
curl https://18.210.238.67.nip.io/api/available_voices

# Get connection details
curl https://18.210.238.67.nip.io/api/connection_details
```

## 🚀 Deployment Options

The `deploy.ps1` script offers interactive deployment:

1. **Worker Only**: Deploy just the worker files
2. **Full Backend**: Deploy backend + worker
3. **Environment**: Automatically syncs `.env` file

## 📋 Development Workflow

1. **Local Development**: Test changes locally with `uvicorn` and `worker.py dev`
2. **Deploy**: Use `.\deploy.ps1` to deploy to server
3. **Monitor**: Use `.\check-services.ps1` and `.\logs-*.ps1` to monitor
4. **Debug**: Check logs and API endpoints for issues

## 🔒 Security

- **Environment Variables**: All sensitive data in `.env` file
- **HTTPS**: SSL certificates for secure communication
- **SSH Keys**: Secure server access with key-based authentication
- **Service Isolation**: Separate systemd services for backend and worker