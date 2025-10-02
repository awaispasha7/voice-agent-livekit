# 🎙️ Alive5 Voice Agent

AI-powered voice agent with intelligent conversation flows, intent detection, and seamless agent transfers.

## 🚀 Quick Start

### Prerequisites
- Python 3.12+
- LiveKit, OpenAI, and Alive5 API credentials
- SSH access to deployment server

### Local Development
```bash
git clone <repository-url>
cd voice-agent-livekit-affan
pip install -r requirements.txt
# Create .env file with your credentials
```

### Run Services Locally
```bash
# Backend API (runs on port 8000, proxied by Nginx on port 80)
uvicorn backend.main_dynamic:app --host=0.0.0.0 --port=8000

# Worker
python backend/worker/main_flow_based.py start
```

## 🌐 Production Deployment

### Server: 18.210.238.67 (Ubuntu 24)

#### Smart Deployment (PowerShell)
```powershell
# Deploy with intelligent file synchronization
.\deploy.ps1

# Test deployment
.\test-deployment.ps1

# View logs
.\logs-backend.ps1    # Backend logs
.\logs-worker.ps1     # Worker logs
```

### Service URLs
- **Frontend**: https://voice-agent-livekit.vercel.app
- **Backend API**: https://18.210.238.67.nip.io
- **Health Check**: https://18.210.238.67.nip.io/health
- **Template Status**: https://18.210.238.67.nip.io/api/template_status

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
voice-agent-livekit-affan/
├── deploy.ps1              # 🚀 Smart deployment script
├── test-deployment.ps1     # 🧪 Test deployment
├── logs-backend.ps1        # 📊 Backend logs
├── logs-worker.ps1         # 📊 Worker logs
├── backend/                # Backend API application
│   ├── main_dynamic.py     # FastAPI backend with orchestrator
│   ├── llm_utils.py        # Centralized LLM functions
│   ├── conversational_orchestrator.py  # Intelligent conversation routing
│   ├── persistence/        # 📊 Local data persistence
│   │   ├── flow_states/    # Flow state JSON files
│   │   ├── user_profiles/  # User profile JSON files
│   │   └── debug_logs/     # Debug logs for analysis
│   └── worker/             # LiveKit worker
├── frontend/               # Frontend UI
│   ├── index.html          # Main HTML
│   └── main_dynamic.js     # JavaScript client
├── docs/                   # Documentation
├── KMS/                    # KMS logs
├── README.md               # This file
└── requirements.txt        # Python dependencies
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
```

## 🎯 Key Features

- **Smart Deployment**: Only syncs missing/changed files
- **HTTPS Support**: Secure communication with SSL certificates
- **Auto-Renewal**: SSL certificates automatically renew via Let's Encrypt
- **Service Management**: Systemd services with auto-restart
- **Real-time Logs**: Easy log monitoring with PowerShell scripts
- **Health Monitoring**: Built-in health checks and status endpoints
- **🧠 Intelligent Orchestrator**: GPT-4 powered conversation routing with context awareness
- **📊 Local Persistence**: User profiles and flow states saved to JSON files for debugging
- **🐛 Debug System**: Comprehensive logging and API endpoints for testing and analysis

## 🐛 Debug & Testing System

### Local Persistence
The system automatically saves conversation data to JSON files for debugging and testing:

- **User Profiles**: `backend/persistence/user_profiles/` - Extracted user information, preferences, refused fields
- **Flow States**: `backend/persistence/flow_states/` - Current conversation flow and progress
- **Debug Logs**: `backend/persistence/debug_logs/` - Orchestrator decisions and reasoning

### Debug API Endpoints

#### View Room Data
```bash
# List all rooms with saved data
curl https://18.210.238.67.nip.io/api/debug/rooms

# View specific room data (flow state, user profile, debug logs)
curl https://18.210.238.67.nip.io/api/debug/room/{room_name}
```

#### Clear Room Data
```bash
# Clear all debug data for a specific room
curl -X DELETE https://18.210.238.67.nip.io/api/debug/room/{room_name}
```

### What You Can Debug
- **User Data Extraction**: See what information was extracted from user messages
- **Orchestrator Decisions**: Understand why the system made specific routing decisions
- **Flow Progression**: Track how conversations flow through different steps
- **Refusal Handling**: See how the system handles user refusals and preferences
- **Context Preservation**: Verify that conversation context is maintained

## 📞 Troubleshooting

### Common Issues
1. **External Access Failed**: Check AWS Security Group (port 80)
2. **Service Won't Start**: Check logs with `.\logs-backend.ps1`
3. **Missing Files**: Re-run `.\deploy.ps1` to sync files
4. **Flow Not Progressing**: Check debug logs to see orchestrator decisions
5. **User Data Not Extracted**: Review user profile persistence files

### Log Commands
```powershell
# View service logs
.\logs-backend.ps1
.\logs-worker.ps1

# Test deployment
.\test-deployment.ps1
```

### Debug Commands
```bash
# Check orchestrator decisions
curl https://18.210.238.67.nip.io/api/debug/room/{room_name}

# View all active rooms
curl https://18.210.238.67.nip.io/api/debug/rooms
```