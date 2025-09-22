# 🎙️ Alive5 Voice Agent

AI-powered voice agent with intelligent conversation flows, intent detection, and seamless agent transfers.

## 🚀 Quick Start

### Prerequisites
- Python 3.11+
- LiveKit, OpenAI, and Alive5 API credentials

### Local Development
```bash
git clone <repository-url>
cd voice-agent-livekit-affan
pip install -r requirements.txt
cp .env.example .env  # Edit with your credentials
```

### Run Services
```bash
# Backend API
uvicorn backend.main_dynamic:app --host=0.0.0.0 --port=8000

# Worker
python backend/worker/main_flow_based.py
```

## 🌐 Production Deployment

### Server: 18.210.238.67 (Ubuntu 24)

#### PowerShell (Windows)
```powershell
# Deploy to client server
.\deploy-to-client-server.ps1

# Test deployment
.\test-deployment.ps1
```

#### Bash (Linux/Mac)
```bash
# Deploy to client server
chmod +x deploy-to-client-server.sh
./deploy-to-client-server.sh

# Test deployment
chmod +x test-deployment.sh
./test-deployment.sh
```

### Service URLs
- **Backend API**: http://18.210.238.67:8000
- **Health Check**: http://18.210.238.67/health

## 🔧 Management
```bash
# Check status
ssh -i alive5-voice-ai-agent.pem ubuntu@18.210.238.67 'sudo systemctl status alive5-backend alive5-worker'

# View logs
ssh -i alive5-voice-ai-agent.pem ubuntu@18.210.238.67 'sudo journalctl -u alive5-backend -f'
```

## 📁 Project Structure
```
├── backend/           # Backend API
├── frontend/          # Frontend UI
├── .env              # Environment variables
├── requirements.txt  # Dependencies
└── deploy-to-client-server.sh  # Deployment script
```

## 🔑 Environment Variables
```bash
OPENAI_API_KEY=your-key
LIVEKIT_URL=wss://your-livekit-url
LIVEKIT_API_KEY=your-key
LIVEKIT_API_SECRET=your-secret
A5_API_KEY=your-key
A5_BASE_URL=https://api-v2-stage.alive5.com
BACKEND_URL=http://18.210.238.67:8000
```

## 📞 Support
Check service logs for troubleshooting:
```bash
sudo journalctl -u alive5-backend -f
sudo journalctl -u alive5-worker -f
```