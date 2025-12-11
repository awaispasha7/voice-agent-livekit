# 🎙️ Alive5 Voice Agent

AI-powered voice agent with intelligent conversation flows, intent detection, and seamless FAQ integration.

## 🚀 Quick Start

### Prerequisites
- Python 3.12+ (for local development)
- Docker and Docker Compose (for containerized deployment)
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

## 🐳 Docker Deployment

### Prerequisites
- Docker Engine 20.10+
- Docker Compose 2.0+
- `.env` file with all required environment variables

### Quick Start with Docker
```bash
# Build and start all services
docker-compose up -d

# View logs
docker-compose logs -f

# Stop all services
docker-compose down
```

### Building Containers
```bash
# Build all services
docker-compose build

# Build specific service
docker-compose build backend
docker-compose build worker

# Build without cache (fresh build)
docker-compose build --no-cache

# Build and start in one command
docker-compose up -d --build
```

### Running Containers
```bash
# Start all services in detached mode
docker-compose up -d

# Start specific service
docker-compose up -d backend
docker-compose up -d worker

# Start with logs visible
docker-compose up

# Restart all services
docker-compose restart

# Restart specific service
docker-compose restart backend
docker-compose restart worker
```

### Viewing Logs
```bash
# All services logs (follow mode)
docker-compose logs -f

# Specific service logs
docker-compose logs -f backend
docker-compose logs -f worker

# Last 100 lines
docker-compose logs --tail=100

# Logs since specific time
docker-compose logs --since 10m

# View logs without following
docker-compose logs
```

### Container Management
```bash
# List running containers
docker-compose ps

# Check container status
docker-compose ps backend
docker-compose ps worker

# Stop all services
docker-compose stop

# Stop specific service
docker-compose stop backend
docker-compose stop worker

# Remove containers (keeps volumes)
docker-compose down

# Remove containers and volumes
docker-compose down -v

# Remove containers, volumes, and images
docker-compose down -v --rmi all
```

### Executing Commands in Containers
```bash
# Execute command in backend container
docker-compose exec backend bash
docker-compose exec backend python --version

# Execute command in worker container
docker-compose exec worker bash
docker-compose exec worker python alive5-backend/alive5-worker/worker.py download-files

# Run one-off command
docker-compose run --rm backend python alive5-backend/main.py
docker-compose run --rm worker python alive5-backend/alive5-worker/worker.py dev
```

### Debugging
```bash
# Check container health
docker-compose ps

# Inspect container
docker inspect voice-agent-livekit-backend-1
docker inspect voice-agent-livekit-worker-1

# View container resource usage
docker stats

# Check container logs for errors
docker-compose logs backend | grep -i error
docker-compose logs worker | grep -i error

# Access container shell
docker-compose exec backend sh
docker-compose exec worker sh
```

### Environment Variables
```bash
# Verify .env file is loaded
docker-compose config

# Override .env with environment variables
BACKEND_URL=http://localhost:8000 docker-compose up

# Check environment variables in running container
docker-compose exec backend env
docker-compose exec worker env
```

### Troubleshooting Docker Issues
```bash
# Rebuild from scratch
docker-compose down -v
docker-compose build --no-cache
docker-compose up -d

# Check if ports are in use
# Windows
netstat -ano | findstr :8000
# Linux/Mac
lsof -i :8000

# View Docker system info
docker system df
docker system prune  # Clean up unused resources (be careful!)

# Check Docker Compose version
docker-compose version

# Validate docker-compose.yml
docker-compose config
```

### Production Deployment with Docker
```bash
# Build production images
docker-compose -f docker-compose.yml build

# Start services with restart policy
docker-compose up -d

# Monitor services
docker-compose ps
docker-compose logs -f

# Update services (after code changes)
docker-compose pull  # If using images from registry
docker-compose build
docker-compose up -d
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
├── docker-compose.yml      # 🐳 Docker Compose configuration
├── Dockerfile.backend       # 🐳 Backend container definition
├── Dockerfile.worker       # 🐳 Worker container definition
├── alive5-backend/         # Backend API application
│   ├── main.py            # FastAPI backend (simplified)
│   ├── cached_voices.json # Voice configuration
│   └── alive5-worker/     # LiveKit worker
│       ├── worker.py      # Main worker entry point
│       ├── functions.py   # Business logic (flows, FAQ)
│       ├── system_prompt.py # LLM system prompt
│       └── docker-entrypoint.sh # Worker container entrypoint
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
- **🧠 Intelligent Agent**: gpt-5-nano-2025-08-07 powered conversation with function calling
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