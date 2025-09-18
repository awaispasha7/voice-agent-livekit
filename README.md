# Alive5 Voice Agent Project

This project implements a dynamic voice agent using LiveKit, FastAPI, and Alive5 flow integration. It consists of three main components: a worker that processes voice data, a backend API for authentication and flow management, and a web interface for user interaction.

## 🚀 New Hosting Stack

| Component   | Hosting Choice     | Cost (Monthly)           |
| ----------- | ------------------ | ------------------------ |
| Frontend    | Vercel (Free)      | \$0                      |
| Backend API | Render Free        | \$0                      |
| Worker      | Render Free        | \$0                      |
| **Total**   |                    | **\$0/month**            |

## Project Structure

```
voice-agent-livekit-affan/
├── backend/
│   ├── main_dynamic.py      # FastAPI server with flow management
│   └── worker/
│       ├── main_flow_based.py   # 🚀 Flow-based worker
│       └── README.md            # Worker documentation
├── frontend/
│   ├── index.html           # Main HTML file
│   └── main_dynamic.js      # Frontend with flow processing
├── requirements.txt         # Backend dependencies
├── render.yaml              # Render deployment config
├── .python-version          # Python version specification
├── DEPLOYMENT.md            # Detailed deployment guide
└── README.md                # Project documentation
```

## Quick Start

### Local Development

1. **Clone the repository:**
   ```bash
   git clone <repository-url>
   cd voice-agent-livekit-affan
   ```

2. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

3. **Create a .env file with your API keys:**
   ```bash
   OPENAI_API_KEY=your_openai_key
   LIVEKIT_URL=your_livekit_url
   LIVEKIT_API_KEY=your_livekit_key
   LIVEKIT_API_SECRET=your_livekit_secret
   A5_BASE_URL=https://api-v2-stage.alive5.com
   A5_API_KEY=your_alive5_key
   ```

4. **Run the services:**
   - **Backend API:**
     ```bash
     uvicorn backend.main_dynamic:app --host=0.0.0.0 --port=8000
     ```
   
   - **Worker:**
     ```bash
     python backend/worker/main_flow_based.py dev
     ```

5. **Access the web interface:**
   ```bash
   cd frontend
   python -m http.server 3000
   ```
   Open http://localhost:3000

   **Note**: The frontend will automatically connect to `http://localhost:8000` for the backend API.

### Production Deployment

**Frontend**: Already deployed at [https://voice-agent-livekit-affan.vercel.app/](https://voice-agent-livekit-affan.vercel.app/)

For backend and worker deployment, see [DEPLOYMENT.md](DEPLOYMENT.md) for detailed instructions on deploying to Render.

## Features

### 🚀 Flow-Based System (Recommended)
- **Dynamic template loading** from Alive5 API
- **Structured conversation flows** (pricing, support, billing, agent transfer)
- **Real-time template updates** without code changes
- **Fallback to FAQ bot** for general questions



## Usage

- **Flow-based worker**: Follows client-defined conversation templates
- **Token server**: Generates authentication tokens and manages flow processing
- **Web interface**: Real-time voice interaction with live transcript display
- **Testing**: `python worker/main_flow_based.py test`

## Contributing

Contributions are welcome! Please submit a pull request or open an issue for any enhancements or bug fixes.

## License

This project is licensed under the MIT License. See the LICENSE file for more details.