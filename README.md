# Alive5 Voice Agent Project

This project implements a dynamic voice agent using LiveKit, FastAPI, and Alive5 flow integration. It consists of three main components: a worker that processes voice data, a token server for authentication and flow management, and a web interface for user interaction.

## Project Structure

```
voice-agent
├── worker/
│   ├── main_flow_based.py   # 🚀 Flow-based worker (RECOMMENDED)
│   ├── main_dynamic.py      # Intent-based worker
│   └── README.md            # Worker documentation
├── token-server/
│   ├── main_dynamic.py      # FastAPI server with flow management
│   ├── Procfile             # Heroku deployment config
│   └── requirements.txt     # Backend dependencies
├── web/
│   ├── index.html           # Main HTML file
│   └── main_dynamic.js      # Frontend with flow processing
├── requirements.txt         # Dependencies
└── README.md                # Project documentation
```

## Setup Instructions

1. **Clone the repository:**
   ```
   git clone <repository-url>
   cd voice-agent
   ```

2. **pip install requirments.txt**

3. **Create a .env file in the voice agent folder with your API keys:**

     ```
     OPENAI_API_KEY=your_openai_key
     LIVEKIT_URL=your_livekit_url
     LIVEKIT_API_KEY=your_livekit_key
     LIVEKIT_API_SECRET=your_livekit_secret
     DEEPGRAM_API_KEY=your_deepgram_key
     CARTESIA_API_KEY=your_cartesia_key
     A5_BASE_URL=https://api-v2-stage.alive5.com
     A5_API_KEY=your_alive5_key
     ```

4. **Run the services:**
   - **Flow-based worker (recommended):**
     ```
     python worker/main_flow_based.py download-files
     python worker/main_flow_based.py dev
     ```
   - **Alternative worker:**
     ```
     python worker/main_dynamic.py dev    # Intent-based
     ```
   - **Start the token server:**
     ``` 
     python token-server/main_dynamic.py
     ```

5. **Access the web interface:**
   ```
   cd web
   npx serve -s . -l 5000
   ```

## Features

### 🚀 Flow-Based System (Recommended)
- **Dynamic template loading** from Alive5 API
- **Structured conversation flows** (pricing, support, billing, agent transfer)
- **Real-time template updates** without code changes
- **Fallback to FAQ bot** for general questions

### 🔄 Intent-Based System
- **AI-powered intent detection** (sales, support, billing)
- **Dynamic conversation adaptation**
- **User data extraction** and session tracking

## Usage

- **Flow-based worker**: Follows client-defined conversation templates
- **Token server**: Generates authentication tokens and manages flow processing
- **Web interface**: Real-time voice interaction with live transcript display
- **Testing**: `python worker/main_flow_based.py test`

## Contributing

Contributions are welcome! Please submit a pull request or open an issue for any enhancements or bug fixes.

## License

This project is licensed under the MIT License. See the LICENSE file for more details.