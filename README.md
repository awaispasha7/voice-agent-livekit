# Voice Agent Project

This project implements a voice agent using LiveKit and FastAPI. It consists of three main components: a worker that processes voice data, a token server for authentication, and a web interface for user interaction.

## Project Structure

```
voice-agent
├── worker
│   ├── main.py              # LiveKit worker with VoicePipelineAgent
├── token-server
│   ├── main.py              # FastAPI token server
├── web
│   ├── index.html           # Main HTML file for the web interface
│   └── main.js              # JavaScript code for the web application
├── requirements.txt
└── README.md                # Project documentation
```

## Setup Instructions

1. **Clone the repository:**
   ```
   git clone <repository-url>
   cd voice-agent
   ```

2. **pip install requirments.txt**

3. **create a .env file in the voice agent folder, place your api keys in there**

      OPENAI_API_KEY=
      LIVEKIT_URL=
      LIVEKIT_API_KEY=
      LIVEKIT_API_SECRET=
      DEEPGRAM_API_KEY=
      CARTESIA_API_KEY=

4. **Run the services:**
   - Start the worker:
     ```
     python worker/main_dynamic.py download-files
     python worker/main_dynamic.py dev
     ```
   - Start the token server:
     ``` 
     python token-server/main_dynamic.py

     ```

5. **Access the web interface:**
   - cd web
   - npx serve -s . -l 5000

## Usage

- The worker processes voice data and manages connections using the VoicePipelineAgent.
- The token server generates and serves tokens for authentication with LiveKit.
- The web interface allows users to interact with the voice agent and manage voice data.

## Contributing

Contributions are welcome! Please submit a pull request or open an issue for any enhancements or bug fixes.

## License

This project is licensed under the MIT License. See the LICENSE file for more details.