# Alive5 HITL Voice Handoff Dashboard

Angular dashboard for human agents to handle voice call handoffs from AI agents.

## Features

- **Real-time Call Queue**: See incoming calls awaiting human assistance
- **LiveKit Integration**: Join calls directly via WebRTC
- **Alive5 Socket.IO**: Receive incoming call notifications
- **Call Controls**: Mute, end call, or transfer back to AI
- **Call Duration Tracking**: Real-time call timer

## Architecture

### Components

1. **CallQueueComponent**: Displays incoming calls, allows accept/reject
2. **ActiveCallComponent**: Shows active call details and controls
3. **CallHistoryComponent**: (Future) Shows completed call history

### Services

1. **LiveKitService**: Manages LiveKit room connection and audio streaming
2. **Alive5SocketService**: Handles Socket.IO connection for incoming call notifications
3. **CallStateService**: Coordinates call state between LiveKit and Alive5

## Setup

### Prerequisites

- Node.js 18+ and npm
- Angular CLI: `npm install -g @angular/cli`

### Installation

```bash
cd alive5-dashboard
npm install
```

### Configuration

Create `src/environments/environment.ts`:

```typescript
export const environment = {
  production: false,
  backendUrl: 'http://localhost:8000',
  alive5SocketUrl: 'wss://api-stage.alive5.com',
  alive5ApiKey: 'your-api-key-here'
};
```

### Development Server

```bash
ng serve
```

Navigate to `http://localhost:4200`

### Build for Production

```bash
ng build --configuration production
```

## Usage

### For Human Agents

1. **Login**: Enter your agent ID and name
2. **Connect**: Dashboard connects to Alive5 Socket.IO
3. **Monitor Queue**: Incoming calls appear in the call queue
4. **Accept Call**: Click "Accept Call" to join via LiveKit
5. **Handle Call**: Use mute/unmute, view caller info
6. **End Call**: End call or transfer back to AI

### Integration with Backend

The dashboard communicates with the FastAPI backend via:

- `POST /api/human-agent/request-takeover`: Register takeover intent
- `POST /api/human-agent/generate-token`: Get LiveKit token
- `POST /api/human-agent/end-handoff`: End the handoff

## Development

### Adding New Features

1. **New Component**: `ng generate component components/feature-name`
2. **New Service**: `ng generate service services/service-name`
3. **New Model**: Add to `models/` directory

### Testing

```bash
# Unit tests
ng test

# E2E tests
ng e2e
```

## Deployment

### Vercel (Recommended)

See [VERCEL_DEPLOYMENT.md](./VERCEL_DEPLOYMENT.md) for complete Vercel deployment instructions.

**Quick Deploy:**

1. Push code to GitHub/GitLab
2. Import project in Vercel
3. Set environment variables (BACKEND_URL, A5_SOCKET_URL, A5_API_KEY, LIVEKIT_URL)
4. Deploy!

### Docker (Alternative)

```dockerfile
FROM node:18-alpine as build
WORKDIR /app
COPY package*.json ./
RUN npm ci
COPY . .
RUN npm run build --configuration production

FROM nginx:alpine
COPY --from=build /app/dist/alive5-dashboard/browser /usr/share/nginx/html
EXPOSE 80
CMD ["nginx", "-g", "daemon off;"]
```

### Build and Run

```bash
docker build -t alive5-dashboard .
docker run -p 8080:80 alive5-dashboard
```

## Troubleshooting

### LiveKit Connection Issues

- Check `LIVEKIT_URL` in backend
- Verify LiveKit API credentials
- Check browser console for WebRTC errors

### Socket.IO Connection Issues

- Verify `A5_SOCKET_URL` configuration
- Check API key permissions
- Monitor browser network tab for socket events

### Audio Issues

- Ensure microphone permissions granted
- Check browser audio settings
- Verify LiveKit audio track publishing

## License

Proprietary - Alive5

