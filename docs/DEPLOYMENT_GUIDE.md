# 🚀 Alive5 Voice Agent - Deployment Guide

## Overview

This guide provides step-by-step instructions for deploying the Alive5 Voice Agent system to production. The system consists of three main components:

- **Frontend** (Vercel) - Voice interface and real-time communication
- **Backend API** (Render) - Flow logic and intent detection
- **Worker** (Render) - Voice processing and LiveKit integration

## Prerequisites

### Required Accounts
- [Vercel](https://vercel.com) account (free tier)
- [Render](https://render.com) account (free tier)
- [GitHub](https://github.com) account
- [LiveKit](https://livekit.io) account
- [OpenAI](https://openai.com) account
- [Alive5](https://alive5.com) account

### Required Credentials
- LiveKit API Key and Secret
- OpenAI API Key
- Alive5 API Key and Base URL
- GitHub repository access

## Step 1: Repository Setup

### 1.1 Fork/Clone Repository
```bash
git clone https://github.com/your-username/voice-agent-livekit-affan.git
cd voice-agent-livekit-affan
```

### 1.2 Environment Configuration
Create a `.env` file with your production credentials:

```bash
# LiveKit Configuration
LIVEKIT_URL=wss://your-livekit-url.livekit.cloud
LIVEKIT_API_KEY=your-livekit-api-key
LIVEKIT_API_SECRET=your-livekit-api-secret

# OpenAI Configuration
OPENAI_API_KEY=your-openai-api-key

# Alive5 Configuration
A5_BASE_URL=https://api-v2-stage.alive5.com
A5_API_KEY=your-a5-api-key
A5_FAQ_BOT_ID=your-faq-bot-id

# Application URLs (will be updated after deployment)
FRONTEND_URL=https://your-frontend-url.vercel.app
BACKEND_URL=https://your-backend-url.onrender.com
```

## Step 2: Frontend Deployment (Vercel)

### 2.1 Connect to Vercel
1. Go to [Vercel Dashboard](https://vercel.com/dashboard)
2. Click "New Project"
3. Import your GitHub repository
4. Select the `frontend` folder as the root directory

### 2.2 Configure Build Settings
- **Framework Preset**: Other
- **Build Command**: (leave empty)
- **Output Directory**: (leave empty)
- **Install Command**: (leave empty)

### 2.3 Environment Variables
Add these environment variables in Vercel:

| Variable | Value | Description |
|----------|-------|-------------|
| `API_BASE_URL` | `https://your-backend-url.onrender.com` | Backend API URL |
| `BACKEND_URL` | `https://your-backend-url.onrender.com` | Backend API URL |
| `FRONTEND_URL` | `https://your-frontend-url.vercel.app` | Frontend URL |

### 2.4 Deploy
1. Click "Deploy"
2. Wait for deployment to complete
3. Note the deployment URL (e.g., `https://voice-agent-frontend.vercel.app`)

## Step 3: Backend Deployment (Render)

### 3.1 Create Web Service
1. Go to [Render Dashboard](https://dashboard.render.com)
2. Click "New +" → "Web Service"
3. Connect your GitHub repository

### 3.2 Configure Service
- **Name**: `voice-agent-backend`
- **Environment**: `Python 3`
- **Region**: Choose closest to your users
- **Branch**: `main` (or your default branch)
- **Root Directory**: (leave empty)
- **Build Command**: `pip install -r requirements.txt`
- **Start Command**: `uvicorn backend.main_dynamic:app --host=0.0.0.0 --port=$PORT`

### 3.3 Environment Variables
Add these environment variables in Render:

| Variable | Value | Description |
|----------|-------|-------------|
| `LIVEKIT_URL` | `wss://your-livekit-url.livekit.cloud` | LiveKit WebSocket URL |
| `LIVEKIT_API_KEY` | `your-livekit-api-key` | LiveKit API Key |
| `LIVEKIT_API_SECRET` | `your-livekit-api-secret` | LiveKit API Secret |
| `OPENAI_API_KEY` | `your-openai-api-key` | OpenAI API Key |
| `A5_BASE_URL` | `https://api-v2-stage.alive5.com` | Alive5 API Base URL |
| `A5_API_KEY` | `your-a5-api-key` | Alive5 API Key |
| `A5_FAQ_BOT_ID` | `your-faq-bot-id` | Alive5 FAQ Bot ID |
| `FRONTEND_URL` | `https://your-frontend-url.vercel.app` | Frontend URL |
| `BACKEND_URL` | `https://your-backend-url.onrender.com` | Backend URL |
| `PYTHON_VERSION` | `3.11.0` | Python Version |

### 3.4 Deploy
1. Click "Create Web Service"
2. Wait for deployment to complete
3. Note the service URL (e.g., `https://voice-agent-backend.onrender.com`)

## Step 4: Worker Deployment (Render)

### 4.1 Create Background Worker
1. Go to [Render Dashboard](https://dashboard.render.com)
2. Click "New +" → "Background Worker"
3. Connect your GitHub repository

### 4.2 Configure Worker
- **Name**: `voice-agent-worker`
- **Environment**: `Python 3`
- **Region**: Same as backend
- **Branch**: `main` (or your default branch)
- **Root Directory**: (leave empty)
- **Build Command**: `pip install -r requirements.txt`
- **Start Command**: `python backend/worker/main_flow_based.py`

### 4.3 Environment Variables
Add the same environment variables as the backend:

| Variable | Value | Description |
|----------|-------|-------------|
| `LIVEKIT_URL` | `wss://your-livekit-url.livekit.cloud` | LiveKit WebSocket URL |
| `LIVEKIT_API_KEY` | `your-livekit-api-key` | LiveKit API Key |
| `LIVEKIT_API_SECRET` | `your-livekit-api-secret` | LiveKit API Secret |
| `OPENAI_API_KEY` | `your-openai-api-key` | OpenAI API Key |
| `A5_BASE_URL` | `https://api-v2-stage.alive5.com` | Alive5 API Base URL |
| `A5_API_KEY` | `your-a5-api-key` | Alive5 API Key |
| `A5_FAQ_BOT_ID` | `your-faq-bot-id` | Alive5 FAQ Bot ID |
| `BACKEND_URL` | `https://your-backend-url.onrender.com` | Backend URL |
| `PYTHON_VERSION` | `3.11.0` | Python Version |

### 4.4 Deploy
1. Click "Create Background Worker"
2. Wait for deployment to complete

## Step 5: Update Configuration

### 5.1 Update Frontend Environment Variables
1. Go back to Vercel Dashboard
2. Navigate to your project settings
3. Update environment variables with actual URLs:
   - `API_BASE_URL`: `https://your-backend-url.onrender.com`
   - `BACKEND_URL`: `https://your-backend-url.onrender.com`
   - `FRONTEND_URL`: `https://your-frontend-url.vercel.app`

### 5.2 Redeploy Frontend
1. Trigger a new deployment in Vercel
2. Wait for deployment to complete

## Step 6: Testing

### 6.1 Health Check
Test the backend API:
```bash
curl https://your-backend-url.onrender.com/health
```

Expected response:
```json
{
  "status": "ok",
  "active_sessions": 0,
  "timestamp": 1640995200.123
}
```

### 6.2 Frontend Test
1. Open your frontend URL in a browser
2. Allow microphone permissions
3. Test voice interaction

### 6.3 End-to-End Test
1. Say "Hello" - should get greeting response
2. Say "I need pricing information" - should start pricing flow
3. Say "Can I speak with someone over the phone?" - should initiate transfer

## Step 7: Monitoring Setup

### 7.1 Render Monitoring
- Monitor service health in Render dashboard
- Set up alerts for service failures
- Monitor resource usage

### 7.2 Vercel Monitoring
- Monitor deployment status
- Check function execution logs
- Monitor performance metrics

### 7.3 Application Monitoring
- Check backend logs for errors
- Monitor worker logs for voice processing issues
- Track session success rates

## Step 8: Production Optimization

### 8.1 Performance Optimization
- Enable Vercel's CDN for static assets
- Configure Render's auto-scaling
- Optimize API response times

### 8.2 Security Hardening
- Enable HTTPS (automatic with Vercel/Render)
- Configure CORS properly
- Set up rate limiting
- Monitor for security issues

### 8.3 Backup Strategy
- Regular database backups (if using persistent storage)
- Code repository backups
- Environment variable backups

## Troubleshooting

### Common Issues

#### 1. Backend Not Starting
**Symptoms**: 502 Bad Gateway errors
**Solutions**:
- Check environment variables
- Verify Python version compatibility
- Check build logs for errors

#### 2. Worker Not Connecting
**Symptoms**: No voice responses
**Solutions**:
- Verify LiveKit credentials
- Check worker logs
- Ensure backend URL is correct

#### 3. Frontend Not Loading
**Symptoms**: Blank page or errors
**Solutions**:
- Check environment variables
- Verify build process
- Check browser console for errors

#### 4. Intent Detection Not Working
**Symptoms**: Generic responses only
**Solutions**:
- Verify OpenAI API key
- Check bot template loading
- Review intent detection logs

### Debug Commands

#### Check Backend Health
```bash
curl -X GET https://your-backend-url.onrender.com/health
```

#### Test Intent Detection
```bash
curl -X POST https://your-backend-url.onrender.com/api/test_intent_detection \
  -H "Content-Type: application/json" \
  -d '{"user_message": "I need pricing information"}'
```

#### Check Template Info
```bash
curl -X GET https://your-backend-url.onrender.com/api/template_info
```

## Cost Optimization

### Free Tier Limits
- **Vercel**: 100GB bandwidth, 100 serverless function executions
- **Render**: 750 hours/month for free tier services

### Optimization Tips
- Monitor usage regularly
- Optimize API calls
- Use efficient data structures
- Implement caching where appropriate

## Maintenance

### Regular Tasks
- Monitor service health
- Update dependencies
- Review logs for issues
- Test functionality regularly

### Updates
- Deploy updates through Git
- Test in staging environment first
- Monitor deployment closely
- Rollback if issues occur

---

## Support

For deployment issues or questions:
1. Check the troubleshooting section
2. Review service logs
3. Contact the development team
4. Check service status pages

**Deployment completed successfully! 🎉**
