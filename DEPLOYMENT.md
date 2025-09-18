# Voice Agent Deployment Guide

This guide covers deploying the Alive5 Voice Agent using the new hosting stack:

| Component   | Hosting Choice     | Cost (Monthly)           |
| ----------- | ------------------ | ------------------------ |
| Frontend    | Vercel (Free)      | \$0                      |
| Backend API | Render Free        | \$0 (or \$7 if upgraded) |
| Worker      | Render 1 GB Worker | \$15                     |
| **Total**   |                    | **\$15–\$22/month**      |

## Prerequisites

1. **Render Account**: Sign up at [render.com](https://render.com)
2. **GitHub Repository**: Push your code to GitHub
3. **Environment Variables**: Prepare all required environment variables
4. **Frontend**: Already deployed to Vercel ✅

## Environment Variables

### Backend API (Render)
- `LIVEKIT_API_KEY`: Your LiveKit API key
- `LIVEKIT_API_SECRET`: Your LiveKit API secret
- `LIVEKIT_URL`: Your LiveKit WebSocket URL
- `OPENAI_API_KEY`: Your OpenAI API key
- `A5_BASE_URL`: Alive5 API base URL
- `A5_API_KEY`: Alive5 API key

### Worker (Render)
- Same environment variables as Backend API

## Deployment Steps

### 1. Deploy Backend API to Render (Free Plan)

1. Go to [Render Dashboard](https://dashboard.render.com)
2. Click "New +" → "Web Service"
3. Connect your GitHub repository
4. Configure the service:
   - **Name**: `voice-agent-backend`
   - **Environment**: `Python 3`
   - **Build Command**: `pip install -r requirements.txt`
   - **Start Command**: `uvicorn backend.main_dynamic:app --host=0.0.0.0 --port=$PORT`
   - **Plan**: Free
5. Add all environment variables
6. Click "Create Web Service"
7. Wait for deployment to complete
8. Note the service URL (e.g., `https://voice-agent-backend.onrender.com`)

### 2. Deploy Worker to Render (Free Plan)

1. In Render Dashboard, click "New +" → "Background Worker"
2. Connect your GitHub repository
3. Configure the worker:
   - **Name**: `voice-agent-worker`
   - **Environment**: `Python 3`
   - **Build Command**: `pip install -r requirements.txt`
   - **Start Command**: `python backend/worker/main_flow_based.py`
   - **Plan**: Free
4. Add all environment variables (same as backend)
5. Click "Create Background Worker"
6. Wait for deployment to complete

**Note**: Both services use Render's free plan, keeping costs at $0/month.

### 3. Frontend Already Deployed ✅

The frontend has already been deployed to Vercel and is ready to use:
**Frontend URL**: [https://voice-agent-livekit-affan.vercel.app/](https://voice-agent-livekit-affan.vercel.app/)

### 4. Update Frontend Configuration

The frontend now uses environment variables from your `.env` file for configuration. You have several options:

#### Option A: Set Environment Variables in Vercel
1. Go to your Vercel project dashboard
2. Go to Settings → Environment Variables
3. Add these variables:
   - `BACKEND_URL` = `https://your-backend-url.onrender.com`
   - `FRONTEND_URL` = `https://voice-agent-livekit-affan.vercel.app`
4. Redeploy the project

#### Option B: Local Development (Default)
- The frontend defaults to `http://localhost:8000` for backend
- The frontend defaults to `http://localhost:3000` for frontend
- No additional configuration needed for local testing

## Testing the Deployment

1. **Test Backend API**:
   ```bash
   curl https://your-backend-url.onrender.com/health
   ```

2. **Test Frontend**:
   - Open your Vercel URL
   - Try joining a room
   - Check browser console for any errors

3. **Test Worker**:
   - Check Render logs for worker activity
   - Ensure it's processing voice sessions

## Monitoring and Maintenance

### Render Monitoring
- Check service health in Render dashboard
- Monitor logs for errors
- Set up alerts for downtime

### Vercel Monitoring
- Check deployment status in Vercel dashboard
- Monitor function logs if using serverless functions

### Cost Optimization
- Monitor Render usage to avoid overage charges
- Consider upgrading to paid plans for better performance
- Use Render's auto-sleep feature for free tier

## Troubleshooting

### Common Issues

1. **Backend not responding**:
   - Check Render logs
   - Verify environment variables
   - Ensure worker is running

2. **Frontend can't connect to backend**:
   - Check CORS settings
   - Verify API URLs in config.js
   - Check browser console for errors

3. **Worker not processing**:
   - Check worker logs in Render
   - Verify environment variables
   - Ensure LiveKit credentials are correct

### Environment Variable Issues
- Double-check all environment variables are set correctly
- Ensure no extra spaces or quotes in values
- Restart services after changing environment variables

## Migration from Heroku

If migrating from Heroku:

1. Export environment variables from Heroku:
   ```bash
   heroku config -a your-app-name
   ```

2. Set the same variables in Render

3. Update frontend configuration to point to new backend URL

4. Test thoroughly before switching DNS/domains

## Security Considerations

- Use environment variables for all sensitive data
- Enable HTTPS (automatic with Vercel and Render)
- Regularly rotate API keys
- Monitor for unusual activity

## Support

- **Render Support**: [render.com/docs](https://render.com/docs)
- **Vercel Support**: [vercel.com/docs](https://vercel.com/docs)
- **LiveKit Support**: [docs.livekit.io](https://docs.livekit.io)
