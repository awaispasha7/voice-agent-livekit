# Environment Variables Update Script

## Overview

The `update-all-env.ps1` script is a unified solution for updating environment variables across your entire voice agent system. It updates both the frontend (Vercel) and backend (server) environment variables in one command.

## What It Does

1. **Frontend (Vercel)**: Updates all environment variables in your Vercel deployment
2. **Backend (Server)**: Uploads the `.env` file to your server and restarts the services
3. **Health Check**: Verifies that the backend is running correctly after the update

## Prerequisites

- **Vercel CLI**: Must be installed and logged in
  ```bash
  npm install -g vercel
  vercel login
  ```
- **SSH Key**: The `alive5-voice-ai-agent.pem` file must be in the project root
- **PowerShell**: Run on Windows PowerShell or PowerShell Core

## Usage

### Basic Usage
```powershell
.\update-all-env.ps1
```
This updates both frontend and backend with the current `.env` file.

### Dry Run (Recommended First)
```powershell
.\update-all-env.ps1 -DryRun
```
Shows what would be updated without making any changes.

### Different Vercel Environment
```powershell
.\update-all-env.ps1 -Environment preview
```
Updates the preview environment instead of production.

### Help
```powershell
.\update-all-env.ps1 -Help
```
Shows detailed usage information.

## Examples

```powershell
# Test what would be updated
.\update-all-env.ps1 -DryRun

# Update production environment
.\update-all-env.ps1

# Update preview environment
.\update-all-env.ps1 -Environment preview

# Update development environment
.\update-all-env.ps1 -Environment development
```

## What Happens When You Run It

1. **Reads** your local `.env` file
2. **Updates Vercel** environment variables (frontend)
3. **Uploads** `.env` file to your server
4. **Restarts** backend and worker services
5. **Tests** the API to ensure everything is working
6. **Reports** success/failure for each step

## Environment Variables

The script reads from your local `.env` file and updates these variables:

- `OPENAI_API_KEY`
- `LIVEKIT_URL`, `LIVEKIT_API_KEY`, `LIVEKIT_API_SECRET`
- `DEEPGRAM_API_KEY`
- `CARTESIA_API_KEY`
- `A5_API_KEY`, `A5_BASE_URL`, `A5_BOTCHAIN_NAME`, etc.
- `FRONTEND_URL`, `BACKEND_URL`
- And all other variables in your `.env` file

## Troubleshooting

### Vercel CLI Not Found
```bash
npm install -g vercel
vercel login
```

### SSH Key Not Found
Ensure `alive5-voice-ai-agent.pem` is in the project root directory.

### Services Not Restarting
Check that the SSH connection to your server is working:
```powershell
ssh -i alive5-voice-ai-agent.pem ubuntu@18.210.238.67 "echo 'Connection test'"
```

### API Health Check Fails
The script will show the error. Check the server logs:
```powershell
ssh -i alive5-voice-ai-agent.pem ubuntu@18.210.238.67 "sudo journalctl -u alive5-backend --no-pager -n 20"
```

## Verification

After running the script, you can verify the updates:

1. **Frontend**: Check your Vercel dashboard for updated environment variables
2. **Frontend**: Visit https://voice-agent-livekit.vercel.app
3. **Backend**: Visit https://18.210.238.67.nip.io/health
4. **Template**: Visit https://18.210.238.67.nip.io/api/template_status

## Benefits

- **One Command**: Updates everything at once
- **Consistent**: Ensures frontend and backend use the same environment variables
- **Safe**: Dry run mode lets you test before applying changes
- **Verified**: Automatically tests the API after updates
- **Clear Output**: Shows exactly what's happening at each step

## When to Use

- After changing any environment variables in your `.env` file
- When deploying to different environments (production, preview, development)
- When setting up the system for the first time
- When troubleshooting environment-related issues

This script ensures your voice agent always uses the latest environment variables across all components! 🚀
