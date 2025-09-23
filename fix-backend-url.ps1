#!/usr/bin/env pwsh

Write-Host "Fixing Worker Backend URL Configuration" -ForegroundColor Cyan
Write-Host "===========================================" -ForegroundColor Cyan

$SERVER_IP = "18.210.238.67"
$SSH_KEY = "alive5-voice-ai-agent.pem"
$USER = "ubuntu"

Write-Host "`nCurrent issue: Worker trying to connect to port 8000, but backend runs on port 80" -ForegroundColor Red

Write-Host "`n1. Checking current backend URL configuration..." -ForegroundColor Yellow
ssh -i $SSH_KEY -o ConnectTimeout=10 $USER@$SERVER_IP "grep -n 'BACKEND_URL' /home/ubuntu/voice-agent-livekit-affan/backend/worker/main_flow_based.py"

Write-Host "`n2. Fixing backend URL configuration..." -ForegroundColor Green
ssh -i $SSH_KEY -o ConnectTimeout=10 $USER@$SERVER_IP "
    # Update .env file
    if grep -q 'BACKEND_URL' /home/ubuntu/voice-agent-livekit-affan/.env; then
        sed -i 's|BACKEND_URL=.*|BACKEND_URL=https://18.210.238.67.nip.io|' /home/ubuntu/voice-agent-livekit-affan/.env
    else
        echo 'BACKEND_URL=https://18.210.238.67.nip.io' >> /home/ubuntu/voice-agent-livekit-affan/.env
    fi
    
    # Update worker file
    sed -i 's|BACKEND_URL = os.getenv(\"BACKEND_URL\", \"http://localhost:8000\")|BACKEND_URL = os.getenv(\"BACKEND_URL\", \"https://18.210.238.67.nip.io\")|' /home/ubuntu/voice-agent-livekit-affan/backend/worker/main_flow_based.py
    
    echo 'Updated BACKEND_URL in .env:'
    grep 'BACKEND_URL' /home/ubuntu/voice-agent-livekit-affan/.env
"

Write-Host "`n3. Restarting worker service..." -ForegroundColor Green
ssh -i $SSH_KEY -o ConnectTimeout=10 $USER@$SERVER_IP "
    sudo systemctl restart alive5-worker
    sleep 3
    sudo systemctl status alive5-worker --no-pager -l
"

Write-Host "`n4. Testing the fix..." -ForegroundColor Green
ssh -i $SSH_KEY -o ConnectTimeout=10 $USER@$SERVER_IP "
    curl -s -o /dev/null -w '%{http_code}' https://18.210.238.67.nip.io/health
    echo ' - Backend health check'
    
    echo 'Recent worker logs:'
    sudo journalctl -u alive5-worker --no-pager -n 5
"

Write-Host "`nBackend URL fix applied!" -ForegroundColor Green
Write-Host "Key changes:" -ForegroundColor White
Write-Host "- Updated BACKEND_URL from http://18.210.238.67:8000 to https://18.210.238.67.nip.io" -ForegroundColor White
Write-Host "- Updated .env file with correct URL" -ForegroundColor White
Write-Host "- Restarted worker service" -ForegroundColor White

Write-Host "`nThe worker should now connect properly to the backend!" -ForegroundColor Cyan
Write-Host "Test your voice agent again - it should respond much faster now." -ForegroundColor White
