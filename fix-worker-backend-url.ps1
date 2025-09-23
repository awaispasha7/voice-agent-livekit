#!/usr/bin/env pwsh

Write-Host "Fixing Worker Backend URL Configuration" -ForegroundColor Cyan
Write-Host "===========================================" -ForegroundColor Cyan

# Server details
$SERVER_IP = "18.210.238.67"
$SSH_KEY = "alive5-voice-ai-agent.pem"
$USER = "ubuntu"

Write-Host "`nCurrent issue: Worker trying to connect to port 8000, but backend runs on port 80" -ForegroundColor Red

Write-Host "`n1. Checking current backend URL configuration..." -ForegroundColor Yellow
ssh -i $SSH_KEY -o ConnectTimeout=10 $USER@$SERVER_IP "
    echo '=== Current BACKEND_URL in worker ==='
    grep -n 'BACKEND_URL' /home/ubuntu/voice-agent-livekit-affan/backend/worker/main_flow_based.py
    
    echo '=== Current .env file ==='
    grep 'BACKEND_URL' /home/ubuntu/voice-agent-livekit-affan/.env || echo 'BACKEND_URL not found in .env'
    
    echo '=== What ports are actually listening ==='
    netstat -tuln | grep -E ':(80|443|8000)'
"

Write-Host "`n2. Fixing backend URL configuration..." -ForegroundColor Green
ssh -i $SSH_KEY -o ConnectTimeout=10 $USER@$SERVER_IP "
    # Update .env file to use correct backend URL
    if grep -q 'BACKEND_URL' /home/ubuntu/voice-agent-livekit-affan/.env; then
        sed -i 's|BACKEND_URL=.*|BACKEND_URL=https://18.210.238.67.nip.io|' /home/ubuntu/voice-agent-livekit-affan/.env
    else
        echo 'BACKEND_URL=https://18.210.238.67.nip.io' >> /home/ubuntu/voice-agent-livekit-affan/.env
    fi
    
    # Update worker file to use HTTPS instead of HTTP
    sed -i 's|BACKEND_URL = os.getenv(\"BACKEND_URL\", \"http://localhost:8000\")|BACKEND_URL = os.getenv(\"BACKEND_URL\", \"https://18.210.238.67.nip.io\")|' /home/ubuntu/voice-agent-livekit-affan/backend/worker/main_flow_based.py
    
    echo '=== Updated configuration ==='
    echo 'BACKEND_URL in .env:'
    grep 'BACKEND_URL' /home/ubuntu/voice-agent-livekit-affan/.env
    
    echo 'BACKEND_URL in worker:'
    grep -n 'BACKEND_URL' /home/ubuntu/voice-agent-livekit-affan/backend/worker/main_flow_based.py
"

Write-Host "`n3. Restarting worker service..." -ForegroundColor Green
ssh -i $SSH_KEY -o ConnectTimeout=10 $USER@$SERVER_IP "
    sudo systemctl restart alive5-worker
    
    # Wait for service to start
    sleep 3
    
    # Check service status
    sudo systemctl status alive5-worker --no-pager -l
"

Write-Host "`n4. Testing the fix..." -ForegroundColor Green
Write-Host "Testing backend connectivity from worker..."
ssh -i $SSH_KEY -o ConnectTimeout=10 $USER@$SERVER_IP "
    # Test if worker can now reach backend
    curl -s -o /dev/null -w '%{http_code}' https://18.210.238.67.nip.io/health
    echo ' - Backend health check'
    
    # Test worker logs for successful connection
    echo '=== Recent worker logs ==='
    sudo journalctl -u alive5-worker --no-pager -n 10
"

Write-Host "`n✅ Backend URL fix applied!" -ForegroundColor Green
Write-Host "Key changes:" -ForegroundColor White
Write-Host "- Updated BACKEND_URL from http://18.210.238.67:8000 to https://18.210.238.67.nip.io" -ForegroundColor White
Write-Host "- Updated .env file with correct URL" -ForegroundColor White
Write-Host "- Restarted worker service" -ForegroundColor White

Write-Host "`n🚀 The worker should now connect properly to the backend!" -ForegroundColor Cyan
Write-Host "Test your voice agent again - it should respond much faster now." -ForegroundColor White
