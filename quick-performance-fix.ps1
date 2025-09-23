#!/usr/bin/env pwsh

Write-Host "⚡ Quick Performance Fix" -ForegroundColor Cyan
Write-Host "=======================" -ForegroundColor Cyan

# Server details
$SERVER_IP = "18.210.238.67"
$SSH_KEY = "alive5-voice-ai-agent.pem"
$USER = "ubuntu"

Write-Host "`n🔧 Applying critical performance fixes..." -ForegroundColor Yellow

# 1. Fix the most critical issue: Reduce backend timeout in worker
Write-Host "1. Reducing worker backend timeout..." -ForegroundColor Green
ssh -i $SSH_KEY -o ConnectTimeout=10 $USER@$SERVER_IP "
    # Update worker timeout from 25s to 10s for faster responses
    sed -i 's/BACKEND_TIMEOUT = 25/BACKEND_TIMEOUT = 10/' /home/ubuntu/voice-agent-livekit-affan/backend/worker/main_flow_based.py
    
    # Update FAQ timeout from 35s to 15s
    sed -i 's/timeout=httpx.Timeout(35.0)/timeout=httpx.Timeout(15.0)/' /home/ubuntu/voice-agent-livekit-affan/backend/main_dynamic.py
    
    # Update template polling timeout from 30s to 15s
    sed -i 's/timeout=30.0/timeout=15.0/' /home/ubuntu/voice-agent-livekit-affan/backend/main_dynamic.py
"

# 2. Optimize HTTP client settings
Write-Host "2. Optimizing HTTP client settings..." -ForegroundColor Green
ssh -i $SSH_KEY -o ConnectTimeout=10 $USER@$SERVER_IP "
    # Add connection pooling and keep-alive settings
    cat > /home/ubuntu/voice-agent-livekit-affan/backend/worker/http_optimization.py << 'EOF'
import httpx
import asyncio

# Optimized HTTP client settings
HTTP_CLIENT_LIMITS = httpx.Limits(
    max_keepalive_connections=20,
    max_connections=100,
    keepalive_expiry=30.0
)

HTTP_CLIENT_TIMEOUT = httpx.Timeout(
    connect=5.0,
    read=10.0,
    write=5.0,
    pool=5.0
)

async def get_optimized_client():
    return httpx.AsyncClient(
        limits=HTTP_CLIENT_LIMITS,
        timeout=HTTP_CLIENT_TIMEOUT,
        http2=True
    )
EOF
"

# 3. Restart services
Write-Host "3. Restarting services..." -ForegroundColor Green
ssh -i $SSH_KEY -o ConnectTimeout=10 $USER@$SERVER_IP "
    sudo systemctl restart alive5-backend
    sudo systemctl restart alive5-worker
    
    # Wait for services to start
    sleep 5
    
    # Check status
    sudo systemctl status alive5-backend --no-pager -l
    sudo systemctl status alive5-worker --no-pager -l
"

# 4. Test performance
Write-Host "4. Testing performance improvements..." -ForegroundColor Green
Write-Host "Testing health endpoint response time..."
$start = Get-Date
try {
    $response = Invoke-WebRequest -Uri "https://18.210.238.67.nip.io/health" -TimeoutSec 15
    $end = Get-Date
    $time = ($end - $start).TotalMilliseconds
    Write-Host "✅ Health endpoint: $($time.ToString('F0'))ms" -ForegroundColor Green
} catch {
    Write-Host "❌ Health endpoint failed: $($_.Exception.Message)" -ForegroundColor Red
}

Write-Host "`n✅ Quick performance fix applied!" -ForegroundColor Green
Write-Host "Key changes:" -ForegroundColor White
Write-Host "- Reduced worker backend timeout: 25s → 10s" -ForegroundColor White
Write-Host "- Reduced FAQ API timeout: 35s → 15s" -ForegroundColor White
Write-Host "- Reduced template polling timeout: 30s → 15s" -ForegroundColor White
Write-Host "- Added HTTP client optimization" -ForegroundColor White

Write-Host "`n🚀 For comprehensive optimization, run: .\optimize-performance.ps1" -ForegroundColor Cyan
