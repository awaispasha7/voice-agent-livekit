#!/usr/bin/env pwsh

Write-Host "🔍 Performance Diagnostic Tool" -ForegroundColor Cyan
Write-Host "=================================" -ForegroundColor Cyan

# Server details
$SERVER_IP = "18.210.238.67"
$SSH_KEY = "alive5-voice-ai-agent.pem"
$USER = "ubuntu"

Write-Host "`n1. Checking server resources..." -ForegroundColor Yellow
ssh -i $SSH_KEY -o ConnectTimeout=10 $USER@$SERVER_IP "
    echo '=== CPU Usage ==='
    top -bn1 | grep 'Cpu(s)' | head -1
    echo '=== Memory Usage ==='
    free -h
    echo '=== Disk Usage ==='
    df -h /
    echo '=== Load Average ==='
    uptime
    echo '=== Network Connections ==='
    netstat -tuln | grep -E ':(80|443|8000)'
"

Write-Host "`n2. Checking service status..." -ForegroundColor Yellow
ssh -i $SSH_KEY -o ConnectTimeout=10 $USER@$SERVER_IP "
    echo '=== Backend Service ==='
    sudo systemctl status alive5-backend --no-pager -l
    echo '=== Worker Service ==='
    sudo systemctl status alive5-worker --no-pager -l
    echo '=== Nginx Service ==='
    sudo systemctl status nginx --no-pager -l
"

Write-Host "`n3. Testing API response times..." -ForegroundColor Yellow
Write-Host "Testing health endpoint..."
$healthStart = Get-Date
try {
    $healthResponse = Invoke-WebRequest -Uri "https://18.210.238.67.nip.io/health" -TimeoutSec 30
    $healthEnd = Get-Date
    $healthTime = ($healthEnd - $healthStart).TotalMilliseconds
    Write-Host "✅ Health endpoint: $($healthTime.ToString('F0'))ms" -ForegroundColor Green
} catch {
    Write-Host "❌ Health endpoint failed: $($_.Exception.Message)" -ForegroundColor Red
}

Write-Host "`nTesting template status endpoint..."
$templateStart = Get-Date
try {
    $templateResponse = Invoke-WebRequest -Uri "https://18.210.238.67.nip.io/api/template_status" -TimeoutSec 30
    $templateEnd = Get-Date
    $templateTime = ($templateEnd - $templateStart).TotalMilliseconds
    Write-Host "✅ Template status: $($templateTime.ToString('F0'))ms" -ForegroundColor Green
} catch {
    Write-Host "❌ Template status failed: $($_.Exception.Message)" -ForegroundColor Red
}

Write-Host "`n4. Checking recent logs for errors..." -ForegroundColor Yellow
ssh -i $SSH_KEY -o ConnectTimeout=10 $USER@$SERVER_IP "
    echo '=== Backend Logs (last 20 lines) ==='
    sudo journalctl -u alive5-backend --no-pager -n 20
    echo '=== Worker Logs (last 20 lines) ==='
    sudo journalctl -u alive5-worker --no-pager -n 20
"

Write-Host "`n5. Checking network latency..." -ForegroundColor Yellow
ssh -i $SSH_KEY -o ConnectTimeout=10 $USER@$SERVER_IP "
    echo '=== Internal network test ==='
    curl -w '@-' -o /dev/null -s 'https://18.210.238.67.nip.io/health' <<< '
    time_namelookup:  %{time_namelookup}s
    time_connect:     %{time_connect}s
    time_appconnect:  %{time_appconnect}s
    time_pretransfer: %{time_pretransfer}s
    time_redirect:    %{time_redirect}s
    time_starttransfer: %{time_starttransfer}s
    time_total:       %{time_total}s
    '
"

Write-Host "`n6. Checking Python process resources..." -ForegroundColor Yellow
ssh -i $SSH_KEY -o ConnectTimeout=10 $USER@$SERVER_IP "
    echo '=== Python Processes ==='
    ps aux | grep python | grep -v grep
    echo '=== Process Memory Usage ==='
    ps -eo pid,ppid,cmd,%mem,%cpu --sort=-%mem | head -10
"

Write-Host "`n🔍 Performance Diagnostic Complete!" -ForegroundColor Cyan
Write-Host "Review the output above for performance bottlenecks." -ForegroundColor White
