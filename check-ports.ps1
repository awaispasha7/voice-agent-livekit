# Check Port Status and Network Connectivity
Write-Host "Port Status and Network Connectivity Check" -ForegroundColor Green
Write-Host "=========================================" -ForegroundColor Green

$serverIP = "18.210.238.67"

Write-Host ""
Write-Host "Checking server connectivity..." -ForegroundColor Cyan
ssh -i alive5-voice-ai-agent.pem -o ConnectTimeout=10 ubuntu@$serverIP "echo 'SSH connection: OK'"

Write-Host ""
Write-Host "Checking which ports are listening on server..." -ForegroundColor Cyan
ssh -i alive5-voice-ai-agent.pem -o ConnectTimeout=30 ubuntu@$serverIP "sudo ss -tlnp | grep -E ':(80|443|8000)'"

Write-Host ""
Write-Host "Checking firewall status..." -ForegroundColor Cyan
ssh -i alive5-voice-ai-agent.pem -o ConnectTimeout=30 ubuntu@$serverIP "sudo ufw status"

Write-Host ""
Write-Host "Testing local HTTP access..." -ForegroundColor Cyan
ssh -i alive5-voice-ai-agent.pem -o ConnectTimeout=30 ubuntu@$serverIP "curl -s http://localhost/health || echo 'Local HTTP failed'"

Write-Host ""
Write-Host "Testing local HTTPS access..." -ForegroundColor Cyan
ssh -i alive5-voice-ai-agent.pem -o ConnectTimeout=30 ubuntu@$serverIP "curl -s https://localhost/health || echo 'Local HTTPS failed'"

Write-Host ""
Write-Host "Testing external HTTP access..." -ForegroundColor Cyan
try {
    $httpResponse = Invoke-RestMethod -Uri "http://$serverIP/health" -TimeoutSec 10
    Write-Host "External HTTP: PASS" -ForegroundColor Green
    Write-Host "Response: $httpResponse" -ForegroundColor White
} catch {
    Write-Host "External HTTP: FAIL" -ForegroundColor Red
    Write-Host "Error: $($_.Exception.Message)" -ForegroundColor White
}

Write-Host ""
Write-Host "Testing external HTTPS access..." -ForegroundColor Cyan
try {
    $httpsResponse = Invoke-RestMethod -Uri "https://$serverIP.nip.io/health" -TimeoutSec 10
    Write-Host "External HTTPS: PASS" -ForegroundColor Green
    Write-Host "Response: $httpsResponse" -ForegroundColor White
} catch {
    Write-Host "External HTTPS: FAIL" -ForegroundColor Red
    Write-Host "Error: $($_.Exception.Message)" -ForegroundColor White
}

Write-Host ""
Write-Host "Checking server's public IP..." -ForegroundColor Cyan
ssh -i alive5-voice-ai-agent.pem -o ConnectTimeout=30 ubuntu@$serverIP "curl -s ifconfig.me || curl -s ipinfo.io/ip"

Write-Host ""
Write-Host "Checking DNS resolution..." -ForegroundColor Cyan
try {
    $dnsResult = Resolve-DnsName "$serverIP.nip.io" -ErrorAction Stop
    Write-Host "DNS Resolution: PASS" -ForegroundColor Green
    Write-Host "Resolved to: $($dnsResult.IPAddress)" -ForegroundColor White
} catch {
    Write-Host "DNS Resolution: FAIL" -ForegroundColor Red
    Write-Host "Error: $($_.Exception.Message)" -ForegroundColor White
}

Write-Host ""
Write-Host "Service Status Check..." -ForegroundColor Cyan
ssh -i alive5-voice-ai-agent.pem -o ConnectTimeout=30 ubuntu@$serverIP "sudo systemctl status alive5-backend --no-pager -l | head -10"
ssh -i alive5-voice-ai-agent.pem -o ConnectTimeout=30 ubuntu@$serverIP "sudo systemctl status nginx --no-pager -l | head -10"

Write-Host ""
Write-Host "Summary:" -ForegroundColor Yellow
Write-Host "- If External HTTP works but HTTPS fails: Port 443 is blocked" -ForegroundColor White
Write-Host "- If both fail: Check AWS Security Group for both ports" -ForegroundColor White
Write-Host "- If local works but external fails: AWS Security Group issue" -ForegroundColor White
Write-Host "- If DNS fails: Network connectivity issue" -ForegroundColor White
