# View worker service logs
Write-Host "Alive5 Worker Logs" -ForegroundColor Green
Write-Host "==================" -ForegroundColor Green
Write-Host "Press Ctrl+C to exit" -ForegroundColor Yellow
Write-Host ""

ssh -i alive5-voice-ai-agent.pem -o ConnectTimeout=10 ubuntu@18.210.238.67 "sudo journalctl -u alive5-worker -f --no-pager"
