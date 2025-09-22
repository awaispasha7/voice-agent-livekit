# Configure everything for port 80 only
Write-Host "Configuring for Port 80 Only" -ForegroundColor Green
Write-Host "============================" -ForegroundColor Green

$nipDomain = "18.210.238.67.nip.io"

# 1. Stop all services
Write-Host "1. Stopping all services..." -ForegroundColor Cyan
ssh -i alive5-voice-ai-agent.pem -o ConnectTimeout=30 ubuntu@18.210.238.67 "sudo systemctl stop alive5-backend alive5-worker nginx"

# 2. Create simple Nginx configuration for port 80 only
Write-Host "2. Creating Nginx configuration for port 80..." -ForegroundColor Cyan
$nginxConfig = @"
server {
    listen 80;
    server_name $nipDomain;

    location / {
        proxy_pass http://127.0.0.1:8000;
        proxy_set_header Host `$host;
        proxy_set_header X-Real-IP `$remote_addr;
        proxy_set_header X-Forwarded-For `$proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto `$scheme;
    }
}
"@

echo $nginxConfig | ssh -i alive5-voice-ai-agent.pem -o ConnectTimeout=30 ubuntu@18.210.238.67 "sudo tee /etc/nginx/sites-enabled/fastapi > /dev/null"

# 3. Remove default site
Write-Host "3. Removing default Nginx site..." -ForegroundColor Cyan
ssh -i alive5-voice-ai-agent.pem -o ConnectTimeout=30 ubuntu@18.210.238.67 "sudo rm -f /etc/nginx/sites-enabled/default"

# 4. Test and start Nginx
Write-Host "4. Testing and starting Nginx..." -ForegroundColor Cyan
ssh -i alive5-voice-ai-agent.pem -o ConnectTimeout=30 ubuntu@18.210.238.67 "sudo nginx -t && sudo systemctl start nginx"

# 5. Start backend services
Write-Host "5. Starting backend services..." -ForegroundColor Cyan
ssh -i alive5-voice-ai-agent.pem -o ConnectTimeout=30 ubuntu@18.210.238.67 "sudo systemctl start alive5-backend alive5-worker"

# 6. Test the setup
Write-Host "6. Testing the setup..." -ForegroundColor Cyan
ssh -i alive5-voice-ai-agent.pem -o ConnectTimeout=30 ubuntu@18.210.238.67 "curl -s http://$nipDomain/health"

Write-Host ""
Write-Host "✅ Port 80 Configuration Complete!" -ForegroundColor Green
Write-Host "Your backend is now available at:" -ForegroundColor Yellow
Write-Host "  http://$nipDomain" -ForegroundColor White
Write-Host "  http://$nipDomain/health" -ForegroundColor White
Write-Host "  http://$nipDomain/api/connection_details" -ForegroundColor White
Write-Host ""
Write-Host "Frontend will use HTTP, avoiding Mixed Content issues!" -ForegroundColor Green
