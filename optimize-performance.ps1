#!/usr/bin/env pwsh

Write-Host "🚀 Performance Optimization Script" -ForegroundColor Cyan
Write-Host "===================================" -ForegroundColor Cyan

# Server details
$SERVER_IP = "18.210.238.67"
$SSH_KEY = "alive5-voice-ai-agent.pem"
$USER = "ubuntu"

Write-Host "`n🔧 Optimizing server performance..." -ForegroundColor Yellow

# 1. Optimize system settings
Write-Host "1. Optimizing system settings..." -ForegroundColor Green
ssh -i $SSH_KEY -o ConnectTimeout=10 $USER@$SERVER_IP "
    # Increase file descriptor limits
    echo '* soft nofile 65536' | sudo tee -a /etc/security/limits.conf
    echo '* hard nofile 65536' | sudo tee -a /etc/security/limits.conf
    
    # Optimize network settings
    echo 'net.core.rmem_max = 16777216' | sudo tee -a /etc/sysctl.conf
    echo 'net.core.wmem_max = 16777216' | sudo tee -a /etc/sysctl.conf
    echo 'net.ipv4.tcp_rmem = 4096 87380 16777216' | sudo tee -a /etc/sysctl.conf
    echo 'net.ipv4.tcp_wmem = 4096 65536 16777216' | sudo tee -a /etc/sysctl.conf
    echo 'net.core.netdev_max_backlog = 5000' | sudo tee -a /etc/sysctl.conf
    
    # Apply settings
    sudo sysctl -p
"

# 2. Optimize Python/uvicorn settings
Write-Host "2. Optimizing Python application settings..." -ForegroundColor Green
ssh -i $SSH_KEY -o ConnectTimeout=10 $USER@$SERVER_IP "
    # Create optimized systemd service files
    sudo tee /etc/systemd/system/alive5-backend.service > /dev/null << 'EOF'
[Unit]
Description=Alive5 Voice Agent Backend
After=network.target

[Service]
Type=exec
User=root
Group=root
WorkingDirectory=/home/ubuntu/voice-agent-livekit-affan
Environment=PATH=/home/ubuntu/voice-agent-livekit-affan/venv/bin
ExecStart=/home/ubuntu/voice-agent-livekit-affan/venv/bin/python backend/main_dynamic.py
Restart=always
RestartSec=5
StandardOutput=journal
StandardError=journal

# Performance optimizations
LimitNOFILE=65536
LimitNPROC=4096

# Memory and CPU limits
MemoryMax=2G
CPUQuota=200%

[Install]
WantedBy=multi-user.target
EOF

    sudo tee /etc/systemd/system/alive5-worker.service > /dev/null << 'EOF'
[Unit]
Description=Alive5 Voice Agent Worker
After=network.target

[Service]
Type=exec
User=ubuntu
Group=ubuntu
WorkingDirectory=/home/ubuntu/voice-agent-livekit-affan
Environment=PATH=/home/ubuntu/voice-agent-livekit-affan/venv/bin
ExecStart=/home/ubuntu/voice-agent-livekit-affan/venv/bin/python backend/worker/main_flow_based.py start
Restart=always
RestartSec=5
StandardOutput=journal
StandardError=journal

# Performance optimizations
LimitNOFILE=65536
LimitNPROC=4096

# Memory and CPU limits
MemoryMax=1G
CPUQuota=150%

[Install]
WantedBy=multi-user.target
EOF

    # Reload systemd and restart services
    sudo systemctl daemon-reload
"

# 3. Optimize Nginx configuration
Write-Host "3. Optimizing Nginx configuration..." -ForegroundColor Green
ssh -i $SSH_KEY -o ConnectTimeout=10 $USER@$SERVER_IP "
    sudo tee /etc/nginx/sites-available/fastapi > /dev/null << 'EOF'
server {
    listen 80;
    listen 443 ssl http2;
    server_name 18.210.238.67.nip.io;

    # SSL configuration
    ssl_certificate /etc/letsencrypt/live/18.210.238.67.nip.io/fullchain.pem;
    ssl_certificate_key /etc/letsencrypt/live/18.210.238.67.nip.io/privkey.pem;
    ssl_protocols TLSv1.2 TLSv1.3;
    ssl_ciphers ECDHE-RSA-AES256-GCM-SHA512:DHE-RSA-AES256-GCM-SHA512:ECDHE-RSA-AES256-GCM-SHA384:DHE-RSA-AES256-GCM-SHA384;
    ssl_prefer_server_ciphers off;

    # Performance optimizations
    client_max_body_size 10M;
    client_body_timeout 60s;
    client_header_timeout 60s;
    keepalive_timeout 65;
    keepalive_requests 1000;
    
    # Gzip compression
    gzip on;
    gzip_vary on;
    gzip_min_length 1024;
    gzip_types text/plain text/css text/xml text/javascript application/javascript application/xml+rss application/json;

    # Proxy settings
    proxy_connect_timeout 30s;
    proxy_send_timeout 30s;
    proxy_read_timeout 30s;
    proxy_buffering on;
    proxy_buffer_size 4k;
    proxy_buffers 8 4k;
    proxy_busy_buffers_size 8k;

    location / {
        proxy_pass http://127.0.0.1:8000;
        proxy_set_header Host \$host;
        proxy_set_header X-Real-IP \$remote_addr;
        proxy_set_header X-Forwarded-For \$proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto \$scheme;
        
        # WebSocket support
        proxy_http_version 1.1;
        proxy_set_header Upgrade \$http_upgrade;
        proxy_set_header Connection \"upgrade\";
    }
}
EOF

    # Test and reload Nginx
    sudo nginx -t && sudo systemctl reload nginx
"

# 4. Create optimized backend configuration
Write-Host "4. Creating optimized backend configuration..." -ForegroundColor Green
ssh -i $SSH_KEY -o ConnectTimeout=10 $USER@$SERVER_IP "
    # Create optimized startup script
    cat > /home/ubuntu/voice-agent-livekit-affan/start_backend_optimized.py << 'EOF'
import uvicorn
import os

if __name__ == '__main__':
    # Optimized uvicorn settings for production
    uvicorn.run(
        'backend.main_dynamic:app',
        host='127.0.0.1',
        port=8000,
        workers=1,  # Single worker for now
        loop='asyncio',
        http='httptools',
        access_log=False,  # Disable access logs for performance
        log_level='info',
        timeout_keep_alive=30,
        timeout_graceful_shutdown=30,
        limit_concurrency=1000,
        limit_max_requests=1000,
        backlog=2048
    )
EOF

    # Update the main backend file to use optimized settings
    sed -i 's/uvicorn.run(app, host=\"0.0.0.0\", port=8000)/# Optimized startup moved to start_backend_optimized.py/' /home/ubuntu/voice-agent-livekit-affan/backend/main_dynamic.py
"

# 5. Restart services with optimized settings
Write-Host "5. Restarting services with optimized settings..." -ForegroundColor Green
ssh -i $SSH_KEY -o ConnectTimeout=10 $USER@$SERVER_IP "
    # Stop services
    sudo systemctl stop alive5-backend
    sudo systemctl stop alive5-worker
    
    # Update backend service to use optimized startup
    sudo sed -i 's|ExecStart=.*|ExecStart=/home/ubuntu/voice-agent-livekit-affan/venv/bin/python start_backend_optimized.py|' /etc/systemd/system/alive5-backend.service
    
    # Reload and start services
    sudo systemctl daemon-reload
    sudo systemctl start alive5-backend
    sudo systemctl start alive5-worker
    
    # Enable services
    sudo systemctl enable alive5-backend
    sudo systemctl enable alive5-worker
"

# 6. Verify optimization
Write-Host "6. Verifying optimization..." -ForegroundColor Green
ssh -i $SSH_KEY -o ConnectTimeout=10 $USER@$SERVER_IP "
    echo '=== Service Status ==='
    sudo systemctl status alive5-backend --no-pager -l
    sudo systemctl status alive5-worker --no-pager -l
    
    echo '=== Recent Logs ==='
    sudo journalctl -u alive5-backend --no-pager -n 5
    sudo journalctl -u alive5-worker --no-pager -n 5
"

Write-Host "`n✅ Performance optimization complete!" -ForegroundColor Green
Write-Host "Key optimizations applied:" -ForegroundColor White
Write-Host "- Increased file descriptor limits" -ForegroundColor White
Write-Host "- Optimized network buffer sizes" -ForegroundColor White
Write-Host "- Enhanced Nginx proxy settings" -ForegroundColor White
Write-Host "- Optimized uvicorn configuration" -ForegroundColor White
Write-Host "- Added memory and CPU limits" -ForegroundColor White
Write-Host "- Enabled gzip compression" -ForegroundColor White

Write-Host "`n🔍 Run .\diagnose-performance.ps1 to verify improvements" -ForegroundColor Cyan
