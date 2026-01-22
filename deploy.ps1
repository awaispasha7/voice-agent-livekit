#!/usr/bin/env pwsh

# Alive5 Voice Agent Deployment Script
# Unified deployment script with user selection

param(
    [switch]$Force
)

# Set UTF-8 encoding for emoji support
[Console]::OutputEncoding = [System.Text.Encoding]::UTF8
$OutputEncoding = [System.Text.Encoding]::UTF8

# Colors for output
$ErrorColor = "Red"
$SuccessColor = "Green"
$InfoColor = "Cyan"
$WarningColor = "Yellow"

Write-Host "================================================================================" -ForegroundColor $InfoColor
Write-Host "Alive5 Voice Agent Deployment" -ForegroundColor $InfoColor
Write-Host "================================================================================" -ForegroundColor $InfoColor

# Check if key file exists
if (-not (Test-Path "alive5-voice-ai-agent.pem")) {
    Write-Host "Error: alive5-voice-ai-agent.pem not found!" -ForegroundColor $ErrorColor
    Write-Host "Please ensure the SSH key is in the current directory." -ForegroundColor $ErrorColor
    return
}

# Check if .env file exists
if (-not (Test-Path ".env")) {
    Write-Host "Error: .env file not found!" -ForegroundColor $ErrorColor
    Write-Host "Please ensure the .env file is in the current directory." -ForegroundColor $ErrorColor
    return
}

Write-Host ".env file found - will be deployed to server" -ForegroundColor $SuccessColor

# Server details
$SERVER = "18.210.238.67"
$USER = "ubuntu"
$KEY = "alive5-voice-ai-agent.pem"

# -----------------------------------------------------------------------------
# Nginx helper: ensure /rtc is proxied to LiveKit and avoid conflicting vhosts
# -----------------------------------------------------------------------------
function Set-NginxRtcProxy {
    param(
        [string]$SshTarget,
        [string]$KeyPath,
        [string]$Domain
    )

    Write-Host "Ensuring nginx /rtc proxy (LiveKit) and removing conflicting vhosts..." -ForegroundColor $InfoColor

    # Single-quoted here-string to prevent PowerShell from expanding $host/$scheme/etc
    $remoteScript = @'
set -e

# If nginx isn't installed, skip (some environments may not have it)
if ! command -v nginx >/dev/null 2>&1; then
  echo "nginx not installed - skipping nginx /rtc proxy setup"
  exit 0
fi

DOMAIN="{{DOMAIN}}"

sudo python3 << 'PY'
import os

domain = os.environ.get("DOMAIN", "")
if not domain:
    raise SystemExit("DOMAIN env var missing")

cfg = f"""server {{
    listen 443 ssl http2;
    server_name {domain};

    # LiveKit WebSocket + HTTP validate endpoints
    # Keep path as-is: /rtc, /rtc/v1, /rtc/v1/validate, etc.
    location = /rtc {{
        proxy_pass http://127.0.0.1:7880;
        proxy_http_version 1.1;
        proxy_set_header Upgrade $http_upgrade;
        proxy_set_header Connection "upgrade";
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
        proxy_read_timeout 86400;
        proxy_connect_timeout 60s;
        proxy_send_timeout 60s;
    }}

    location ^~ /rtc/ {{
        proxy_pass http://127.0.0.1:7880;
        proxy_http_version 1.1;
        proxy_set_header Upgrade $http_upgrade;
        proxy_set_header Connection "upgrade";
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
        proxy_read_timeout 86400;
        proxy_connect_timeout 60s;
        proxy_send_timeout 60s;
    }}

    # Backend API
    location / {{
        proxy_pass http://127.0.0.1:8000;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
        proxy_connect_timeout 60s;
        proxy_send_timeout 60s;
        proxy_read_timeout 60s;
    }}

    ssl_certificate /etc/letsencrypt/live/{domain}/fullchain.pem;
    ssl_certificate_key /etc/letsencrypt/live/{domain}/privkey.pem;
    include /etc/letsencrypt/options-ssl-nginx.conf;
    ssl_dhparam /etc/letsencrypt/ssl-dhparams.pem;
}}

server {{
    listen 80;
    server_name {domain};
    return 301 https://$host$request_uri;
}}
"""

with open("/tmp/fastapi-nginx.conf", "w", encoding="utf-8") as f:
    f.write(cfg)
print("Wrote /tmp/fastapi-nginx.conf")
PY

# Publish config as the single vhost for this domain
sudo mv /tmp/fastapi-nginx.conf /etc/nginx/sites-available/fastapi

# Remove conflicting vhost symlink if present (causes nginx to ignore one of them)
sudo rm -f /etc/nginx/sites-enabled/alive5-voice-agent || true

# Ensure fastapi site is enabled
sudo ln -sf /etc/nginx/sites-available/fastapi /etc/nginx/sites-enabled/fastapi

sudo nginx -t
sudo systemctl reload nginx
echo "nginx /rtc proxy ensured"
'@

    $remoteScript = $remoteScript.Replace("{{DOMAIN}}", $Domain)

    # Stream the script over SSH stdin to avoid brittle quoting across PowerShell versions
    $remoteScript | & ssh -i $KeyPath -o StrictHostKeyChecking=no $SshTarget "DOMAIN='$Domain' bash -s" 2>&1 | Out-Null
    if ($LASTEXITCODE -ne 0) {
        Write-Host "  ⚠️  Warning: nginx /rtc proxy setup failed (non-fatal). Please check nginx on the server." -ForegroundColor $WarningColor
    } else {
        Write-Host "  ✅ nginx /rtc proxy ensured" -ForegroundColor $SuccessColor
    }
}

# Deployment options
Write-Host ""
Write-Host "Select deployment option:" -ForegroundColor $InfoColor
Write-Host "1. Deploy worker only (alive5-worker directory)" -ForegroundColor White
Write-Host "2. Deploy full backend (backend + worker)" -ForegroundColor White
Write-Host "3. Deploy all with requirements (backend + worker + install dependencies)" -ForegroundColor White
Write-Host "4. Deploy LiveKit setup files (for self-hosted LiveKit)" -ForegroundColor White
Write-Host ""

do {
    $choice = Read-Host "Enter your choice (1, 2, 3, or 4)"
} while ($choice -notin @("1", "2", "3", "4"))

Write-Host ""

if ($choice -eq "1") {
    Write-Host "Deploying worker only..." -ForegroundColor $InfoColor
    
    # Create directory on server
    Write-Host "  - Creating directory structure..." -ForegroundColor White
    $mkdirResult = & ssh -i $KEY -o StrictHostKeyChecking=no -o ConnectTimeout=10 -o ServerAliveInterval=5 "$USER@$SERVER" "mkdir -p /home/ubuntu/alive5-voice-agent/backend/alive5-worker" 2>&1
    if ($LASTEXITCODE -ne 0) {
        Write-Host "  ❌ Error creating directory structure. Exit code: $LASTEXITCODE" -ForegroundColor $ErrorColor
        Write-Host "  Error output: $mkdirResult" -ForegroundColor $ErrorColor
        return
    }
    
    # Create directory structure on server
    Write-Host "  - Creating directory structure..." -ForegroundColor White
    & ssh -i $KEY -o StrictHostKeyChecking=no "$USER@$SERVER" "mkdir -p /home/ubuntu/alive5-voice-agent/backend/alive5-worker /home/ubuntu/alive5-voice-agent/backend/agentcore" 2>&1 | Out-Null
    
    # Deploy worker files
    $scpTarget = "${USER}@${SERVER}:/home/ubuntu/alive5-voice-agent/backend/alive5-worker/"
    Write-Host "  - Deploying worker.py..." -ForegroundColor White
    & scp -i $KEY -o StrictHostKeyChecking=no alive5-backend/alive5-worker/worker.py $scpTarget
    
    Write-Host "  - Deploying functions.py..." -ForegroundColor White
    & scp -i $KEY -o StrictHostKeyChecking=no alive5-backend/alive5-worker/functions.py $scpTarget
    
    Write-Host "  - Deploying system_prompt.py..." -ForegroundColor White
    & scp -i $KEY -o StrictHostKeyChecking=no alive5-backend/alive5-worker/system_prompt.py $scpTarget
    
    Write-Host "  - Deploying tags_config.py..." -ForegroundColor White
    & scp -i $KEY -o StrictHostKeyChecking=no alive5-backend/alive5-worker/tags_config.py $scpTarget
    
    Write-Host "  - Deploying AgentCore integration files..." -ForegroundColor White
    & scp -i $KEY -o StrictHostKeyChecking=no alive5-backend/alive5-worker/agentcore_integration.py $scpTarget
    & scp -i $KEY -o StrictHostKeyChecking=no alive5-backend/alive5-worker/agentcore_llm_wrapper.py $scpTarget
    
    # Deploy AgentCore module files
    $agentcoreTarget = "${USER}@${SERVER}:/home/ubuntu/alive5-voice-agent/backend/agentcore/"
    Write-Host "  - Deploying AgentCore module files..." -ForegroundColor White
    & scp -i $KEY -o StrictHostKeyChecking=no alive5-backend/agentcore/__init__.py $agentcoreTarget
    & scp -i $KEY -o StrictHostKeyChecking=no alive5-backend/agentcore/agent.py $agentcoreTarget
    & scp -i $KEY -o StrictHostKeyChecking=no alive5-backend/agentcore/client.py $agentcoreTarget
    & scp -i $KEY -o StrictHostKeyChecking=no alive5-backend/agentcore/memory.py $agentcoreTarget
    & scp -i $KEY -o StrictHostKeyChecking=no alive5-backend/agentcore/gateway_tools.py $agentcoreTarget
    
    Write-Host "  - Deploying .env file..." -ForegroundColor White
    & scp -i $KEY -o StrictHostKeyChecking=no .env "${USER}@${SERVER}:/home/ubuntu/alive5-voice-agent/"
    
    Write-Host "  - Deploying requirements.txt..." -ForegroundColor White
    & scp -i $KEY -o StrictHostKeyChecking=no requirements.txt "${USER}@${SERVER}:/home/ubuntu/alive5-voice-agent/"
    
    Write-Host "Worker files deployed successfully!" -ForegroundColor $SuccessColor
    
    # Create/update worker service
    Write-Host "Creating/updating worker service..." -ForegroundColor $InfoColor
    
    # Build simplified service content with EnvironmentFile to load .env
    # Enhanced with better restart policies and resource limits
    $serviceLines = @(
        "[Unit]",
        "Description=Alive5 Voice Agent Worker",
        "After=network.target",
        "StartLimitIntervalSec=300",
        "StartLimitBurst=5",
        "",
        "[Service]",
        "EnvironmentFile=/home/ubuntu/alive5-voice-agent/.env",
        "Type=simple",
        "User=ubuntu",
        "WorkingDirectory=/home/ubuntu/alive5-voice-agent",
        "Environment=`"PATH=/home/ubuntu/alive5-voice-agent/venv/bin`"",
        "ExecStart=/home/ubuntu/alive5-voice-agent/venv/bin/python backend/alive5-worker/worker.py dev",
        "Restart=always",
        "RestartSec=10",
        "TimeoutStartSec=60",
        "TimeoutStopSec=30",
        "# Resource limits to prevent memory exhaustion",
        "# Uncomment and adjust if needed:",
        "# MemoryMax=1.5G",
        "# MemoryHigh=1.2G",
        "# CPUQuota=200%",
        "StandardOutput=journal",
        "StandardError=journal",
        "SyslogIdentifier=alive5-worker",
        "",
        "[Install]",
        "WantedBy=multi-user.target"
    )
    $serviceContent = $serviceLines -join "`n"
    
    # Write service file to server
    $sshTarget = "$USER@$SERVER"
    $tempFile = [System.IO.Path]::GetTempFileName()
    $serviceContent | Out-File -FilePath $tempFile -Encoding UTF8
    $scpTempTarget = "${sshTarget}:/tmp/service.tmp"
    & scp -i $KEY -o StrictHostKeyChecking=no $tempFile $scpTempTarget
    & ssh -i $KEY -o StrictHostKeyChecking=no $sshTarget "sudo mv /tmp/service.tmp /etc/systemd/system/alive5-worker.service"
    if (Test-Path $tempFile) { Remove-Item $tempFile }
    
    # Reload systemd and restart service
    & ssh -i $KEY -o StrictHostKeyChecking=no $sshTarget "sudo systemctl daemon-reload"
    & ssh -i $KEY -o StrictHostKeyChecking=no $sshTarget "sudo systemctl enable alive5-worker"
    & ssh -i $KEY -o StrictHostKeyChecking=no $sshTarget "sudo systemctl restart alive5-worker"
    
    Write-Host "Worker service updated and restarted!" -ForegroundColor $SuccessColor

    # Ensure nginx /rtc proxy so web sessions work after redeploys
    Set-NginxRtcProxy -SshTarget $sshTarget -KeyPath $KEY -Domain "18.210.238.67.nip.io"
    
} elseif ($choice -eq "2") {
    Write-Host "Deploying full backend..." -ForegroundColor $InfoColor
    
    # Create directories on server
    Write-Host "  - Creating directory structure..." -ForegroundColor White
    $mkdirResult = & ssh -i $KEY -o StrictHostKeyChecking=no -o ConnectTimeout=10 -o ServerAliveInterval=5 "$USER@$SERVER" "mkdir -p /home/ubuntu/alive5-voice-agent/backend/alive5-worker /home/ubuntu/alive5-voice-agent/backend/agentcore" 2>&1
    if ($LASTEXITCODE -ne 0) {
        Write-Host "  ❌ Error creating directory structure. Exit code: $LASTEXITCODE" -ForegroundColor $ErrorColor
        Write-Host "  Error output: $mkdirResult" -ForegroundColor $ErrorColor
        return
    }
    
    # Deploy backend files
    $backendTarget = "${USER}@${SERVER}:/home/ubuntu/alive5-voice-agent/backend/"
    Write-Host "  - Deploying backend/main.py..." -ForegroundColor White
    & scp -i $KEY -o StrictHostKeyChecking=no alive5-backend/main.py $backendTarget
    
    Write-Host "  - Deploying cached_voices.json..." -ForegroundColor White
    & scp -i $KEY -o StrictHostKeyChecking=no alive5-backend/cached_voices.json $backendTarget
    
    Write-Host "  - Deploying .env file..." -ForegroundColor White
    & scp -i $KEY -o StrictHostKeyChecking=no .env "${USER}@${SERVER}:/home/ubuntu/alive5-voice-agent/"
    
    Write-Host "  - Deploying requirements.txt..." -ForegroundColor White
    & scp -i $KEY -o StrictHostKeyChecking=no requirements.txt "${USER}@${SERVER}:/home/ubuntu/alive5-voice-agent/"
    
    Write-Host "Backend files deployed successfully!" -ForegroundColor $SuccessColor
    
    # Deploy worker files
    Write-Host "Deploying worker files..." -ForegroundColor $InfoColor
    
    $scpTarget = "${USER}@${SERVER}:/home/ubuntu/alive5-voice-agent/backend/alive5-worker/"
    Write-Host "  - Deploying worker.py..." -ForegroundColor White
    & scp -i $KEY -o StrictHostKeyChecking=no alive5-backend/alive5-worker/worker.py $scpTarget
    
    Write-Host "  - Deploying functions.py..." -ForegroundColor White
    & scp -i $KEY -o StrictHostKeyChecking=no alive5-backend/alive5-worker/functions.py $scpTarget
    
    Write-Host "  - Deploying system_prompt.py..." -ForegroundColor White
    & scp -i $KEY -o StrictHostKeyChecking=no alive5-backend/alive5-worker/system_prompt.py $scpTarget
    
    Write-Host "  - Deploying tags_config.py..." -ForegroundColor White
    & scp -i $KEY -o StrictHostKeyChecking=no alive5-backend/alive5-worker/tags_config.py $scpTarget
    
    Write-Host "  - Deploying AgentCore integration files..." -ForegroundColor White
    & scp -i $KEY -o StrictHostKeyChecking=no alive5-backend/alive5-worker/agentcore_integration.py $scpTarget
    & scp -i $KEY -o StrictHostKeyChecking=no alive5-backend/alive5-worker/agentcore_llm_wrapper.py $scpTarget
    
    # Deploy AgentCore module files
    $agentcoreTarget = "${USER}@${SERVER}:/home/ubuntu/alive5-voice-agent/backend/agentcore/"
    Write-Host "  - Deploying AgentCore module files..." -ForegroundColor White
    & scp -i $KEY -o StrictHostKeyChecking=no alive5-backend/agentcore/__init__.py $agentcoreTarget
    & scp -i $KEY -o StrictHostKeyChecking=no alive5-backend/agentcore/agent.py $agentcoreTarget
    & scp -i $KEY -o StrictHostKeyChecking=no alive5-backend/agentcore/client.py $agentcoreTarget
    & scp -i $KEY -o StrictHostKeyChecking=no alive5-backend/agentcore/memory.py $agentcoreTarget
    & scp -i $KEY -o StrictHostKeyChecking=no alive5-backend/agentcore/gateway_tools.py $agentcoreTarget
    
    Write-Host "Worker files deployed successfully!" -ForegroundColor $SuccessColor
    
    # Create backend service
    Write-Host "Creating backend service..." -ForegroundColor $InfoColor
    
    # Build simplified backend service content with EnvironmentFile to load .env
    $backendServiceLines = @(
        "[Unit]",
        "Description=Alive5 Voice Agent Backend",
        "After=network.target",
        "",
        "[Service]",
        "EnvironmentFile=/home/ubuntu/alive5-voice-agent/.env",
        "Type=simple",
        "User=ubuntu",
        "WorkingDirectory=/home/ubuntu/alive5-voice-agent",
        "Environment=`"PATH=/home/ubuntu/alive5-voice-agent/venv/bin`"",
        "ExecStart=/home/ubuntu/alive5-voice-agent/venv/bin/python backend/main.py",
        "Restart=always",
        "RestartSec=10",
        "",
        "[Install]",
        "WantedBy=multi-user.target"
    )
    $backendServiceContent = $backendServiceLines -join "`n"
    
    # Write backend service file to server
    $sshTarget = "$USER@$SERVER"
    $tempFile = [System.IO.Path]::GetTempFileName()
    $backendServiceContent | Out-File -FilePath $tempFile -Encoding UTF8
    $scpTempTarget = "${sshTarget}:/tmp/backend-service.tmp"
    & scp -i $KEY -o StrictHostKeyChecking=no $tempFile $scpTempTarget
    & ssh -i $KEY -o StrictHostKeyChecking=no $sshTarget "sudo mv /tmp/backend-service.tmp /etc/systemd/system/alive5-backend.service"
    if (Test-Path $tempFile) { Remove-Item $tempFile }
    
    # Create worker service
    Write-Host "Creating worker service..." -ForegroundColor $InfoColor
    
    # Build simplified worker service content with EnvironmentFile to load .env
    $workerServiceLines = @(
        "[Unit]",
        "Description=Alive5 Voice Agent Worker",
        "After=network.target",
        "",
        "[Service]",
        "EnvironmentFile=/home/ubuntu/alive5-voice-agent/.env",
        "Type=simple",
        "User=ubuntu",
        "WorkingDirectory=/home/ubuntu/alive5-voice-agent",
        "Environment=`"PATH=/home/ubuntu/alive5-voice-agent/venv/bin`"",
        "ExecStart=/home/ubuntu/alive5-voice-agent/venv/bin/python backend/alive5-worker/worker.py dev",
        "Restart=always",
        "RestartSec=10",
        "",
        "[Install]",
        "WantedBy=multi-user.target"
    )
    $workerServiceContent = $workerServiceLines -join "`n"
    
    # Write worker service file to server
    $sshTarget = "$USER@$SERVER"
    $tempFile = [System.IO.Path]::GetTempFileName()
    $workerServiceContent | Out-File -FilePath $tempFile -Encoding UTF8
    $scpTempTarget = "${sshTarget}:/tmp/worker-service.tmp"
    & scp -i $KEY -o StrictHostKeyChecking=no $tempFile $scpTempTarget
    & ssh -i $KEY -o StrictHostKeyChecking=no $sshTarget "sudo mv /tmp/worker-service.tmp /etc/systemd/system/alive5-worker.service"
    if (Test-Path $tempFile) { Remove-Item $tempFile }
    
    # Reload systemd and restart services
    & ssh -i $KEY -o StrictHostKeyChecking=no $sshTarget "sudo systemctl daemon-reload"
    & ssh -i $KEY -o StrictHostKeyChecking=no $sshTarget "sudo systemctl enable alive5-backend alive5-worker"
    & ssh -i $KEY -o StrictHostKeyChecking=no $sshTarget "sudo systemctl restart alive5-backend alive5-worker"
    
    Write-Host "All services updated and restarted!" -ForegroundColor $SuccessColor

    # Ensure nginx /rtc proxy so web sessions work after redeploys
    Set-NginxRtcProxy -SshTarget $sshTarget -KeyPath $KEY -Domain "18.210.238.67.nip.io"
    
} elseif ($choice -eq "3") {
    # Option 3: Deploy all with requirements
    Write-Host "Deploying all with requirements..." -ForegroundColor $InfoColor
    
    # Create directories on server
    Write-Host "  - Creating directory structure..." -ForegroundColor White
    $mkdirResult = & ssh -i $KEY -o StrictHostKeyChecking=no -o ConnectTimeout=10 -o ServerAliveInterval=5 "$USER@$SERVER" "mkdir -p /home/ubuntu/alive5-voice-agent/backend/alive5-worker" 2>&1
    if ($LASTEXITCODE -ne 0) {
        Write-Host "  ❌ Error creating directory structure. Exit code: $LASTEXITCODE" -ForegroundColor $ErrorColor
        Write-Host "  Error output: $mkdirResult" -ForegroundColor $ErrorColor
        return
    }
    
    # Deploy backend files
    $backendTarget = "${USER}@${SERVER}:/home/ubuntu/alive5-voice-agent/backend/"
    Write-Host "  - Deploying backend/main.py..." -ForegroundColor White
    & scp -i $KEY -o StrictHostKeyChecking=no alive5-backend/main.py $backendTarget
    
    Write-Host "  - Deploying cached_voices.json..." -ForegroundColor White
    & scp -i $KEY -o StrictHostKeyChecking=no alive5-backend/cached_voices.json $backendTarget
    
    Write-Host "  - Deploying .env file..." -ForegroundColor White
    & scp -i $KEY -o StrictHostKeyChecking=no .env "${USER}@${SERVER}:/home/ubuntu/alive5-voice-agent/"
    
    Write-Host "  - Deploying requirements.txt..." -ForegroundColor White
    & scp -i $KEY -o StrictHostKeyChecking=no requirements.txt "${USER}@${SERVER}:/home/ubuntu/alive5-voice-agent/"
    
    Write-Host "Backend files deployed successfully!" -ForegroundColor $SuccessColor
    
    # Create directory structure on server
    Write-Host "  - Creating directory structure..." -ForegroundColor White
    & ssh -i $KEY -o StrictHostKeyChecking=no "$USER@$SERVER" "mkdir -p /home/ubuntu/alive5-voice-agent/backend/alive5-worker /home/ubuntu/alive5-voice-agent/backend/agentcore" 2>&1 | Out-Null
    
    # Deploy worker files
    Write-Host "Deploying worker files..." -ForegroundColor $InfoColor
    
    $scpTarget = "${USER}@${SERVER}:/home/ubuntu/alive5-voice-agent/backend/alive5-worker/"
    Write-Host "  - Deploying worker.py..." -ForegroundColor White
    & scp -i $KEY -o StrictHostKeyChecking=no alive5-backend/alive5-worker/worker.py $scpTarget
    
    Write-Host "  - Deploying functions.py..." -ForegroundColor White
    & scp -i $KEY -o StrictHostKeyChecking=no alive5-backend/alive5-worker/functions.py $scpTarget
    
    Write-Host "  - Deploying system_prompt.py..." -ForegroundColor White
    & scp -i $KEY -o StrictHostKeyChecking=no alive5-backend/alive5-worker/system_prompt.py $scpTarget
    
    Write-Host "  - Deploying tags_config.py..." -ForegroundColor White
    & scp -i $KEY -o StrictHostKeyChecking=no alive5-backend/alive5-worker/tags_config.py $scpTarget
    
    Write-Host "  - Deploying AgentCore integration files..." -ForegroundColor White
    & scp -i $KEY -o StrictHostKeyChecking=no alive5-backend/alive5-worker/agentcore_integration.py $scpTarget
    & scp -i $KEY -o StrictHostKeyChecking=no alive5-backend/alive5-worker/agentcore_llm_wrapper.py $scpTarget
    
    # Deploy AgentCore module files
    $agentcoreTarget = "${USER}@${SERVER}:/home/ubuntu/alive5-voice-agent/backend/agentcore/"
    Write-Host "  - Deploying AgentCore module files..." -ForegroundColor White
    & scp -i $KEY -o StrictHostKeyChecking=no alive5-backend/agentcore/__init__.py $agentcoreTarget
    & scp -i $KEY -o StrictHostKeyChecking=no alive5-backend/agentcore/agent.py $agentcoreTarget
    & scp -i $KEY -o StrictHostKeyChecking=no alive5-backend/agentcore/client.py $agentcoreTarget
    & scp -i $KEY -o StrictHostKeyChecking=no alive5-backend/agentcore/memory.py $agentcoreTarget
    & scp -i $KEY -o StrictHostKeyChecking=no alive5-backend/agentcore/gateway_tools.py $agentcoreTarget
    
    Write-Host "Worker files deployed successfully!" -ForegroundColor $SuccessColor
    
    # Create virtual environment if it doesn't exist
    Write-Host "Checking virtual environment..." -ForegroundColor $InfoColor
    $venvCheck = & ssh -i $KEY -o StrictHostKeyChecking=no "$USER@$SERVER" "test -d /home/ubuntu/alive5-voice-agent/venv && echo 'exists' || echo 'missing'" 2>&1
    if ($venvCheck -match "missing") {
        Write-Host "  - Creating virtual environment..." -ForegroundColor White
        $venvCreate = & ssh -i $KEY -o StrictHostKeyChecking=no "$USER@$SERVER" "cd /home/ubuntu/alive5-voice-agent && python3 -m venv venv" 2>&1
        if ($LASTEXITCODE -ne 0) {
            Write-Host "  ❌ Error creating virtual environment:" -ForegroundColor $ErrorColor
            Write-Host $venvCreate -ForegroundColor $ErrorColor
            return
        }
        Write-Host "  ✅ Virtual environment created" -ForegroundColor $SuccessColor
    } else {
        Write-Host "  ✅ Virtual environment already exists" -ForegroundColor $SuccessColor
    }
    
    # Install requirements
    Write-Host "Installing requirements..." -ForegroundColor $InfoColor
    Write-Host "  - Updating pip..." -ForegroundColor White
    $installResult = & ssh -i $KEY -o StrictHostKeyChecking=no "$USER@$SERVER" "cd /home/ubuntu/alive5-voice-agent && /home/ubuntu/alive5-voice-agent/venv/bin/pip install --upgrade pip" 2>&1
    if ($LASTEXITCODE -ne 0) {
        Write-Host "  ⚠️ Warning: pip upgrade failed, continuing..." -ForegroundColor $WarningColor
    }
    
    Write-Host "  - Installing packages from requirements.txt..." -ForegroundColor White
    Write-Host "    (This may take 2-5 minutes, especially for livekit-plugins-aws...)" -ForegroundColor $WarningColor
    Write-Host "    Installing with verbose output (you'll see progress)..." -ForegroundColor White
    
    # Install with legacy resolver to avoid backtracking issues
    # The new resolver can take forever with complex dependency trees
    $installCmd = "cd /home/ubuntu/alive5-voice-agent && /home/ubuntu/alive5-voice-agent/venv/bin/pip install --no-cache-dir --timeout=300 --use-deprecated=legacy-resolver -r requirements.txt 2>&1 | tee /tmp/pip-install.log"
    Write-Host ""
    Write-Host "    Using legacy resolver to avoid dependency backtracking..." -ForegroundColor Gray
    
    # Run with unbuffered output for real-time streaming
    $installResult = & ssh -i $KEY -o StrictHostKeyChecking=no -o ServerAliveInterval=10 "$USER@$SERVER" "bash -c '$installCmd'" 2>&1 | ForEach-Object {
        Write-Host $_ -ForegroundColor Gray
        $_
    }
    
    if ($LASTEXITCODE -eq 0) {
        Write-Host "  ✅ Requirements installed successfully!" -ForegroundColor $SuccessColor
    } else {
        Write-Host "  ❌ Error installing requirements:" -ForegroundColor $ErrorColor
        Write-Host $installResult -ForegroundColor $ErrorColor
        
        # Try installing AWS plugin separately with legacy resolver
        Write-Host "  - Attempting to install livekit-plugins-aws separately..." -ForegroundColor $WarningColor
        Write-Host "    (Using legacy resolver to avoid backtracking)..." -ForegroundColor White
        $awsPluginResult = & ssh -i $KEY -o StrictHostKeyChecking=no -o ServerAliveInterval=10 "$USER@$SERVER" "bash -c 'cd /home/ubuntu/alive5-voice-agent && /home/ubuntu/alive5-voice-agent/venv/bin/pip install --no-cache-dir --timeout=300 --use-deprecated=legacy-resolver livekit-plugins-aws>=1.2.0,<2.0.0 2>&1'" 2>&1 | ForEach-Object {
            Write-Host $_ -ForegroundColor Gray
            $_
        }
        if ($LASTEXITCODE -eq 0) {
            Write-Host "  ✅ AWS plugin installed separately!" -ForegroundColor $SuccessColor
            Write-Host "  - Retrying full requirements install..." -ForegroundColor White
            $installResult = & ssh -i $KEY -o StrictHostKeyChecking=no "$USER@$SERVER" $installCmd 2>&1
            if ($LASTEXITCODE -eq 0) {
                Write-Host "  ✅ Requirements installed successfully on retry!" -ForegroundColor $SuccessColor
            } else {
                Write-Host "  ⚠️ Some packages may have failed, but continuing..." -ForegroundColor $WarningColor
            }
        } else {
            Write-Host "  ⚠️ AWS plugin installation also failed. Continuing with service setup..." -ForegroundColor $WarningColor
            Write-Host "  ⚠️ Note: Worker will fall back to OpenAI if Bedrock plugin is not available" -ForegroundColor $WarningColor
        }
    }
    
    # Download model files (turn detector, etc.) - must be after requirements are installed
    Write-Host "Downloading model files..." -ForegroundColor $InfoColor
    Write-Host "  - Running download-files command..." -ForegroundColor White
    Write-Host "    (This downloads turn detector model weights, ~66 MB for English or ~281 MB for multilingual)" -ForegroundColor Gray
    $downloadResult = & ssh -i $KEY -o StrictHostKeyChecking=no "$USER@$SERVER" "cd /home/ubuntu/alive5-voice-agent && /home/ubuntu/alive5-voice-agent/venv/bin/python backend/alive5-worker/worker.py download-files 2>&1" 2>&1 | ForEach-Object {
        Write-Host $_ -ForegroundColor Gray
        $_
    }
    if ($LASTEXITCODE -eq 0) {
        Write-Host "  ✅ Model files downloaded successfully!" -ForegroundColor $SuccessColor
    } else {
        Write-Host "  ⚠️ Warning: download-files failed, but continuing..." -ForegroundColor $WarningColor
        Write-Host "  ⚠️ You can manually run: python backend/alive5-worker/worker.py download-files" -ForegroundColor $WarningColor
    }
    
    # Create backend service
    Write-Host "Creating backend service..." -ForegroundColor $InfoColor
    
    # Build simplified backend service content with EnvironmentFile to load .env
    $backendServiceLines = @(
        "[Unit]",
        "Description=Alive5 Voice Agent Backend",
        "After=network.target",
        "",
        "[Service]",
        "EnvironmentFile=/home/ubuntu/alive5-voice-agent/.env",
        "Type=simple",
        "User=ubuntu",
        "WorkingDirectory=/home/ubuntu/alive5-voice-agent",
        "Environment=`"PATH=/home/ubuntu/alive5-voice-agent/venv/bin`"",
        "ExecStart=/home/ubuntu/alive5-voice-agent/venv/bin/python backend/main.py",
        "Restart=always",
        "RestartSec=10",
        "",
        "[Install]",
        "WantedBy=multi-user.target"
    )
    $backendServiceContent = $backendServiceLines -join "`n"
    
    # Write backend service file to server
    $sshTarget = "$USER@$SERVER"
    $tempFile = [System.IO.Path]::GetTempFileName()
    $backendServiceContent | Out-File -FilePath $tempFile -Encoding UTF8
    $scpTempTarget = "${sshTarget}:/tmp/backend-service.tmp"
    & scp -i $KEY -o StrictHostKeyChecking=no $tempFile $scpTempTarget
    & ssh -i $KEY -o StrictHostKeyChecking=no $sshTarget "sudo mv /tmp/backend-service.tmp /etc/systemd/system/alive5-backend.service"
    if (Test-Path $tempFile) { Remove-Item $tempFile }
    
    # Create worker service
    Write-Host "Creating worker service..." -ForegroundColor $InfoColor
    
    # Build simplified worker service content with EnvironmentFile to load .env
    $workerServiceLines = @(
        "[Unit]",
        "Description=Alive5 Voice Agent Worker",
        "After=network.target",
        "",
        "[Service]",
        "EnvironmentFile=/home/ubuntu/alive5-voice-agent/.env",
        "Type=simple",
        "User=ubuntu",
        "WorkingDirectory=/home/ubuntu/alive5-voice-agent",
        "Environment=`"PATH=/home/ubuntu/alive5-voice-agent/venv/bin`"",
        "ExecStart=/home/ubuntu/alive5-voice-agent/venv/bin/python backend/alive5-worker/worker.py dev",
        "Restart=always",
        "RestartSec=10",
        "",
        "[Install]",
        "WantedBy=multi-user.target"
    )
    $workerServiceContent = $workerServiceLines -join "`n"
    
    # Write worker service file to server
    $sshTarget = "$USER@$SERVER"
    $tempFile = [System.IO.Path]::GetTempFileName()
    $workerServiceContent | Out-File -FilePath $tempFile -Encoding UTF8
    $scpTempTarget = "${sshTarget}:/tmp/worker-service.tmp"
    & scp -i $KEY -o StrictHostKeyChecking=no $tempFile $scpTempTarget
    & ssh -i $KEY -o StrictHostKeyChecking=no $sshTarget "sudo mv /tmp/worker-service.tmp /etc/systemd/system/alive5-worker.service"
    if (Test-Path $tempFile) { Remove-Item $tempFile }
    
    # Reload systemd and restart services
    & ssh -i $KEY -o StrictHostKeyChecking=no $sshTarget "sudo systemctl daemon-reload"
    & ssh -i $KEY -o StrictHostKeyChecking=no $sshTarget "sudo systemctl enable alive5-backend alive5-worker"
    & ssh -i $KEY -o StrictHostKeyChecking=no $sshTarget "sudo systemctl restart alive5-backend alive5-worker"
    
    Write-Host "All services updated and restarted!" -ForegroundColor $SuccessColor
    
    # Verify deployment - show all files including hidden ones
    Write-Host ""
    Write-Host "Verifying deployment..." -ForegroundColor $InfoColor
    Write-Host "  - Root directory files (including hidden):" -ForegroundColor White
    $rootFiles = & ssh -i $KEY -o StrictHostKeyChecking=no "$USER@$SERVER" "ls -la /home/ubuntu/alive5-voice-agent/ | head -20" 2>&1
    Write-Host $rootFiles -ForegroundColor Gray
    
    Write-Host "  - Backend directory files:" -ForegroundColor White
    $backendFiles = & ssh -i $KEY -o StrictHostKeyChecking=no "$USER@$SERVER" "ls -la /home/ubuntu/alive5-voice-agent/backend/ 2>&1" 2>&1
    Write-Host $backendFiles -ForegroundColor Gray
    
    Write-Host "  - Virtual environment check:" -ForegroundColor White
    $venvStatus = & ssh -i $KEY -o StrictHostKeyChecking=no "$USER@$SERVER" "test -d /home/ubuntu/alive5-voice-agent/venv && echo '✅ venv exists' || echo '❌ venv missing'" 2>&1
    Write-Host "    $venvStatus" -ForegroundColor $(if ($venvStatus -match "exists") { $SuccessColor } else { $ErrorColor })
}

if ($choice -eq "4") {
    Write-Host "Deploying LiveKit setup files..." -ForegroundColor $InfoColor
    
    # Check if LiveKit files exist
    if (-not (Test-Path "livekit-docker-compose.yml")) {
        Write-Host "  ❌ Error: livekit-docker-compose.yml not found!" -ForegroundColor $ErrorColor
        Write-Host "  Please ensure you're in the project root directory." -ForegroundColor $ErrorColor
        return
    }
    
    if (-not (Test-Path "livekit-config.yaml")) {
        Write-Host "  ❌ Error: livekit-config.yaml not found!" -ForegroundColor $ErrorColor
        return
    }
    
    if (-not (Test-Path "coturn-config.conf")) {
        Write-Host "  ❌ Error: coturn-config.conf not found!" -ForegroundColor $ErrorColor
        return
    }
    
    if (-not (Test-Path "setup-livekit.sh")) {
        Write-Host "  ❌ Error: setup-livekit.sh not found!" -ForegroundColor $ErrorColor
        return
    }
    
    # Create /opt/livekit directory on server (requires sudo)
    Write-Host "  - Creating /opt/livekit directory on server..." -ForegroundColor White
    $mkdirResult = & ssh -i $KEY -o StrictHostKeyChecking=no "$USER@$SERVER" "sudo mkdir -p /opt/livekit && sudo chown ${USER}:${USER} /opt/livekit" 2>&1
    if ($LASTEXITCODE -ne 0) {
        Write-Host "  ⚠️  Warning: Could not create /opt/livekit, will use /tmp/livekit-setup instead" -ForegroundColor $WarningColor
        $livekitTarget = "${USER}@${SERVER}:/tmp/livekit-setup/"
        & ssh -i $KEY -o StrictHostKeyChecking=no "$USER@$SERVER" "mkdir -p /tmp/livekit-setup" 2>&1 | Out-Null
    } else {
        $livekitTarget = "${USER}@${SERVER}:/opt/livekit/"
        # Remove existing directories that conflict with files we need to deploy
        Write-Host "  - Cleaning up any conflicting directories..." -ForegroundColor White
        & ssh -i $KEY -o StrictHostKeyChecking=no "$USER@$SERVER" "sudo rm -rf /opt/livekit/livekit-config.yaml /opt/livekit/coturn-config.conf 2>/dev/null; true" 2>&1 | Out-Null
    }
    
    # Deploy LiveKit files
    Write-Host "  - Deploying livekit-docker-compose.yml..." -ForegroundColor White
    & scp -i $KEY -o StrictHostKeyChecking=no livekit-docker-compose.yml $livekitTarget
    
    Write-Host "  - Deploying livekit-config.yaml..." -ForegroundColor White
    & scp -i $KEY -o StrictHostKeyChecking=no livekit-config.yaml $livekitTarget
    
    Write-Host "  - Deploying coturn-config.conf..." -ForegroundColor White
    & scp -i $KEY -o StrictHostKeyChecking=no coturn-config.conf $livekitTarget
    
    Write-Host "  - Deploying setup-livekit.sh..." -ForegroundColor White
    & scp -i $KEY -o StrictHostKeyChecking=no setup-livekit.sh $livekitTarget
    
    # Make setup script executable
    Write-Host "  - Making setup script executable..." -ForegroundColor White
    if ($livekitTarget -like "*/opt/livekit/*") {
        & ssh -i $KEY -o StrictHostKeyChecking=no "$USER@$SERVER" "chmod +x /opt/livekit/setup-livekit.sh" 2>&1 | Out-Null
    } else {
        & ssh -i $KEY -o StrictHostKeyChecking=no "$USER@$SERVER" "chmod +x /tmp/livekit-setup/setup-livekit.sh" 2>&1 | Out-Null
    }
    
    # Verify files were deployed
    Write-Host "  - Verifying deployed files..." -ForegroundColor White
    if ($livekitTarget -like "*/opt/livekit/*") {
        $verifyResult = & ssh -i $KEY -o StrictHostKeyChecking=no "$USER@$SERVER" "ls -la /opt/livekit/" 2>&1
    } else {
        $verifyResult = & ssh -i $KEY -o StrictHostKeyChecking=no "$USER@$SERVER" "ls -la /tmp/livekit-setup/" 2>&1
    }
    Write-Host $verifyResult -ForegroundColor Gray
    
    Write-Host ""
    if ($livekitTarget -like "*/opt/livekit/*") {
        Write-Host "✅ LiveKit setup files deployed to /opt/livekit on server" -ForegroundColor $SuccessColor
        $setupPath = "/opt/livekit"
    } else {
        Write-Host "✅ LiveKit setup files deployed to /tmp/livekit-setup on server" -ForegroundColor $SuccessColor
        $setupPath = "/tmp/livekit-setup"
    }
    Write-Host ""
    Write-Host "Next steps:" -ForegroundColor $InfoColor
    Write-Host "  1. SSH into the server:" -ForegroundColor White
    Write-Host "     ssh -i alive5-voice-ai-agent.pem ubuntu@18.210.238.67" -ForegroundColor Cyan
    Write-Host ""
    Write-Host "  2. Run the setup script:" -ForegroundColor White
    Write-Host "     cd $setupPath" -ForegroundColor Cyan
    Write-Host "     sudo ./setup-livekit.sh" -ForegroundColor Cyan
    Write-Host ""
    Write-Host "  3. Save the API keys displayed at the end" -ForegroundColor White
    Write-Host "  4. Update your .env file with the new credentials" -ForegroundColor White
    Write-Host "  5. Update Telnyx SIP Connection settings" -ForegroundColor White
    Write-Host ""
    Write-Host "See LIVEKIT_SETUP.md for detailed instructions." -ForegroundColor $InfoColor
}

Write-Host ""
Write-Host "================================================================================" -ForegroundColor $InfoColor
Write-Host "Deployment completed successfully!" -ForegroundColor $SuccessColor
Write-Host "================================================================================" -ForegroundColor $InfoColor
Write-Host ""
Write-Host "Next steps:" -ForegroundColor $InfoColor
Write-Host "  - Run './check-services.ps1' to verify services are running" -ForegroundColor White
Write-Host "  - Run './logs-worker.ps1' to monitor worker logs" -ForegroundColor White
Write-Host "  - Run './logs-backend.ps1' to monitor backend logs" -ForegroundColor White
