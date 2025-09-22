# Smart deployment script - only copy missing/changed files
Write-Host "Alive5 Voice Agent - Smart Deployment" -ForegroundColor Green
Write-Host "=====================================" -ForegroundColor Green
Write-Host "Server: 18.210.238.67" -ForegroundColor Yellow
Write-Host "User: ubuntu" -ForegroundColor Yellow
Write-Host ""

# Check if SSH key exists
if (-not (Test-Path "alive5-voice-ai-agent.pem")) {
    Write-Host "SSH key file 'alive5-voice-ai-agent.pem' not found!" -ForegroundColor Red
    Write-Host "Please ensure the SSH key is in the current directory." -ForegroundColor Red
    exit 1
}

Write-Host "SSH key found" -ForegroundColor Green

# Fix SSH key permissions
Write-Host "Setting SSH key permissions..." -ForegroundColor Cyan
icacls alive5-voice-ai-agent.pem /inheritance:r | Out-Null
icacls alive5-voice-ai-agent.pem /grant:r "$($env:USERNAME):(R)" | Out-Null
Write-Host "SSH key permissions set" -ForegroundColor Green

# Ensure server directory exists
Write-Host "Ensuring server directory exists..." -ForegroundColor Cyan
ssh -i alive5-voice-ai-agent.pem -o ConnectTimeout=30 ubuntu@18.210.238.67 "mkdir -p /home/ubuntu/alive5-voice-agent"

# Function to sync directory
function Sync-Directory {
    param($LocalPath, $RemotePath, $Description)
    
    Write-Host "Syncing $Description..." -ForegroundColor Cyan
    
    if (Test-Path $LocalPath) {
        # Copy directory recursively, overwriting existing files
        scp -i alive5-voice-ai-agent.pem -o ConnectTimeout=60 -r $LocalPath ubuntu@18.210.238.67:$RemotePath
        Write-Host "$Description synced" -ForegroundColor Green
    } else {
        Write-Host "$Description not found locally, skipping..." -ForegroundColor Yellow
    }
}

# Function to sync file
function Sync-File {
    param($LocalPath, $RemotePath, $Description)
    
    Write-Host "Syncing $Description..." -ForegroundColor Cyan
    
    if (Test-Path $LocalPath) {
        scp -i alive5-voice-ai-agent.pem -o ConnectTimeout=60 $LocalPath ubuntu@18.210.238.67:$RemotePath
        Write-Host "$Description synced" -ForegroundColor Green
    } else {
        Write-Host "$Description not found locally, skipping..." -ForegroundColor Yellow
    }
}

# Sync all necessary files and directories
Write-Host "Syncing application files..." -ForegroundColor Cyan

# Core application files
Sync-Directory "backend" "/home/ubuntu/alive5-voice-agent/" "Backend directory"
Sync-Directory "frontend" "/home/ubuntu/alive5-voice-agent/" "Frontend directory"
Sync-Directory "flow_states" "/home/ubuntu/alive5-voice-agent/" "Flow states directory"

# Configuration files
Sync-File "requirements.txt" "/home/ubuntu/alive5-voice-agent/" "Requirements file"
Sync-File "README.md" "/home/ubuntu/alive5-voice-agent/" "README file"
Sync-File ".env" "/home/ubuntu/alive5-voice-agent/" "Environment file"

# Optional directories (if they exist)
Sync-Directory "docs" "/home/ubuntu/alive5-voice-agent/" "Documentation directory"
Sync-Directory "KMS" "/home/ubuntu/alive5-voice-agent/" "KMS directory"

Write-Host "File synchronization completed!" -ForegroundColor Green

# Install the missing package
Write-Host "Installing python3.12-venv package..." -ForegroundColor Cyan
ssh -i alive5-voice-ai-agent.pem -o ConnectTimeout=60 ubuntu@18.210.238.67 "sudo apt install -y python3.12-venv"

# Stop services
Write-Host "Stopping services..." -ForegroundColor Cyan
ssh -i alive5-voice-ai-agent.pem -o ConnectTimeout=30 ubuntu@18.210.238.67 "sudo systemctl stop alive5-backend alive5-worker 2>/dev/null || true"

# Remove old virtual environment
Write-Host "Removing old virtual environment..." -ForegroundColor Cyan
ssh -i alive5-voice-ai-agent.pem -o ConnectTimeout=30 ubuntu@18.210.238.67 "cd /home/ubuntu/alive5-voice-agent && rm -rf venv"

# Create new virtual environment
Write-Host "Creating new virtual environment..." -ForegroundColor Cyan
ssh -i alive5-voice-ai-agent.pem -o ConnectTimeout=60 ubuntu@18.210.238.67 "cd /home/ubuntu/alive5-voice-agent && python3.12 -m venv venv"

# Install packages
Write-Host "Installing packages..." -ForegroundColor Cyan
ssh -i alive5-voice-ai-agent.pem -o ConnectTimeout=120 ubuntu@18.210.238.67 "cd /home/ubuntu/alive5-voice-agent && venv/bin/pip install -r requirements.txt"

# Create and install service files
Write-Host "Creating service files..." -ForegroundColor Cyan

# Backend service (running as root for port 80)
$backendService = @"
[Unit]
Description=Alive5 Voice Agent Backend
After=network.target

[Service]
Type=simple
User=root
Group=root
WorkingDirectory=/home/ubuntu/alive5-voice-agent
Environment=PATH=/home/ubuntu/alive5-voice-agent/venv/bin
ExecStart=/home/ubuntu/alive5-voice-agent/venv/bin/python -m uvicorn backend.main_dynamic:app --host=0.0.0.0 --port=80
Restart=always
RestartSec=10

[Install]
WantedBy=multi-user.target
"@

# Worker service (with correct start command)
$workerService = @"
[Unit]
Description=Alive5 Voice Agent Worker
After=network.target

[Service]
Type=simple
User=ubuntu
Group=ubuntu
WorkingDirectory=/home/ubuntu/alive5-voice-agent
Environment=PATH=/home/ubuntu/alive5-voice-agent/venv/bin
ExecStart=/home/ubuntu/alive5-voice-agent/venv/bin/python backend/worker/main_flow_based.py start
Restart=always
RestartSec=10

[Install]
WantedBy=multi-user.target
"@

# Install service files
echo $backendService | ssh -i alive5-voice-ai-agent.pem -o ConnectTimeout=30 ubuntu@18.210.238.67 "sudo tee /etc/systemd/system/alive5-backend.service > /dev/null"
echo $workerService | ssh -i alive5-voice-ai-agent.pem -o ConnectTimeout=30 ubuntu@18.210.238.67 "sudo tee /etc/systemd/system/alive5-worker.service > /dev/null"

# Start services
Write-Host "Starting services..." -ForegroundColor Cyan
ssh -i alive5-voice-ai-agent.pem -o ConnectTimeout=30 ubuntu@18.210.238.67 "sudo systemctl daemon-reload && sudo systemctl enable alive5-backend alive5-worker && sudo systemctl start alive5-backend alive5-worker"

# Wait and check
Write-Host "Waiting for services to start..." -ForegroundColor Cyan
Start-Sleep -Seconds 15

Write-Host "Checking service status..." -ForegroundColor Cyan
ssh -i alive5-voice-ai-agent.pem -o ConnectTimeout=30 ubuntu@18.210.238.67 "sudo systemctl status alive5-backend --no-pager -l"
ssh -i alive5-voice-ai-agent.pem -o ConnectTimeout=30 ubuntu@18.210.238.67 "sudo systemctl status alive5-worker --no-pager -l"

# Verify files on server
Write-Host "Verifying files on server..." -ForegroundColor Cyan
ssh -i alive5-voice-ai-agent.pem -o ConnectTimeout=30 ubuntu@18.210.238.67 "ls -la /home/ubuntu/alive5-voice-agent/"

Write-Host ""
Write-Host "Smart deployment completed!" -ForegroundColor Green
Write-Host "Backend: http://18.210.238.67" -ForegroundColor Yellow
Write-Host "Health: http://18.210.238.67/health" -ForegroundColor Yellow
