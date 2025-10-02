# Smart deployment script with interactive deployment choice
Write-Host "Alive5 Voice Agent - Interactive Smart Deployment" -ForegroundColor Green
Write-Host "===============================================" -ForegroundColor Green
Write-Host "Server: 18.210.238.67" -ForegroundColor Yellow
Write-Host "User: ubuntu" -ForegroundColor Yellow
Write-Host ""

# Ask user what they want to deploy
Write-Host "What would you like to deploy?" -ForegroundColor Cyan
Write-Host "1. Backend only (backend/ directory)" -ForegroundColor White
Write-Host "2. Worker only (backend/worker/ directory)" -ForegroundColor White
Write-Host "3. Backend + Worker (backend/ directory including worker)" -ForegroundColor White
Write-Host "4. Full directory (all files and directories)" -ForegroundColor White
Write-Host "5. Full directory with new virtual environment and packages" -ForegroundColor White
Write-Host ""

do {
    $choice = Read-Host "Enter your choice (1-5)"
} while ($choice -notmatch '^[1-5]$')

$deployChoice = switch ($choice) {
    "1" { "backend" }
    "2" { "worker" }
    "3" { "backend_worker" }
    "4" { "full" }
    "5" { "full_with_venv" }
}

Write-Host ""
Write-Host "Selected deployment: $deployChoice" -ForegroundColor Green
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

# Deploy based on user choice
Write-Host "Starting deployment based on choice: $deployChoice" -ForegroundColor Cyan

switch ($deployChoice) {
    "backend" {
        Write-Host "Deploying Backend Only..." -ForegroundColor Cyan
        Sync-Directory "backend" "/home/ubuntu/alive5-voice-agent/" "Backend directory"
        Sync-File "requirements.txt" "/home/ubuntu/alive5-voice-agent/" "Requirements file"
        Sync-File ".env" "/home/ubuntu/alive5-voice-agent/" "Environment file"
    }
    
    "worker" {
        Write-Host "Deploying Worker Only..." -ForegroundColor Cyan
        # For worker only, we need to sync the entire backend directory since worker depends on it
        Sync-Directory "backend" "/home/ubuntu/alive5-voice-agent/" "Backend directory (including worker)"
        Sync-File "requirements.txt" "/home/ubuntu/alive5-voice-agent/" "Requirements file"
        Sync-File ".env" "/home/ubuntu/alive5-voice-agent/" "Environment file"
    }
    
    "backend_worker" {
        Write-Host "Deploying Backend + Worker..." -ForegroundColor Cyan
        Sync-Directory "backend" "/home/ubuntu/alive5-voice-agent/" "Backend directory (including worker)"
        Sync-File "requirements.txt" "/home/ubuntu/alive5-voice-agent/" "Requirements file"
        Sync-File ".env" "/home/ubuntu/alive5-voice-agent/" "Environment file"
    }
    
    "full" {
        Write-Host "Deploying Full Directory..." -ForegroundColor Cyan
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
    }
    
    "full_with_venv" {
        Write-Host "Deploying Full Directory with Virtual Environment Setup..." -ForegroundColor Cyan
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
    }
}

Write-Host "File synchronization completed!" -ForegroundColor Green

# Virtual environment setup (only for full_with_venv option)
if ($deployChoice -eq "full_with_venv") {
    Write-Host "Setting up virtual environment and packages..." -ForegroundColor Cyan
    
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
} else {
    Write-Host "Skipping virtual environment setup (not needed for this deployment type)" -ForegroundColor Yellow
}

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
ExecStart=/home/ubuntu/alive5-voice-agent/venv/bin/python -m uvicorn backend.main_dynamic:app --host=0.0.0.0 --port=8000
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

# Restart services based on deployment choice
Write-Host "Restarting services based on deployment..." -ForegroundColor Cyan

$servicesToRestart = switch ($deployChoice) {
    "backend" { "alive5-backend" }
    "worker" { "alive5-worker" }
    "backend_worker" { "alive5-backend alive5-worker" }
    "full" { "alive5-backend alive5-worker" }
    "full_with_venv" { "alive5-backend alive5-worker" }
}

Write-Host "Restarting services: $servicesToRestart" -ForegroundColor Cyan
ssh -i alive5-voice-ai-agent.pem -o ConnectTimeout=30 ubuntu@18.210.238.67 "sudo systemctl daemon-reload && sudo systemctl enable $servicesToRestart && sudo systemctl restart $servicesToRestart"

# Wait and check
Write-Host "Waiting for services to restart..." -ForegroundColor Cyan
Start-Sleep -Seconds 15

Write-Host "Checking service status..." -ForegroundColor Cyan
ssh -i alive5-voice-ai-agent.pem -o ConnectTimeout=30 ubuntu@18.210.238.67 "sudo systemctl status $servicesToRestart --no-pager -l"

# Verify files on server
Write-Host "Verifying files on server..." -ForegroundColor Cyan
ssh -i alive5-voice-ai-agent.pem -o ConnectTimeout=30 ubuntu@18.210.238.67 "ls -la /home/ubuntu/alive5-voice-agent/"

Write-Host ""
Write-Host "Smart deployment completed!" -ForegroundColor Green
Write-Host "Deployed: $deployChoice" -ForegroundColor Cyan
Write-Host "Backend: http://18.210.238.67" -ForegroundColor Yellow
Write-Host "Health: http://18.210.238.67/health" -ForegroundColor Yellow

# Show what was deployed
$deployedItems = switch ($deployChoice) {
    "backend" { "  - backend/ directory (main backend application)" }
    "worker" { "  - backend/ directory (including worker)" }
    "backend_worker" { "  - backend/ directory (including worker)" }
    "full" { 
        "  - backend/ directory (main backend application)`n" +
        "  - frontend/ directory (web interface)`n" +
        "  - flow_states/ directory (conversation states)`n" +
        "  - Optional directories (docs/, KMS/)"
    }
    "full_with_venv" { 
        "  - backend/ directory (main backend application)`n" +
        "  - frontend/ directory (web interface)`n" +
        "  - flow_states/ directory (conversation states)`n" +
        "  - Optional directories (docs/, KMS/)`n" +
        "  - NEW virtual environment created`n" +
        "  - All packages installed from requirements.txt"
    }
}

Write-Host ""
Write-Host "This deployment included:" -ForegroundColor Yellow
Write-Host $deployedItems -ForegroundColor Yellow
Write-Host "  - Configuration files (requirements.txt, .env, README.md)" -ForegroundColor Yellow
