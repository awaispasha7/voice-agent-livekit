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

# Deployment options
Write-Host ""
Write-Host "Select deployment option:" -ForegroundColor $InfoColor
Write-Host "1. Deploy worker only (alive5-worker directory)" -ForegroundColor White
Write-Host "2. Deploy full backend (backend + worker)" -ForegroundColor White
Write-Host "3. Deploy all with requirements (backend + worker + install dependencies)" -ForegroundColor White
Write-Host ""

do {
    $choice = Read-Host "Enter your choice (1, 2, or 3)"
} while ($choice -notin @("1", "2", "3"))

Write-Host ""

if ($choice -eq "1") {
    Write-Host "Deploying worker only..." -ForegroundColor $InfoColor
    
    # Create directory on server
    Write-Host "  - Creating directory structure..." -ForegroundColor White
    & ssh -i $KEY -o StrictHostKeyChecking=no "$USER@$SERVER" "mkdir -p /home/ubuntu/alive5-voice-agent/backend/alive5-worker"
    
    # Deploy worker files
    $scpTarget = "${USER}@${SERVER}:/home/ubuntu/alive5-voice-agent/backend/alive5-worker/"
    Write-Host "  - Deploying worker.py..." -ForegroundColor White
    & scp -i $KEY -o StrictHostKeyChecking=no alive5-backend/alive5-worker/worker.py $scpTarget
    
    Write-Host "  - Deploying functions.py..." -ForegroundColor White
    & scp -i $KEY -o StrictHostKeyChecking=no alive5-backend/alive5-worker/functions.py $scpTarget
    
    Write-Host "  - Deploying system_prompt.py..." -ForegroundColor White
    & scp -i $KEY -o StrictHostKeyChecking=no alive5-backend/alive5-worker/system_prompt.py $scpTarget
    
    Write-Host "  - Deploying .env file..." -ForegroundColor White
    & scp -i $KEY -o StrictHostKeyChecking=no .env "${USER}@${SERVER}:/home/ubuntu/alive5-voice-agent/"
    
    Write-Host "  - Deploying requirements.txt..." -ForegroundColor White
    & scp -i $KEY -o StrictHostKeyChecking=no requirements.txt "${USER}@${SERVER}:/home/ubuntu/alive5-voice-agent/"
    
    Write-Host "Worker files deployed successfully!" -ForegroundColor $SuccessColor
    
    # Create/update worker service
    Write-Host "Creating/updating worker service..." -ForegroundColor $InfoColor
    
    # Build simplified service content - applications will load .env file themselves
    $serviceLines = @(
        "[Unit]",
        "Description=Alive5 Voice Agent Worker",
        "After=network.target",
        "",
        "[Service]",
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
    
} elseif ($choice -eq "2") {
    Write-Host "Deploying full backend..." -ForegroundColor $InfoColor
    
    # Create directories on server
    Write-Host "  - Creating directory structure..." -ForegroundColor White
    & ssh -i $KEY -o StrictHostKeyChecking=no "$USER@$SERVER" "mkdir -p /home/ubuntu/alive5-voice-agent/backend/alive5-worker"
    
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
    
    Write-Host "Worker files deployed successfully!" -ForegroundColor $SuccessColor
    
    # Create backend service
    Write-Host "Creating backend service..." -ForegroundColor $InfoColor
    
    # Build simplified backend service content - applications will load .env file themselves
    $backendServiceLines = @(
        "[Unit]",
        "Description=Alive5 Voice Agent Backend",
        "After=network.target",
        "",
        "[Service]",
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
    
    # Build simplified worker service content - applications will load .env file themselves
    $workerServiceLines = @(
        "[Unit]",
        "Description=Alive5 Voice Agent Worker",
        "After=network.target",
        "",
        "[Service]",
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
    
} else {
    # Option 3: Deploy all with requirements
    Write-Host "Deploying all with requirements..." -ForegroundColor $InfoColor
    
    # Create directories on server
    Write-Host "  - Creating directory structure..." -ForegroundColor White
    & ssh -i $KEY -o StrictHostKeyChecking=no "$USER@$SERVER" "mkdir -p /home/ubuntu/alive5-voice-agent/backend/alive5-worker"
    
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
    
    Write-Host "Worker files deployed successfully!" -ForegroundColor $SuccessColor
    
    # Install requirements
    Write-Host "Installing requirements..." -ForegroundColor $InfoColor
    Write-Host "  - Updating pip..." -ForegroundColor White
    $installResult = & ssh -i $KEY -o StrictHostKeyChecking=no "$USER@$SERVER" "cd /home/ubuntu/alive5-voice-agent && /home/ubuntu/alive5-voice-agent/venv/bin/pip install --upgrade pip" 2>&1
    if ($LASTEXITCODE -ne 0) {
        Write-Host "  ⚠️ Warning: pip upgrade failed, continuing..." -ForegroundColor $WarningColor
    }
    
    Write-Host "  - Installing packages from requirements.txt..." -ForegroundColor White
    $installResult = & ssh -i $KEY -o StrictHostKeyChecking=no "$USER@$SERVER" "cd /home/ubuntu/alive5-voice-agent && /home/ubuntu/alive5-voice-agent/venv/bin/pip install -r requirements.txt" 2>&1
    if ($LASTEXITCODE -eq 0) {
        Write-Host "  ✅ Requirements installed successfully!" -ForegroundColor $SuccessColor
    } else {
        Write-Host "  ❌ Error installing requirements:" -ForegroundColor $ErrorColor
        Write-Host $installResult -ForegroundColor $ErrorColor
        Write-Host "  ⚠️ Continuing with service setup..." -ForegroundColor $WarningColor
    }
    
    # Create backend service
    Write-Host "Creating backend service..." -ForegroundColor $InfoColor
    
    # Build simplified backend service content - applications will load .env file themselves
    $backendServiceLines = @(
        "[Unit]",
        "Description=Alive5 Voice Agent Backend",
        "After=network.target",
        "",
        "[Service]",
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
    
    # Build simplified worker service content - applications will load .env file themselves
    $workerServiceLines = @(
        "[Unit]",
        "Description=Alive5 Voice Agent Worker",
        "After=network.target",
        "",
        "[Service]",
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
