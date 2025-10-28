#!/usr/bin/env pwsh

# Alive5 Backend Logs Viewer
# Clean, real-time backend service logs

Write-Host "================================================================================" -ForegroundColor Cyan
Write-Host "Alive5 Backend Logs" -ForegroundColor Cyan
Write-Host "================================================================================" -ForegroundColor Cyan
Write-Host "Press Ctrl+C to exit" -ForegroundColor Yellow
Write-Host ""

# Set UTF-8 encoding for proper display
$OutputEncoding = [System.Text.Encoding]::UTF8
[Console]::OutputEncoding = [System.Text.Encoding]::UTF8

# Check if key file exists
if (-not (Test-Path "alive5-voice-ai-agent.pem")) {
    Write-Host "Error: alive5-voice-ai-agent.pem not found!" -ForegroundColor Red
    Write-Host "Please ensure the SSH key is in the current directory." -ForegroundColor Red
    exit 1
}

try {
    ssh -i alive5-voice-ai-agent.pem -o ConnectTimeout=10 ubuntu@18.210.238.67 "sudo journalctl -u alive5-backend -f --no-pager" | ForEach-Object {
        # Remove systemd prefixes and clean up output
        if ($_ -match '^[A-Za-z]{3}\s+\d{1,2}\s+\d{2}:\d{2}:\d{2}\s+[^\s]+\s+python\[\d+\]:\s*(.*)$') {
            Write-Host $matches[1]
        } else {
            Write-Host $_
        }
    }
} catch {
    Write-Host "Error connecting to server or service not found" -ForegroundColor Red
    Write-Host "Make sure the backend service is running: ./check-services.ps1" -ForegroundColor Yellow
}