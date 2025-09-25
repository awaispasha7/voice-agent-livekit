# View worker service logs with clean output (no systemd prefixes)
Write-Host "Alive5 Worker Logs" -ForegroundColor Green
Write-Host "==================" -ForegroundColor Green
Write-Host "Press Ctrl+C to exit" -ForegroundColor Yellow
Write-Host ""

# Set UTF-8 encoding for proper emoji display
$OutputEncoding = [System.Text.Encoding]::UTF8
[Console]::OutputEncoding = [System.Text.Encoding]::UTF8

ssh -i alive5-voice-ai-agent.pem -o ConnectTimeout=10 ubuntu@18.210.238.67 "sudo journalctl -u alive5-worker -f --no-pager" | ForEach-Object {
    # Remove systemd prefixes like "Sep 25 12:32:06 ip-172-26-1-10 python[93882]: "
    if ($_ -match '^[A-Za-z]{3}\s+\d{1,2}\s+\d{2}:\d{2}:\d{2}\s+[^\s]+\s+python\[\d+\]:\s*(.*)$') {
        # Clean up encoded emojis
        $cleanMessage = $matches[1] -replace [char]0x2261 + [char]0x0192 + [char]0x00DC + [char]0x0107, [char]0x2705
        Write-Host $cleanMessage
    } else {
        # Clean up encoded emojis
        $cleanLine = $_ -replace [char]0x2261 + [char]0x0192 + [char]0x00DC + [char]0x0107, [char]0x2705
        Write-Host $cleanLine
    }
}
