# Fix AWS Security Group for HTTPS Access
Write-Host "AWS Security Group Fix for HTTPS Access" -ForegroundColor Green
Write-Host "=======================================" -ForegroundColor Green

$serverIP = "18.210.238.67"

Write-Host ""
Write-Host "Current Status:" -ForegroundColor Yellow
Write-Host "- Backend is running correctly on the server" -ForegroundColor Green
Write-Host "- SSL certificate is configured and valid" -ForegroundColor Green
Write-Host "- Nginx is proxying HTTPS requests" -ForegroundColor Green
Write-Host "- Issue: AWS Security Group blocking port 443" -ForegroundColor Red

Write-Host ""
Write-Host "Required Action:" -ForegroundColor Cyan
Write-Host "Open port 443 (HTTPS) in AWS Security Group" -ForegroundColor White

Write-Host ""
Write-Host "Step-by-Step Instructions:" -ForegroundColor Cyan
Write-Host "1. Login to AWS Console: https://console.aws.amazon.com/ec2/" -ForegroundColor White
Write-Host "2. Navigate to: EC2 → Security Groups" -ForegroundColor White
Write-Host "3. Find security group for instance: $serverIP" -ForegroundColor White
Write-Host "4. Click 'Edit inbound rules'" -ForegroundColor White
Write-Host "5. Click 'Add rule'" -ForegroundColor White
Write-Host "6. Configure the rule:" -ForegroundColor White
Write-Host "   - Type: HTTPS" -ForegroundColor Gray
Write-Host "   - Protocol: TCP" -ForegroundColor Gray
Write-Host "   - Port range: 443" -ForegroundColor Gray
Write-Host "   - Source: 0.0.0.0/0" -ForegroundColor Gray
Write-Host "   - Description: Alive5 Voice Agent HTTPS Access" -ForegroundColor Gray
Write-Host "7. Click 'Save rules'" -ForegroundColor White

Write-Host ""
Write-Host "Alternative: AWS CLI (if configured):" -ForegroundColor Cyan
Write-Host "aws ec2 authorize-security-group-ingress --group-id <SECURITY_GROUP_ID> --protocol tcp --port 443 --cidr 0.0.0.0/0" -ForegroundColor Gray

Write-Host ""
Write-Host "After opening port 443, test with:" -ForegroundColor Cyan
Write-Host "https://$serverIP.nip.io/health" -ForegroundColor White

Write-Host ""
Write-Host "Why port 443 is needed:" -ForegroundColor Yellow
Write-Host "- Frontend is deployed on Vercel (HTTPS only)" -ForegroundColor White
Write-Host "- Browsers block HTTP requests from HTTPS pages" -ForegroundColor White
Write-Host "- Backend must use HTTPS to communicate with frontend" -ForegroundColor White
Write-Host "- Port 443 is the standard HTTPS port" -ForegroundColor White

Write-Host ""
Write-Host "Security Note:" -ForegroundColor Green
Write-Host "Port 443 is the standard HTTPS port used by all major websites" -ForegroundColor White
Write-Host "This is a secure, industry-standard configuration" -ForegroundColor White

Write-Host ""
Write-Host "Expected Result:" -ForegroundColor Green
Write-Host "Once port 443 is open, your voice agent will be accessible at:" -ForegroundColor White
Write-Host "https://$serverIP.nip.io" -ForegroundColor Yellow
Write-Host "https://$serverIP.nip.io/health" -ForegroundColor Yellow
Write-Host "https://$serverIP.nip.io/api/connection_details" -ForegroundColor Yellow
