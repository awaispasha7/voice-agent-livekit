#!/bin/bash

# Test script for Alive5 Voice Agent deployment
# Server: 18.210.238.67

echo "🧪 Testing Alive5 Voice Agent Deployment"
echo "========================================"
echo "Server: 18.210.238.67"
echo ""

# Test SSH connection
echo "🔍 Testing SSH connection..."
ssh -i alive5-voice-ai-agent.pem -o ConnectTimeout=10 -o StrictHostKeyChecking=no ubuntu@18.210.238.67 "echo 'SSH connection successful'" || {
    echo "❌ SSH connection failed!"
    exit 1
}
echo "✅ SSH connection successful"

# Test service status
echo "🔍 Checking service status..."
ssh -i alive5-voice-ai-agent.pem ubuntu@18.210.238.67 << 'EOF'
echo "Backend service status:"
sudo systemctl is-active alive5-backend
echo "Worker service status:"
sudo systemctl is-active alive5-worker
EOF

# Test health endpoints
echo "🔍 Testing health endpoints..."

echo "Testing direct backend access..."
curl -f http://18.210.238.67:8000/health && echo "✅ Backend health check passed" || echo "❌ Backend health check failed"

echo "Testing nginx proxy..."
curl -f http://18.210.238.67/health && echo "✅ Nginx proxy health check passed" || echo "❌ Nginx proxy health check failed"

echo "Testing template status..."
curl -f http://18.210.238.67/api/template_status && echo "✅ Template status check passed" || echo "❌ Template status check failed"

# Test API endpoints
echo "🔍 Testing API endpoints..."

echo "Testing flow intents endpoint..."
curl -f -X POST http://18.210.238.67/api/detect_flow_intent \
  -H "Content-Type: application/json" \
  -d '{"message": "Hello, I need help with pricing"}' && echo "✅ Flow intent detection passed" || echo "❌ Flow intent detection failed"

echo "Testing template refresh endpoint..."
curl -f -X POST http://18.210.238.67/api/refresh_template && echo "✅ Template refresh passed" || echo "❌ Template refresh failed"

echo ""
echo "🎉 Deployment test completed!"
echo ""
echo "🌐 Your services are available at:"
echo "   Backend API: http://18.210.238.67:8000"
echo "   Health Check: http://18.210.238.67/health"
echo "   Template Status: http://18.210.238.67/api/template_status"
