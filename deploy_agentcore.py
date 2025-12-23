"""
Deploy AgentCore Agent using AWS SDK (boto3)
Uses credentials from .env file
"""

import os
import boto3
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Get AWS credentials from .env
aws_access_key = os.getenv("AWS_ACCESS_KEY_ID")
aws_secret_key = os.getenv("AWS_SECRET_ACCESS_KEY")
aws_region = os.getenv("AWS_DEFAULT_REGION", "us-east-1")

if not aws_access_key or not aws_secret_key:
    print("❌ Error: AWS_ACCESS_KEY_ID and AWS_SECRET_ACCESS_KEY must be set in .env")
    exit(1)

# Initialize boto3 client
bedrock_agent = boto3.client(
    'bedrock-agent',
    region_name=aws_region,
    aws_access_key_id=aws_access_key,
    aws_secret_access_key=aws_secret_key
)

# Agent configuration
agent_name = "alive5-voice-agent"
agent_path = Path("alive5-backend/agentcore/agent.py")

if not agent_path.exists():
    print(f"❌ Error: Agent file not found at {agent_path}")
    exit(1)

print(f"📦 Deploying agent: {agent_name}")
print(f"📄 Agent file: {agent_path}")
print(f"🌍 Region: {aws_region}")
print()

try:
    # Read agent code
    with open(agent_path, 'r', encoding='utf-8') as f:
        agent_code = f.read()
    
    print("✅ Agent code loaded")
    
    # Note: The actual deployment API might be different
    # This is a template - you may need to adjust based on actual Bedrock Agent API
    print("\n⚠️  Note: Bedrock Agent API for deployment may require:")
    print("   1. Creating an agent first")
    print("   2. Creating an agent version with the code")
    print("   3. Deploying the version")
    print("\n💡 Recommendation: Use AWS Console or check AWS CLI commands")
    print("\n📋 Alternative: Use AWS Console at:")
    print("   https://console.aws.amazon.com/bedrock/")
    print("   Navigate to: Agents → Create Agent")
    
    # Try to list existing agents first
    try:
        print("\n🔍 Checking existing agents...")
        response = bedrock_agent.list_agents()
        agents = response.get('agentSummaries', [])
        if agents:
            print(f"   Found {len(agents)} existing agent(s):")
            for agent in agents:
                print(f"   - {agent.get('agentName')} (ID: {agent.get('agentId')})")
        else:
            print("   No existing agents found")
    except Exception as e:
        print(f"   ⚠️  Could not list agents: {e}")
        print("   This might be normal if you don't have permissions or the API is different")
    
except Exception as e:
    print(f"❌ Error: {e}")
    import traceback
    traceback.print_exc()
    exit(1)

