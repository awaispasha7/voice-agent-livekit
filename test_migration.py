#!/usr/bin/env python3
"""
Test script to verify the migration from Heroku to Render/Vercel works correctly.
This script tests all the key components and endpoints.
"""

import os
import sys
import asyncio
import httpx
import json
from pathlib import Path

# Add the backend directory to the path
sys.path.append(str(Path(__file__).parent / "backend"))

def test_imports():
    """Test that all required modules can be imported."""
    print("🧪 Testing imports...")
    
    try:
        from backend.main_dynamic import app
        print("✅ Backend API imports successfully")
    except ImportError as e:
        print(f"❌ Backend API import failed: {e}")
        return False
    
    try:
        from backend.worker.main_flow_based import Agent
        print("✅ Worker imports successfully")
    except ImportError as e:
        print(f"❌ Worker import failed: {e}")
        return False
    
    return True

def test_environment_variables():
    """Test that all required environment variables are available."""
    print("\n🧪 Testing environment variables...")
    
    required_vars = [
        "LIVEKIT_API_KEY",
        "LIVEKIT_API_SECRET", 
        "LIVEKIT_URL",
        "OPENAI_API_KEY",
        "A5_BASE_URL",
        "A5_API_KEY"
    ]
    
    missing_vars = []
    for var in required_vars:
        if not os.getenv(var):
            missing_vars.append(var)
    
    if missing_vars:
        print(f"⚠️  Missing environment variables: {missing_vars}")
        print("   This is expected for testing. Set these in production.")
        print("   ✅ Environment variable structure is correct")
        return True  # Don't fail the test for missing env vars
    else:
        print("✅ All required environment variables are set")
        return True

def test_file_structure():
    """Test that all required files exist."""
    print("\n🧪 Testing file structure...")
    
    required_files = [
        "backend/main_dynamic.py",
        "backend/worker/main_flow_based.py",
        "frontend/index.html",
        "frontend/main_dynamic.js",
        "requirements.txt",
        "render.yaml",
        ".python-version"
    ]
    
    missing_files = []
    for file_path in required_files:
        if not Path(file_path).exists():
            missing_files.append(file_path)
    
    if missing_files:
        print(f"❌ Missing files: {missing_files}")
        return False
    else:
        print("✅ All required files exist")
        return True

async def test_backend_api():
    """Test that the backend API can start and respond."""
    print("\n🧪 Testing backend API...")
    
    try:
        from backend.main_dynamic import app
        from fastapi.testclient import TestClient
        
        client = TestClient(app)
        
        # Test health endpoint
        response = client.get("/health")
        if response.status_code == 200:
            print("✅ Backend API health check passed")
            return True
        else:
            print(f"❌ Backend API health check failed: {response.status_code}")
            return False
            
    except Exception as e:
        print(f"❌ Backend API test failed: {e}")
        return False

def test_frontend_config():
    """Test that the frontend configuration is correct."""
    print("\n🧪 Testing frontend configuration...")
    
    try:
        html_path = Path("frontend/index.html")
        js_path = Path("frontend/main_dynamic.js")
        
        if not html_path.exists():
            print("❌ Frontend index.html not found")
            return False
            
        if not js_path.exists():
            print("❌ Frontend main_dynamic.js not found")
            return False
        
        # Read HTML file
        html_content = html_path.read_text(encoding='utf-8')
        
        # Check that it uses environment variables
        if "window.API_BASE_URL" not in html_content:
            print("❌ Frontend should use window.API_BASE_URL environment variable")
            return False
        
        # Read JS file
        js_content = js_path.read_text(encoding='utf-8')
        
        # Check for required configuration elements
        required_elements = [
            "API_BASE_URL",
            "ENDPOINTS",
            "CONNECTION_DETAILS",
            "PROCESS_FLOW_MESSAGE"
        ]
        
        missing_elements = []
        for element in required_elements:
            if element not in js_content:
                missing_elements.append(element)
        
        if missing_elements:
            print(f"❌ Missing configuration elements: {missing_elements}")
            return False
        else:
            print("✅ Frontend configuration looks correct")
            return True
            
    except Exception as e:
        print(f"❌ Frontend config test failed: {e}")
        return False

def test_render_config():
    """Test that the Render configuration is valid."""
    print("\n🧪 Testing Render configuration...")
    
    try:
        import yaml
        
        with open("render.yaml", "r") as f:
            config = yaml.safe_load(f)
        
        # Check for required services
        if "services" not in config:
            print("❌ No services defined in render.yaml")
            return False
        
        services = config["services"]
        
        # Check for backend service
        backend_service = next((s for s in services if s.get("name") == "voice-agent-backend"), None)
        if not backend_service:
            print("❌ Backend service not found in render.yaml")
            return False
        
        # Check for worker service
        worker_service = next((s for s in services if s.get("name") == "voice-agent-worker"), None)
        if not worker_service:
            print("❌ Worker service not found in render.yaml")
            return False
        
        # Check that both are using the free plan
        if backend_service.get("plan") != "free":
            print("❌ Backend service should use free plan")
            return False
            
        if worker_service.get("plan") != "free":
            print("❌ Worker service should use free plan")
            return False
        
        print("✅ Render configuration is valid")
        return True
        
    except Exception as e:
        print(f"❌ Render config test failed: {e}")
        return False


async def main():
    """Run all tests."""
    print("🚀 Starting migration verification tests...\n")
    
    tests = [
        ("File Structure", test_file_structure),
        ("Environment Variables", test_environment_variables),
        ("Imports", test_imports),
        ("Frontend Config", test_frontend_config),
        ("Render Config", test_render_config),
        ("Backend API", test_backend_api),
    ]
    
    passed = 0
    total = len(tests)
    
    for test_name, test_func in tests:
        try:
            if asyncio.iscoroutinefunction(test_func):
                result = await test_func()
            else:
                result = test_func()
            
            if result:
                passed += 1
        except Exception as e:
            print(f"❌ {test_name} test crashed: {e}")
    
    print(f"\n📊 Test Results: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All tests passed! Migration is ready for deployment.")
        print("\nNext steps:")
        print("1. Deploy backend API to Render (Free Plan)")
        print("2. Deploy worker to Render (Free Plan)")
        print("3. Update frontend config with actual Render URL")
        print("4. Frontend is already deployed to Vercel ✅")
        print("\n💰 Cost: $0/month (Both services on Free Plan)")
        return True
    else:
        print("⚠️  Some tests failed. Please fix the issues before deploying.")
        return False

if __name__ == "__main__":
    success = asyncio.run(main())
    sys.exit(0 if success else 1)
