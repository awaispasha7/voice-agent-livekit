"""
Function Definitions and Handlers for Alive5 Voice Agent
"""

import json
import logging
import os
from typing import Dict, Any
import httpx
from dotenv import load_dotenv
from pathlib import Path

# Load environment variables
load_dotenv(Path(__file__).parent / "../../.env")

logger = logging.getLogger(__name__)

# Environment variables
A5_BASE_URL = os.getenv("A5_BASE_URL")
A5_API_KEY = os.getenv("A5_API_KEY")
A5_TEMPLATE_URL = os.getenv("A5_TEMPLATE_URL", "/1.0/org-botchain/generate-template")
A5_FAQ_URL = os.getenv("A5_FAQ_URL", "/public/1.0/get-faq-bot-response-by-bot-id")
FAQ_BOT_ID = os.getenv("FAQ_BOT_ID", "faq_b9952a56-fc7b-41c9-b0a0-5c662ddb039e")

# ============================================================================
# FUNCTION HANDLERS
# ============================================================================

async def handle_load_bot_flows(botchain_name: str, org_name: str = "alive5stage0") -> Dict[str, Any]:
    """Load Alive5 bot flow definitions dynamically"""
    try:
        if not A5_BASE_URL or not A5_API_KEY:
            return {
                "success": False,
                "error": "Missing Alive5 API configuration"
            }
        
        logger.info(f"🔧 Loading bot flows for {botchain_name}")
        
        async with httpx.AsyncClient(timeout=30.0) as client:
            response = await client.post(
                f"{A5_BASE_URL}{A5_TEMPLATE_URL}",
                headers={
                    "X-A5-APIKEY": A5_API_KEY,
                    "Content-Type": "application/json"
                },
                json={
                    "botchain_name": botchain_name,
                    "org_name": org_name
                }
            )
            
            if response.status_code == 200:
                data = response.json()
                return {
                    "success": True,
                    "data": data,
                    "template": data,
                    "intents": list(data.get("data", {}).keys()) if data.get("data") else []
                }
            else:
                logger.error(f"❌ Failed to load flows: {response.status_code} - {response.text}")
                return {
                    "success": False,
                    "error": f"API returned status {response.status_code}"
                }
                
    except Exception as e:
        logger.error(f"❌ Error loading bot flows: {e}")
        return {
            "success": False,
            "error": str(e)
        }

async def handle_faq_bot_request(
    faq_question: str,
    bot_id: str = FAQ_BOT_ID,
    isVoice: bool = True,
    waiting_callback = None
) -> Dict[str, Any]:
    """Call the Alive5 FAQ bot API"""
    try:
        if not A5_BASE_URL or not A5_API_KEY:
            return {
                "success": False,
                "error": "FAQ API not configured"
            }
        
        logger.info(f"🔧 FAQ bot request: {faq_question}")
        
        if waiting_callback:
            await waiting_callback("Please wait while I fetch the information from the Alive5 website...")
        
        # Create a task for the HTTP request
        import asyncio
        
        async def make_request():
            async with httpx.AsyncClient(timeout=30.0) as client:
                response = await client.post(
                    f"{A5_BASE_URL}{A5_FAQ_URL}",
                    headers={
                        "X-A5-APIKEY": A5_API_KEY,
                        "Content-Type": "application/json"
                    },
                    json={
                        "bot_id": bot_id,
                        "faq_question": faq_question,
                        "isVoice": isVoice
                    }
                )
                return response
        
        # Create periodic update messages
        async def periodic_updates():
            if not waiting_callback:
                return
            
            update_messages = [
                "Still searching for the best answer...",
                "Almost there, please bear with me...",
                "Just a moment more, I'm finding the details...",
                "Still working on it, thank you for your patience..."
            ]
            
            for i, message in enumerate(update_messages):
                await asyncio.sleep(5)  # Wait 5 seconds between updates
                if waiting_callback:
                    await waiting_callback(message)
        
        # Run both the request and periodic updates concurrently
        request_task = asyncio.create_task(make_request())
        updates_task = asyncio.create_task(periodic_updates())
        
        # Wait for the request to complete
        response = await request_task
        
        # Cancel the updates task since we got the response
        updates_task.cancel()
        
        if response.status_code == 200:
            data = response.json()
            logger.info(f"✅ FAQ bot response received")
            return {
                "success": True,
                "data": data
            }
        else:
            logger.error(f"❌ FAQ bot error: {response.status_code} - {response.text}")
            return {
                "success": False,
                "error": f"FAQ API returned status {response.status_code}"
            }
                
    except Exception as e:
        logger.error(f"❌ Error calling FAQ bot: {e}")
        return {
            "success": False,
            "error": str(e)
        }