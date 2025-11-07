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

async def handle_bedrock_knowledge_base_request(
    query_text: str,
    max_results: int = 5,
    waiting_callback = None
) -> Dict[str, Any]:
    """Call Amazon Bedrock Knowledge Base directly (faster than Alive5 API wrapper)"""
    try:
        bedrock_api_key = os.getenv("BEDROCK_API_KEY")
        knowledge_base_id = os.getenv("BEDROCK_KNOWLEDGE_BASE_ID", "2CFKDFOXVH")
        bedrock_endpoint = os.getenv("BEDROCK_ENDPOINT", "https://bedrock-agent-runtime.us-east-1.amazonaws.com")
        
        if not bedrock_api_key:
            logger.warning("⚠️ BEDROCK_API_KEY not set, falling back to Alive5 FAQ API")
            return await handle_faq_bot_request(query_text, isVoice=True, waiting_callback=waiting_callback)
        
        logger.info(f"🔧 Bedrock Knowledge Base request: {query_text}")
        
        if waiting_callback:
            await waiting_callback("Let me check that for you...")
        
        url = f"{bedrock_endpoint}/knowledgebases/{knowledge_base_id}/retrieve"
        
        headers = {
            'Authorization': f'Bearer {bedrock_api_key}',
            'Content-Type': 'application/json',
            'Accept': 'application/json'
        }
        
        payload = {
            "retrievalQuery": {
                "text": query_text
            },
            "retrievalConfiguration": {
                "vectorSearchConfiguration": {
                    "numberOfResults": max_results
                }
            }
        }
        
        async with httpx.AsyncClient(timeout=30.0) as client:
            response = await client.post(url, headers=headers, json=payload)
            
            if response.status_code == 200:
                data = response.json()
                retrieval_results = data.get('retrievalResults', [])
                
                if retrieval_results:
                    # Combine all retrieved content into a single answer
                    # Take the top result(s) and format for voice
                    combined_text = ""
                    for i, result in enumerate(retrieval_results[:3]):  # Use top 3 results
                        content = result.get('content', {}).get('text', '')
                        if content:
                            if i > 0:
                                combined_text += " "  # Add space between results
                            combined_text += content
                    
                    # Format response similar to Alive5 API format for compatibility
                    logger.info(f"✅ Bedrock Knowledge Base response received ({len(retrieval_results)} results)")
                    return {
                        "success": True,
                        "data": {
                            "answer": combined_text.strip(),
                            "urls": [],  # Bedrock doesn't return URLs in this format
                            "source": "bedrock_knowledge_base"
                        }
                    }
                else:
                    logger.info("ℹ️ No results found in Bedrock Knowledge Base")
                    return {
                        "success": True,
                        "data": {
                            "answer": None,
                            "urls": [],
                            "source": "bedrock_knowledge_base"
                        }
                    }
            else:
                logger.error(f"❌ Bedrock API error: {response.status_code} - {response.text}")
                # Fallback to Alive5 API if Bedrock fails
                logger.info("🔄 Falling back to Alive5 FAQ API...")
                return await handle_faq_bot_request(query_text, isVoice=True, waiting_callback=waiting_callback)
                
    except Exception as e:
        logger.error(f"❌ Error calling Bedrock Knowledge Base: {e}")
        # Fallback to Alive5 API if Bedrock fails
        logger.info("🔄 Falling back to Alive5 FAQ API...")
        return await handle_faq_bot_request(query_text, isVoice=True, waiting_callback=waiting_callback)


async def handle_faq_bot_request(
    faq_question: str,
    bot_id: str = FAQ_BOT_ID,
    isVoice: bool = True,
    waiting_callback = None
) -> Dict[str, Any]:
    """Call the Alive5 FAQ bot API (fallback method)"""
    try:
        if not A5_BASE_URL or not A5_API_KEY:
            return {
                "success": False,
                "error": "FAQ API not configured"
            }
        
        logger.info(f"🔧 FAQ bot request (Alive5 API): {faq_question}")
        
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