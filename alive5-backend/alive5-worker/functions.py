"""
Function Definitions and Handlers for Voice Agent
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
    """Load bot flow definitions dynamically"""
    try:
        if not A5_BASE_URL or not A5_API_KEY:
            return {
                "success": False,
                "error": "Missing API configuration"
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
    waiting_callback = None,
    faq_bot_id: str = None,
    org_name: str = None
) -> Dict[str, Any]:
    """Call Amazon Bedrock Knowledge Base directly using boto3 (faster than FAQ API wrapper)
    
    Args:
        query_text: The user's question
        max_results: Maximum number of results to return
        waiting_callback: Optional callback to notify user while waiting
        faq_bot_id: FAQ bot ID to filter by (orgbot_name in metadata). If None, no filter is applied.
        org_name: Organization name to filter by (org_name in metadata). Optional, can be used with faq_bot_id.
    """
    try:
        # Check if boto3 is available
        try:
            import boto3
            from botocore.exceptions import ClientError, BotoCoreError
        except ImportError:
            logger.warning("⚠️ boto3 not installed, falling back to FAQ API")
            logger.info("   Install with: pip install boto3")
            return await handle_faq_bot_request(query_text, isVoice=True, waiting_callback=waiting_callback)
        
        knowledge_base_id = os.getenv("BEDROCK_KNOWLEDGE_BASE_ID", "2CFKDFOXVH")
        
        # Get AWS credentials from environment
        aws_access_key_id = os.getenv("AWS_ACCESS_KEY_ID")
        aws_secret_access_key = os.getenv("AWS_SECRET_ACCESS_KEY")
        aws_region = os.getenv("AWS_DEFAULT_REGION", "us-east-1")
        
        if not aws_access_key_id or not aws_secret_access_key:
            logger.warning("⚠️ AWS credentials not set, falling back to FAQ API")
            logger.info("   Set AWS_ACCESS_KEY_ID and AWS_SECRET_ACCESS_KEY in .env")
            return await handle_faq_bot_request(query_text, bot_id=faq_bot_id, isVoice=True, waiting_callback=waiting_callback)
        
        logger.info("=" * 80)
        logger.info(f"🔧 Bedrock Knowledge Base request: {query_text}")
        logger.info("=" * 80)
        # if faq_bot_id:
        #     logger.info(f"   Filtering by FAQ bot ID (orgbot_name): {faq_bot_id}")
        # if org_name:
        #     logger.info(f"   Filtering by org_name: {org_name}")
        
        # if waiting_callback:
        #     await waiting_callback("Let me check that for you...")
        
        # Create Bedrock Runtime client with explicit credentials
        # boto3 will use these credentials and handle AWS Signature V4 automatically
        bedrock_runtime = boto3.client(
            'bedrock-agent-runtime',
            region_name=aws_region,
            aws_access_key_id=aws_access_key_id,
            aws_secret_access_key=aws_secret_access_key
        )
        
        # Build retrieval configuration with optional filters
        retrieval_config = {
            'vectorSearchConfiguration': {
                'numberOfResults': max_results
            }
        }
        
        # Add filter if FAQ bot ID is provided
        # According to AWS docs: https://docs.aws.amazon.com/bedrock/latest/APIReference/API_agent-runtime_RetrievalFilter.html
        # We can filter by metadata attributes using RetrievalFilter
        if faq_bot_id or org_name:
            filters = []
            
            # Filter by orgbot_name (FAQ bot ID) if provided
            if faq_bot_id:
                # Ensure FAQ bot ID is in the correct format (with "faq_" prefix if needed)
                orgbot_name = faq_bot_id if faq_bot_id.startswith("faq_") else f"faq_{faq_bot_id}"
                filters.append({
                    'equals': {
                        'key': 'orgbot_name',
                        'value': orgbot_name
                    }
                })
            
            # Filter by org_name if provided (can be combined with orgbot_name using andAll)
            if org_name:
                filters.append({
                    'equals': {
                        'key': 'org_name',
                        'value': org_name
                    }
                })
            
            # If multiple filters, combine them with andAll
            if len(filters) == 1:
                retrieval_config['vectorSearchConfiguration']['filter'] = filters[0]
            elif len(filters) > 1:
                retrieval_config['vectorSearchConfiguration']['filter'] = {
                    'andAll': filters
                }
        
        # Call the retrieve API
        response = bedrock_runtime.retrieve(
            knowledgeBaseId=knowledge_base_id,
            retrievalQuery={
                'text': query_text
            },
            retrievalConfiguration=retrieval_config
        )
        
        # Process results
        retrieval_results = response.get('retrievalResults', [])
        
        if retrieval_results:
            # Log metadata for each result
            logger.info(f"✅ Bedrock Knowledge Base response received ({len(retrieval_results)} results)")
            # logger.info(f"📊 Results metadata:")
            
            for i, result in enumerate(retrieval_results, 1):
                metadata = result.get('metadata', {})
                score = result.get('score', 0)
                orgbot_name = metadata.get('orgbot_name', 'N/A')
                org_name_meta = metadata.get('org_name', 'N/A')
                reference_url = metadata.get('reference_url', 'N/A')
                
                # logger.info(f"   Result {i}:")
                # logger.info(f"      - Score: {score:.4f}")
                # logger.info(f"      - orgbot_name: {orgbot_name}")
                # logger.info(f"      - org_name: {org_name_meta}")
                # logger.info(f"      - reference_url: {reference_url}")
                
                # # Log if filter matched (if filters were applied)
                # if faq_bot_id:
                #     expected_orgbot = faq_bot_id if faq_bot_id.startswith("faq_") else f"faq_{faq_bot_id}"
                #     if orgbot_name == expected_orgbot:
                #         logger.info(f"      ✅ Filter match: orgbot_name matches filter")
                #     else:
                #         logger.warning(f"      ⚠️ Filter mismatch: expected '{expected_orgbot}', got '{orgbot_name}'")
                
                # if org_name and org_name_meta:
                #     if org_name_meta == org_name:
                #         logger.info(f"      ✅ Filter match: org_name matches filter")
                #     else:
                #         logger.warning(f"      ⚠️ Filter mismatch: expected '{org_name}', got '{org_name_meta}'")
            
            # Combine all retrieved content into a single answer
            # Take the top result(s) and format for voice
            combined_text = ""
            for i, result in enumerate(retrieval_results[:3]):  # Use top 3 results
                content = result.get('content', {}).get('text', '')
                if content:
                    if i > 0:
                        combined_text += " "  # Add space between results
                    combined_text += content
            
            # Log the actual response content (truncated if too long)
            response_preview = combined_text.strip()
            # if len(response_preview) > 500:
            #     logger.info(f"📄 Response content (first 500 chars): {response_preview[:500]}...")
            # else:
            #     logger.info(f"📄 Response content: {response_preview}")

            logger.info(f"📄 Response content: {response_preview}")
            
            # Log individual results summary
            # for i, result in enumerate(retrieval_results[:3], 1):
            #     content = result.get('content', {}).get('text', '')
            #     score = result.get('score', 0)
            #     if content:
            #         content_preview = content[:200] + "..." if len(content) > 200 else content
            #         logger.info(f"   Result {i} (score: {score:.4f}): {content_preview}")
            
            # Note: The combined_text may contain raw RAG data with metadata, timestamps, IDs, etc.
            # The LLM will process and summarize this in the system prompt
            return {
                "success": True,
                "data": {
                    "answer": combined_text.strip(),
                    "urls": [],  # Bedrock doesn't return URLs in this format
                    "source": "bedrock_knowledge_base",
                    "needs_summarization": True  # Flag to indicate this is RAG data that needs processing
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
            
    except ClientError as e:
        error_code = e.response['Error']['Code']
        error_message = e.response['Error']['Message']
        logger.error(f"❌ Bedrock API error ({error_code}): {error_message}")
        # Fallback to FAQ API if Bedrock fails
        logger.info("🔄 Falling back to FAQ API...")
        return await handle_faq_bot_request(query_text, bot_id=faq_bot_id, isVoice=True, waiting_callback=waiting_callback)
        
    except BotoCoreError as e:
        logger.error(f"❌ boto3 error: {e}")
        # Fallback to FAQ API if boto3 fails
        logger.info("🔄 Falling back to FAQ API...")
        return await handle_faq_bot_request(query_text, bot_id=faq_bot_id, isVoice=True, waiting_callback=waiting_callback)
        
    except Exception as e:
        logger.error(f"❌ Unexpected error calling Bedrock Knowledge Base: {e}")
        import traceback
        logger.error(f"   Traceback: {traceback.format_exc()}")
        # Fallback to FAQ API if anything else fails
        logger.info("🔄 Falling back to FAQ API...")
        return await handle_faq_bot_request(query_text, bot_id=faq_bot_id, isVoice=True, waiting_callback=waiting_callback)


async def handle_faq_bot_request(
    faq_question: str,
    bot_id: str = FAQ_BOT_ID,
    isVoice: bool = True,
    waiting_callback = None
) -> Dict[str, Any]:
    """Call the FAQ bot API (fallback method)"""
    try:
        if not A5_BASE_URL or not A5_API_KEY:
            return {
                "success": False,
                "error": "FAQ API not configured"
            }
        
        logger.info(f"🔧 FAQ bot request: {faq_question}")
        
        if waiting_callback:
            await waiting_callback("Please wait while I fetch the information...")
        
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