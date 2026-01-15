"""
Tag Configuration for Voice Agent
Provides predefined tags and loading logic for conversation tagging
"""

import os
import json
import logging
from typing import List, Optional
from pathlib import Path

logger = logging.getLogger(__name__)

# Default tags if none are configured
DEFAULT_TAGS = [
    "Sales",
    "Support",
    "Billing",
    "Technical",
    "General Inquiry",
    "Product Demo",
    "Pricing",
    "Onboarding",
    "Cancellation",
    "Feedback",
    "Account Management",
    "Feature Request",
    "Bug Report",
    "Integration",
    "Training"
]


def load_available_tags() -> List[str]:
    """
    Load available tags from configuration.
    
    Priority:
    1. AVAILABLE_TAGS environment variable (comma-separated)
    2. tags.json file in the same directory
    3. Default tags list
    
    NOTE: Alive5 API fetching is available but not yet implemented.
    Once fetch_tags_from_alive5_api() is implemented, it can be added to the priority list.
    
    Returns:
        List of available tag strings
    """
    # TODO: Once Alive5 API is available, add this as priority 1:
    # try:
    #     api_tags = await fetch_tags_from_alive5_api()
    #     if api_tags:
    #         return api_tags
    # except Exception as e:
    #     logger.warning(f"Could not fetch tags from Alive5 API: {e}")
    
    # Try environment variable first
    env_tags = os.getenv("AVAILABLE_TAGS")
    if env_tags:
        tags = [tag.strip() for tag in env_tags.split(",") if tag.strip()]
        if tags:
            return tags
    
    # Try tags.json file
    config_file = Path(__file__).parent / "tags.json"
    if config_file.exists():
        try:
            with open(config_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
                if isinstance(data, list):
                    tags = [str(tag).strip() for tag in data if tag]
                    if tags:
                        return tags
                elif isinstance(data, dict) and "tags" in data:
                    tags = [str(tag).strip() for tag in data["tags"] if tag]
                    if tags:
                        return tags
        except Exception as e:
            logger.warning(f"Could not load tags from {config_file}: {e}")
    
    # Fall back to default tags
    return DEFAULT_TAGS.copy()


def get_available_tags() -> List[str]:
    """
    Get the list of available tags (cached).
    
    Returns:
        List of available tag strings
    """
    if not hasattr(get_available_tags, '_cached_tags'):
        get_available_tags._cached_tags = load_available_tags()
    return get_available_tags._cached_tags


def reload_tags():
    """
    Reload tags from configuration (clears cache).
    Useful for testing or when configuration changes.
    """
    if hasattr(get_available_tags, '_cached_tags'):
        delattr(get_available_tags, '_cached_tags')
    return get_available_tags()


async def fetch_tags_from_alive5_api(org_name: str = None, api_key: str = None) -> Optional[List[str]]:
    """
    BLUEPRINT: Fetch available tags from Alive5 API.
    
    TODO: Implement this function once Alive5 provides the API endpoint.
    
    This function should:
    1. Make an HTTP request to the Alive5 API to fetch available tags
    2. Parse the response to extract tag names
    3. Return a list of tag strings
    
    Expected API format (to be confirmed):
    - Endpoint: TBD (e.g., GET /api/tags or GET /api/organizations/{org_name}/tags)
    - Authentication: x-a5-apikey header
    - Response format: TBD (likely JSON array or object with tags array)
    
    Args:
        org_name: Organization name (optional, may be needed for API call)
        api_key: Alive5 API key (optional, will use A5_API_KEY env var if not provided)
    
    Returns:
        List of tag strings if successful, None if API is not available or fails
    
    Example implementation (to be updated with actual API details):
    ```python
    import httpx
    from dotenv import load_dotenv
    
    load_dotenv()
    api_key = api_key or os.getenv("A5_API_KEY")
    base_url = os.getenv("A5_BASE_URL", "https://api-stage.alive5.com")
    
    if not api_key:
        logger.warning("A5_API_KEY not configured - cannot fetch tags from API")
        return None
    
    try:
        async with httpx.AsyncClient(timeout=10.0) as client:
            # TODO: Update endpoint once Alive5 provides it
            endpoint = f"{base_url}/api/tags"  # or /api/organizations/{org_name}/tags
            headers = {
                "x-a5-apikey": api_key,
                "Content-Type": "application/json"
            }
            
            response = await client.get(endpoint, headers=headers)
            
            if response.status_code == 200:
                data = response.json()
                # TODO: Parse response format once confirmed
                # Expected formats:
                # - Direct array: ["Sales", "Support", ...]
                # - Object with tags: {"tags": ["Sales", "Support", ...]}
                # - Object with data: {"data": {"tags": [...]}}
                
                tags = []
                if isinstance(data, list):
                    tags = [str(tag) for tag in data if tag]
                elif isinstance(data, dict):
                    if "tags" in data:
                        tags = [str(tag) for tag in data["tags"] if tag]
                    elif "data" in data and isinstance(data["data"], dict) and "tags" in data["data"]:
                        tags = [str(tag) for tag in data["data"]["tags"] if tag]
                
                if tags:
                    logger.info(f"✅ Fetched {len(tags)} tags from Alive5 API")
                    return tags
                else:
                    logger.warning("⚠️ Alive5 API returned empty or invalid tag list")
                    return None
            else:
                logger.warning(f"⚠️ Alive5 API returned status {response.status_code}: {response.text}")
                return None
                
    except Exception as e:
        logger.error(f"❌ Error fetching tags from Alive5 API: {e}", exc_info=True)
        return None
    ```
    """
    # TODO: Implement this function once Alive5 provides the API endpoint
    # For now, return None to indicate API is not available
    logger.debug("🏷️ fetch_tags_from_alive5_api() called - API not yet implemented")
    return None

