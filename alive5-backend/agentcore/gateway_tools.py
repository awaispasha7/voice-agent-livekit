"""
AgentCore Gateway Tools
Moves function tools to AgentCore Gateway for better management
"""

import os
import logging
from typing import Dict, Any, Optional
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables
env_path = Path(__file__).parent.parent.parent / ".env"
if env_path.exists():
    load_dotenv(env_path)

logger = logging.getLogger("agentcore-gateway")

try:
    from bedrock_agentcore.gateway import GatewayClient, GatewayTool
    AGENTCORE_GATEWAY_AVAILABLE = True
except ImportError:
    GatewayClient = None
    GatewayTool = None
    AGENTCORE_GATEWAY_AVAILABLE = False
    logger.warning("bedrock-agentcore gateway not available. Will use direct function calls.")

def _gateway_tool(fn):
    """Decorator that becomes a no-op when AgentCore Gateway isn't installed/enabled."""
    if AGENTCORE_GATEWAY_AVAILABLE and GatewayTool:
        return GatewayTool(fn)
    return fn


# Import existing function handlers
import sys
sys.path.insert(0, str(Path(__file__).parent.parent / "alive5-worker"))
from functions import (
    handle_load_bot_flows,
    handle_faq_bot_request,
    handle_bedrock_knowledge_base_request
)


@_gateway_tool
async def load_bot_flows(
    botchain_name: str,
    org_name: str = "alive5stage0"
) -> Dict[str, Any]:
    """
    Load bot flow definitions dynamically
    
    Args:
        botchain_name: Name of the botchain
        org_name: Organization name
        
    Returns:
        Bot flow template data
    """
    return await handle_load_bot_flows(botchain_name, org_name)


@_gateway_tool
async def faq_bot_request(
    query_text: str,
    faq_bot_id: Optional[str] = None,
    org_name: Optional[str] = None,
    is_voice: bool = True
) -> Dict[str, Any]:
    """
    Query FAQ bot for company/service information
    
    Args:
        query_text: User's question
        faq_bot_id: Optional FAQ bot ID
        org_name: Optional organization name
        is_voice: Whether this is a voice interaction
        
    Returns:
        FAQ bot response
    """
    return await handle_faq_bot_request(
        query_text,
        bot_id=faq_bot_id,
        org_name=org_name,
        isVoice=is_voice
    )


@_gateway_tool
async def bedrock_knowledge_base_query(
    query_text: str,
    max_results: int = 5,
    faq_bot_id: Optional[str] = None,
    org_name: Optional[str] = None
) -> Dict[str, Any]:
    """
    Query Bedrock Knowledge Base directly
    
    Args:
        query_text: User's question
        max_results: Maximum number of results
        faq_bot_id: Optional FAQ bot ID for filtering
        org_name: Optional organization name for filtering
        
    Returns:
        Knowledge base results
    """
    return await handle_bedrock_knowledge_base_request(
        query_text,
        max_results=max_results,
        faq_bot_id=faq_bot_id,
        org_name=org_name
    )


@_gateway_tool
async def transfer_call_to_human(
    room_name: str,
    transfer_number: Optional[str] = None
) -> Dict[str, Any]:
    """
    Transfer a phone call to a human agent or phone number
    
    Args:
        room_name: The LiveKit room name (session identifier)
        transfer_number: Phone number to transfer to (optional, uses default if not provided)
        
    Returns:
        Transfer status and message
    """
    import httpx
    from urllib.parse import quote
    
    backend_url = os.getenv("BACKEND_URL", "http://18.210.238.67")
    
    # Get session data to retrieve call_control_id
    encoded_room_name = quote(room_name, safe='')
    async with httpx.AsyncClient(timeout=5.0) as client:
        response = await client.get(f"{backend_url}/api/sessions/{encoded_room_name}")
        if response.status_code != 200:
            return {
                "success": False,
                "message": "Unable to retrieve session data for transfer."
            }
        
        session_data = response.json()
        call_control_id = session_data.get("call_control_id")
        source = session_data.get("user_data", {}).get("source")
        
        # Check if this is a web session
        if not call_control_id or source != "telnyx_phone":
            return {
                "success": False,
                "is_web_session": True,
                "message": "Call transfers are only available for phone calls, not through this web interface."
            }
        
        # Get transfer number or use default
        if not transfer_number:
            transfer_number = os.getenv("TELNYX_CALL_CENTER_NUMBER")
        
        if not transfer_number:
            return {
                "success": False,
                "message": "Call transfers are not currently configured."
            }
        
        # Call backend transfer endpoint
        transfer_response = await client.post(
            f"{backend_url}/api/telnyx/transfer",
            json={
                "room_name": room_name,
                "call_control_id": call_control_id,
                "transfer_to": transfer_number
            },
            timeout=15.0
        )
        
        if transfer_response.status_code == 200:
            return {
                "success": True,
                "is_web_session": False,
                "message": "Transfer will happen after you speak the acknowledgment message."
            }
        else:
            return {
                "success": False,
                "message": "Unable to process transfer request."
            }


@_gateway_tool
async def save_collected_data(
    room_name: str,
    field_name: str,
    value: str
) -> Dict[str, Any]:
    """
    Save user response to collected_data and update CRM in real-time
    
    Args:
        room_name: The LiveKit room name (session identifier)
        field_name: The field name (e.g., "full_name", "email", "phone", "notes_entry")
        value: The user's response to save
        
    Returns:
        Success status and message
    """
    def _normalize_field_name(name: str) -> str:
        n = (name or "").strip()
        n_l = n.lower().replace("-", "_").replace(" ", "_")
        if n_l in {"fullname", "full_name", "name"}:
            return "full_name"
        if n_l in {"first", "first_name", "firstname"}:
            return "first_name"
        if n_l in {"last", "last_name", "lastname"}:
            return "last_name"
        if n_l in {"email", "email_address"}:
            return "email"
        if n_l in {"phone", "phone_number", "phone_mobile", "phonemobile"}:
            return "phone"
        if n_l in {"notes_entry", "notes", "note"}:
            return "notes_entry"
        if n_l in {"accountid", "account_id", "account"}:
            return "account_id"
        if n_l in {"company"}:
            return "company"
        if n_l in {"companytitle", "company_title", "title", "company_position"}:
            return "company_title"
        return n_l or n
    
    def _normalize_phone_number(phone: str) -> str:
        """Normalize US phone numbers to include +1 prefix"""
        import re
        if not phone:
            return phone
        
        # Remove all non-digit characters except +
        digits_only = re.sub(r'[^\d+]', '', phone)
        
        # If it already starts with +1, return as is
        if digits_only.startswith('+1'):
            return digits_only
        
        # If it starts with 1 (without +), add +
        if digits_only.startswith('1') and len(digits_only) == 11:
            return '+' + digits_only
        
        # If it's 10 digits (US number without country code), add +1
        if len(digits_only) == 10:
            return '+1' + digits_only
        
        # If it's 11 digits starting with 1, add +
        if len(digits_only) == 11 and digits_only[0] == '1':
            return '+' + digits_only
        
        # Return original if we can't normalize
        return phone

    import httpx
    from urllib.parse import quote
    
    backend_url = os.getenv("BACKEND_URL", "http://18.210.238.67")
    normalized_field = _normalize_field_name(field_name)
    
    # Get current session data
    encoded_room_name = quote(room_name, safe='')
    async with httpx.AsyncClient(timeout=5.0) as client:
        response = await client.get(f"{backend_url}/api/sessions/{encoded_room_name}")
        if response.status_code != 200:
            return {
                "success": False,
                "message": "Unable to retrieve session data."
            }
        
        session_data = response.json()
        user_data = session_data.get("user_data", {})
        collected_data = user_data.get("collected_data", {})
        
        # Update collected_data based on field_name
        if normalized_field == "full_name":
            collected_data["full_name"] = value
            name_parts = value.strip().split(' ', 1)
            first_name = name_parts[0] if name_parts else ""
            last_name = name_parts[1] if len(name_parts) > 1 else ""
            # Also store split name fields in collected_data for consistency
            if first_name:
                collected_data["first_name"] = first_name
            if last_name:
                collected_data["last_name"] = last_name
            # Update collected_data in user_data before setting individual fields
            user_data["collected_data"] = collected_data
            # Update session with first_name and last_name for CRM
            user_data["first_name"] = first_name
            user_data["last_name"] = last_name
        elif normalized_field == "first_name":
            collected_data["first_name"] = value
            user_data["first_name"] = value
            user_data["collected_data"] = collected_data
        elif normalized_field == "last_name":
            collected_data["last_name"] = value
            user_data["last_name"] = value
            user_data["collected_data"] = collected_data
        elif normalized_field == "email":
            # Ensure email is properly saved to both collected_data and user_data
            collected_data["email"] = value
            user_data["collected_data"] = collected_data
            user_data["email"] = value
        elif normalized_field == "phone":
            # Normalize phone number to include +1 prefix for US numbers
            normalized_phone = _normalize_phone_number(value)
            collected_data["phone"] = normalized_phone
            user_data["collected_data"] = collected_data
            user_data["phone"] = normalized_phone
        elif normalized_field == "account_id":
            collected_data["account_id"] = value
            user_data["collected_data"] = collected_data
            user_data["account_id"] = value
        elif normalized_field == "company":
            collected_data["company"] = value
            user_data["collected_data"] = collected_data
            user_data["company"] = value
        elif normalized_field == "company_title":
            collected_data["company_title"] = value
            user_data["collected_data"] = collected_data
            user_data["company_title"] = value
        elif normalized_field == "notes_entry":
            if "notes_entry" not in collected_data:
                collected_data["notes_entry"] = []
            collected_data["notes_entry"].append(value)
            notes_str = " | ".join(collected_data.get("notes_entry", []))
            user_data["collected_data"] = collected_data
            user_data["notes"] = notes_str
        else:
            return {
                "success": False,
                "message": f"Unknown field_name: {field_name}"
            }
        
        # Update session via backend API
        update_response = await client.post(
            f"{backend_url}/api/sessions/update",
            json={
                "room_name": room_name,
                "user_data": user_data
            }
        )
        
        if update_response.status_code == 200:
            return {
                "success": True,
                "message": f"Data for {normalized_field} saved successfully and CRM updated in real-time."
            }
        else:
            return {
                "success": False,
                "message": "Error saving data to session."
            }


class AgentCoreGateway:
    """AgentCore Gateway client for tool access"""
    
    def __init__(self):
        self.enabled = os.getenv("USE_AGENTCORE_GATEWAY", "true").lower() == "true"
        self.gateway_client = None
        
        if self.enabled and AGENTCORE_GATEWAY_AVAILABLE:
            try:
                self.gateway_client = GatewayClient()
                logger.info("✅ AgentCore Gateway enabled")
            except Exception as e:
                logger.warning(f"⚠️ Failed to initialize AgentCore Gateway: {e}")
                self.enabled = False
        else:
            logger.info("ℹ️ AgentCore Gateway disabled - using direct function calls")
    
    async def call_tool(
        self,
        tool_name: str,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Call a tool through AgentCore Gateway
        
        Args:
            tool_name: Name of the tool to call
            **kwargs: Tool arguments
            
        Returns:
            Tool response
        """
        if not self.enabled or not self.gateway_client:
            # Fallback to direct function call
            return await self._direct_function_call(tool_name, **kwargs)
        
        try:
            result = await self.gateway_client.invoke_tool(
                tool_name=tool_name,
                **kwargs
            )
            return result
        except Exception as e:
            logger.error(f"Error calling tool via Gateway: {e}")
            # Fallback to direct function call
            return await self._direct_function_call(tool_name, **kwargs)
    
    async def _direct_function_call(
        self,
        tool_name: str,
        **kwargs
    ) -> Dict[str, Any]:
        """Fallback to direct function calls"""
        if tool_name == "load_bot_flows":
            return await handle_load_bot_flows(
                kwargs.get("botchain_name"),
                kwargs.get("org_name", "alive5stage0")
            )
        elif tool_name == "faq_bot_request":
            return await handle_faq_bot_request(
                kwargs.get("query_text"),
                bot_id=kwargs.get("faq_bot_id"),
                org_name=kwargs.get("org_name"),
                isVoice=kwargs.get("is_voice", True)
            )
        elif tool_name == "bedrock_knowledge_base_query":
            return await handle_bedrock_knowledge_base_request(
                kwargs.get("query_text"),
                max_results=kwargs.get("max_results", 5),
                faq_bot_id=kwargs.get("faq_bot_id"),
                org_name=kwargs.get("org_name")
            )
        elif tool_name == "transfer_call_to_human":
            # Call the Gateway tool function directly (defined above)
            return await transfer_call_to_human(
                kwargs.get("room_name"),
                kwargs.get("transfer_number")
            )
        elif tool_name == "save_collected_data":
            # Call the Gateway tool function directly (defined above)
            return await save_collected_data(
                kwargs.get("room_name"),
                kwargs.get("field_name"),
                kwargs.get("value")
            )
        else:
            return {
                "success": False,
                "error": f"Unknown tool: {tool_name}"
            }
    
    def is_enabled(self) -> bool:
        """Check if AgentCore Gateway is enabled"""
        return self.enabled

