import pytest
import asyncio
from unittest.mock import AsyncMock, patch, MagicMock
from augagent.mcp_client import MCPToolAdapter, HAS_MCP

@pytest.mark.asyncio
@pytest.mark.skipif(not HAS_MCP, reason="MCP SDK not installed")
async def test_mcp_adapter():
    # We will mock the stdio_client and ClientSession to avoid starting a real process
    mock_session = AsyncMock()
    
    mock_tool = MagicMock()
    mock_tool.name = "fetch_weather"
    mock_tool.description = "Get the weather"
    mock_tool.inputSchema = {
        "type": "object",
        "properties": {
            "city": {"type": "string"}
        },
        "required": ["city"]
    }
    
    mock_session.list_tools.return_value = MagicMock(tools=[mock_tool])
    
    mock_stdio = MagicMock()
    mock_stdio.__aenter__.return_value = (AsyncMock(), AsyncMock())
    
    mock_client_session_ctx = MagicMock()
    mock_client_session_ctx.__aenter__.return_value = mock_session
    
    with patch("augagent.mcp_client.stdio_client", return_value=mock_stdio):
        with patch("augagent.mcp_client.ClientSession", return_value=mock_client_session_ctx):
            adapter = MCPToolAdapter(command="echo", args=["test"])
            tools = await adapter.get_tools()
            
            assert len(tools) == 1
            t = tools[0]
            assert t.name == "fetch_weather"
            assert t.description == "Get the weather"
            # verify schema fields
            schema = t.args_schema.model_json_schema()
            assert "city" in schema["properties"]
