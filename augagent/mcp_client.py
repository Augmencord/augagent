"""Model Context Protocol (MCP) Integration for AugAgent."""

import json
import asyncio
from typing import Any, List, Dict, Optional
from pydantic import BaseModel, create_model, Field
from contextlib import AsyncExitStack

try:
    from mcp import ClientSession, StdioServerParameters
    from mcp.client.stdio import stdio_client
    from mcp.client.sse import sse_client
    HAS_MCP = True
except ImportError:
    HAS_MCP = False

from augagent.tools import AugTool

class MCPToolAdapter:
    """Adapts an MCP Server's tools into native AugTool objects with a persistent session."""
    
    def __init__(
        self, 
        command: Optional[str] = None, 
        args: Optional[List[str]] = None, 
        env: Optional[Dict[str, str]] = None,
        sse_url: Optional[str] = None
    ):
        if not HAS_MCP:
            raise ImportError("The 'mcp' package is required. Install with pip install augagent[mcp]")
            
        self.command = command
        self.args = args or []
        self.env = env
        self.sse_url = sse_url
        
        self._exit_stack = AsyncExitStack()
        self._session: Optional[ClientSession] = None

    async def connect(self):
        """Establish a persistent connection to the MCP server."""
        if self._session:
            return

        if self.sse_url:
            transport = await self._exit_stack.enter_async_context(sse_client(self.sse_url))
        elif self.command:
            params = StdioServerParameters(command=self.command, args=self.args, env=self.env)
            transport = await self._exit_stack.enter_async_context(stdio_client(params))
        else:
            raise ValueError("Must provide either sse_url or command")

        read, write = transport
        self._session = await self._exit_stack.enter_async_context(ClientSession(read, write))
        await self._session.initialize()

    async def disconnect(self):
        """Close the persistent connection."""
        await self._exit_stack.aclose()
        self._session = None
        
    async def get_tools(self) -> List[AugTool]:
        """Fetch tools from the connected MCP server and convert to AugTools."""
        if not self._session:
            await self.connect()
            
        assert self._session is not None
            
        aug_tools = []
        result = await self._session.list_tools()
        
        for tool in result.tools:
            model_name = f"MCP_{tool.name}_Args"
            
            fields: Dict[str, Any] = {}
            if tool.inputSchema and "properties" in tool.inputSchema:
                for prop_name, prop_def in tool.inputSchema["properties"].items():
                    prop_type: Any = Any
                    t = prop_def.get("type")
                    if t == "string":
                        prop_type = str
                    elif t == "integer":
                        prop_type = int
                    elif t == "boolean":
                        prop_type = bool
                    elif t == "number":
                        prop_type = float
                    elif t == "array":
                        prop_type = list
                    elif t == "object":
                        prop_type = dict
                        
                    is_required = prop_name in tool.inputSchema.get("required", [])
                    default = ... if is_required else None
                    
                    fields[prop_name] = (prop_type, Field(default=default, description=prop_def.get("description", "")))
                    
            args_schema = create_model(model_name, **fields)
            
            def make_tool_func(t_name):
                async def _run(**kwargs) -> str:
                    if not self._session:
                        await self.connect()
                    assert self._session is not None
                    call_result = await self._session.call_tool(t_name, arguments=kwargs)
                    return json.dumps([c.model_dump() for c in call_result.content])
                return _run

            aug_tool = AugTool(
                name=tool.name,
                description=tool.description or "",
                args_schema=args_schema,
                func=make_tool_func(tool.name)
            )
            aug_tools.append(aug_tool)
            
        return aug_tools

    async def __aenter__(self):
        await self.connect()
        return self
        
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        await self.disconnect()
