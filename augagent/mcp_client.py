"""Model Context Protocol (MCP) Integration for AugAgent."""

import json
import asyncio
import logging
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
        sse_url: Optional[str] = None,
        name: str = "mcp"
    ):
        if not HAS_MCP:
            raise ImportError("The 'mcp' package is required. Install with pip install augagent[mcp]")
            
        self.command = command
        self.args = args or []
        self.env = env
        self.sse_url = sse_url
        self.name = name
        
        self._exit_stack = AsyncExitStack()
        self._session: Optional[ClientSession] = None

    async def connect(self):
        """Establish a persistent connection to the MCP server with backoff."""
        if self._session:
            return

        for attempt in range(3):
            try:
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
                return
            except Exception as e:
                logging.error(f"Failed to connect to MCP server {self.name}: {e}. Retrying...")
                await asyncio.sleep(2 ** attempt)
        raise ConnectionError(f"Failed to connect to MCP server {self.name} after 3 attempts.")

    async def disconnect(self):
        """Close the persistent connection."""
        await self._exit_stack.aclose()
        self._session = None

    async def health_check(self) -> bool:
        """Ping the MCP server to ensure it is healthy."""
        try:
            if not self._session:
                await self.connect()
            # Simple check: list tools to see if it responds
            if self._session:
                await self._session.list_tools()
            return True
        except Exception as e:
            logging.error(f"MCP server {self.name} health check failed: {e}")
            await self.disconnect()
            return False
            
    async def get_tools(self) -> List[AugTool]:
        """Fetch tools from the connected MCP server and convert to AugTools."""
        if not await self.health_check():
            logging.warning(f"Skipping tools for unhealthy MCP server: {self.name}")
            return []
            
        assert self._session is not None
            
        aug_tools = []
        result = await self._session.list_tools()
        
        for tool in result.tools:
            tool_name_namespaced = f"{self.name}.{tool.name}"
            model_name = f"MCP_{self.name}_{tool.name}_Args".replace("-", "_")
            
            fields: Dict[str, Any] = {}
            input_schema: Dict[str, Any] = getattr(tool, "inputSchema", {}) or getattr(tool, "input_schema", {})
            if input_schema and "properties" in input_schema:
                for prop_name, prop_def in input_schema["properties"].items():
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
                        
                    is_required = prop_name in input_schema.get("required", [])
                    default = ... if is_required else None
                    
                    fields[prop_name] = (prop_type, Field(default=default, description=prop_def.get("description", "")))
                    
            args_schema = create_model(model_name, **fields)
            
            def make_tool_func(t_name):
                async def _run(**kwargs) -> str:
                    if not await self.health_check():
                        return json.dumps({"error": f"MCP server {self.name} is currently unavailable."})
                    assert self._session is not None
                    call_result = await self._session.call_tool(t_name, arguments=kwargs)
                    return json.dumps([c.model_dump() for c in call_result.content])
                return _run

            aug_tool = AugTool(
                name=tool_name_namespaced,
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


class MultiMCPClient:
    """Manages multiple MCP server connections."""
    def __init__(self, adapters: List[MCPToolAdapter]):
        self.adapters = adapters

    async def get_all_tools(self) -> List[AugTool]:
        """Fetch tools from all configured MCP servers."""
        all_tools = []
        for adapter in self.adapters:
            try:
                tools = await adapter.get_tools()
                all_tools.extend(tools)
            except Exception as e:
                logging.error(f"Failed to get tools from adapter {adapter.name}: {e}")
        return all_tools
