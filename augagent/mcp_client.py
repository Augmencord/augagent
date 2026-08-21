"""Model Context Protocol (MCP) Integration for AugAgent."""

import json
from typing import Any, List, Dict
from pydantic import BaseModel, create_model, Field

try:
    from mcp import ClientSession, StdioServerParameters
    from mcp.client.stdio import stdio_client
    HAS_MCP = True
except ImportError:
    HAS_MCP = False

from augagent.tools import AugTool

class MCPToolAdapter:
    """Adapts an MCP Server's tools into native AugTool objects."""
    
    def __init__(self, command: str, args: List[str], env: Dict[str, str] | None = None):
        if not HAS_MCP:
            raise ImportError("The 'mcp' package is required. Install with pip install augagent[mcp]")
        self.server_params = StdioServerParameters(command=command, args=args, env=env)
        
    async def get_tools(self) -> List[AugTool]:
        """Connect to the MCP server, fetch tools, and convert to AugTools."""
        aug_tools = []
        
        # Connect to MCP server over stdio
        async with stdio_client(self.server_params) as (read, write):
            async with ClientSession(read, write) as session:
                await session.initialize()
                
                # Fetch available tools
                result = await session.list_tools()
                
                for tool in result.tools:
                    # Dynamically create a Pydantic model for the tool's arguments based on its JSON schema
                    model_name = f"MCP_{tool.name}_Args"
                    
                    fields = {}
                    if tool.inputSchema and "properties" in tool.inputSchema:
                        for prop_name, prop_def in tool.inputSchema["properties"].items():
                            # Map JSON schema types to Python types. Default to Any if unknown.
                            prop_type = Any
                            if prop_def.get("type") == "string":
                                prop_type = str
                            elif prop_def.get("type") == "integer":
                                prop_type = int
                            elif prop_def.get("type") == "boolean":
                                prop_type = bool
                            elif prop_def.get("type") == "number":
                                prop_type = float
                                
                            is_required = prop_name in tool.inputSchema.get("required", [])
                            default = ... if is_required else None
                            
                            fields[prop_name] = (prop_type, Field(default=default, description=prop_def.get("description", "")))
                            
                    args_schema = create_model(model_name, **fields)
                    
                    # Create the execution wrapper
                    def make_tool_func(t_name):
                        async def _run(**kwargs) -> str:
                            async with stdio_client(self.server_params) as (r, w):
                                async with ClientSession(r, w) as s:
                                    await s.initialize()
                                    call_result = await s.call_tool(t_name, arguments=kwargs)
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
