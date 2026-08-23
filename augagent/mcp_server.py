import json
import sys
from typing import List, Callable, Any
from pydantic import BaseModel
from augagent.tools import AugTool

class MCPServer:
    """
    Exposes AugAgent tools as an MCP (Model Context Protocol) Server.
    Allows external agents to connect to this process and consume its tools over stdin/stdout.
    """
    
    def __init__(self, name: str = "augagent-mcp"):
        self.name = name
        self.tools: dict[str, AugTool] = {}
        
    def register_tool(self, tool: AugTool):
        self.tools[tool.name] = tool
        
    def _handle_request(self, req: dict) -> dict:
        method = req.get("method")
        
        if method == "initialize":
            return {
                "protocolVersion": "2024-11-05",
                "capabilities": {"tools": {}},
                "serverInfo": {"name": self.name, "version": "1.0.0"}
            }
            
        elif method == "tools/list":
            tools_list = []
            for name, t in self.tools.items():
                tools_list.append({
                    "name": name,
                    "description": t.description,
                    "inputSchema": {
                        "type": "object",
                        "properties": {
                            "args": {"type": "string", "description": "JSON arguments"}
                        }
                    }
                })
            return {"tools": tools_list}
            
        elif method == "tools/call":
            params = req.get("params", {})
            name = params.get("name")
            args = params.get("arguments", {})
            
            if name not in self.tools:
                return {"isError": True, "content": [{"type": "text", "text": f"Tool {name} not found"}]}
                
            try:
                # We expect the arguments to map to the tool's expected input
                tool = self.tools[name]
                # Simplistic execution for demonstration
                result = tool.execute(**args)  # type: ignore
                return {"content": [{"type": "text", "text": str(result)}]}
            except Exception as e:
                return {"isError": True, "content": [{"type": "text", "text": str(e)}]}
                
        return {"error": {"code": -32601, "message": f"Method {method} not found"}}

    def serve(self):
        """Run the JSON-RPC server over stdio."""
        for line in sys.stdin:
            line = line.strip()
            if not line:
                continue
            try:
                req = json.loads(line)
                req_id = req.get("id")
                result = self._handle_request(req)
                
                res = {"jsonrpc": "2.0", "id": req_id}
                if "error" in result:
                    res["error"] = result["error"]
                else:
                    res["result"] = result
                    
                print(json.dumps(res), flush=True)
            except Exception as e:
                print(json.dumps({"jsonrpc": "2.0", "error": {"code": -32700, "message": str(e)}}), flush=True)
