from fastapi import FastAPI, HTTPException, WebSocket, Security, Depends, Request
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from pydantic import BaseModel
from typing import List, Dict, Any, Callable
import asyncio
import json
import os
import jwt
from datetime import datetime

from augagent.models import AgentConfig, TaskResult, LLMConfig
from augagent.agent import AugAgent
from augagent.task import AugTask
from augagent.team import AugTeam

# RBAC Configuration
JWT_SECRET = os.getenv("AUGAGENT_JWT_SECRET", "super-secret-default-key")
JWT_ALGORITHM = "HS256"

security = HTTPBearer()

class UserUser(BaseModel):
    username: str
    roles: List[str]
    tenant_id: str

def get_current_user(credentials: HTTPAuthorizationCredentials = Security(security)) -> UserUser:
    token = credentials.credentials
    try:
        payload = jwt.decode(token, JWT_SECRET, algorithms=[JWT_ALGORITHM])
        return UserUser(
            username=payload.get("sub", ""),
            roles=payload.get("roles", []),
            tenant_id=payload.get("tenant_id", "default")
        )
    except jwt.ExpiredSignatureError:
        raise HTTPException(status_code=401, detail="Token has expired")
    except jwt.PyJWTError:
        raise HTTPException(status_code=401, detail="Invalid token")

def require_role(required_role: str) -> Callable:
    def role_checker(user: UserUser = Depends(get_current_user)):
        if required_role not in user.roles and "admin" not in user.roles:
            raise HTTPException(status_code=403, detail=f"Role '{required_role}' required")
        return user
    return role_checker

app = FastAPI(title="AugAgent API with RBAC", description="REST and WebSocket interfaces for AugAgent orchestration")

# Basic in-memory rate limiting per tenant
tenant_request_counts = {}

@app.middleware("http")
async def tenant_rate_limit_middleware(request: Request, call_next):
    # This is a naive implementation for demonstration purposes
    # In production, use Redis for distributed rate limiting
    auth_header = request.headers.get("Authorization")
    if auth_header and auth_header.startswith("Bearer "):
        token = auth_header.split(" ")[1]
        try:
            payload = jwt.decode(token, JWT_SECRET, algorithms=[JWT_ALGORITHM])
            tenant_id = payload.get("tenant_id", "default")
            
            # Simple rate limiting: 100 reqs per process lifetime (just as an example)
            tenant_request_counts[tenant_id] = tenant_request_counts.get(tenant_id, 0) + 1
            if tenant_request_counts[tenant_id] > 1000:
                return HTTPException(status_code=429, detail="Rate limit exceeded for tenant")
        except:
            pass
            
    response = await call_next(request)
    return response

class TaskRequest(BaseModel):
    description: str
    expected_output: str = ""
    agent_name: str
    async_execution: bool = False

class KickoffRequest(BaseModel):
    agents: List[AgentConfig]
    tasks: List[TaskRequest]
    process: str = "sequential"
    inputs: Dict[str, Any] = {}

@app.post("/kickoff", response_model=List[TaskResult])
async def kickoff(request: KickoffRequest, user: UserUser = Depends(require_role("operator"))):
    """
    Execute a team of agents sequentially or hierarchically.
    Requires 'operator' role.
    """
    try:
        agents_map = {}
        agents = []
        for config in request.agents:
            agent = AugAgent.from_config(config)
            agents_map[agent.name] = agent
            agents.append(agent)
            
        tasks = []
        for tr in request.tasks:
            if tr.agent_name not in agents_map:
                raise HTTPException(status_code=400, detail=f"Agent '{tr.agent_name}' not defined in agents list.")
            task = AugTask(
                description=tr.description,
                expected_output=tr.expected_output,
                agent=agents_map[tr.agent_name],
                async_execution=tr.async_execution
            )
            tasks.append(task)
            
        team = AugTeam(agents=agents, tasks=tasks, process=request.process)
        # Pass tenant ID for namespaced execution context
        request.inputs["tenant_id"] = user.tenant_id
        results = await team.akickoff(request.inputs)
        return results
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.websocket("/stream")
async def websocket_stream(websocket: WebSocket):
    """
    WebSocket endpoint for real-time streaming of an agent's reasoning process.
    Expected message format: {"prompt": "...", "agent_config": {...}}
    Authentication is done via the 'token' query parameter (JWT).
    """
    await websocket.accept()
    
    token = websocket.query_params.get("token")
    if not token:
        await websocket.send_text(json.dumps({"type": "error", "message": "Missing token"}))
        await websocket.close(code=1008)
        return

    try:
        payload = jwt.decode(token, JWT_SECRET, algorithms=[JWT_ALGORITHM])
        roles = payload.get("roles", [])
        if "viewer" not in roles and "admin" not in roles and "operator" not in roles:
            await websocket.send_text(json.dumps({"type": "error", "message": "Insufficient permissions"}))
            await websocket.close(code=1008)
            return
            
        data = await websocket.receive_text()
        req = json.loads(data)
        
        prompt = req.get("prompt", "")
        agent_config_dict = req.get("agent_config", {})
        
        config = AgentConfig.model_validate(agent_config_dict)
        agent = AugAgent.from_config(config)
        
        async def stream_callback(chunk: str):
            await websocket.send_text(json.dumps({"type": "chunk", "content": chunk}))
            
        result = await agent.execute(prompt, stream_callback=stream_callback)
        await websocket.send_text(json.dumps({"type": "result", "output": result.output}))
        
    except jwt.PyJWTError:
        await websocket.send_text(json.dumps({"type": "error", "message": "Invalid token"}))
        await websocket.close(code=1008)
    except Exception as e:
        await websocket.send_text(json.dumps({"type": "error", "message": str(e)}))
    finally:
        await websocket.close()
