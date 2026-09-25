from fastapi import FastAPI, HTTPException, WebSocket, Security, Depends, Request, BackgroundTasks
from fastapi.responses import JSONResponse, StreamingResponse
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from pydantic import BaseModel, Field
from typing import List, Dict, Any, Callable, Optional
import asyncio
import json
import os
import jwt
import time
import uuid
from datetime import datetime

try:
    from prometheus_client import generate_latest, CONTENT_TYPE_LATEST, Counter, Histogram, Gauge  # type: ignore
    HAS_PROMETHEUS = True
except ImportError:
    HAS_PROMETHEUS = False

from augagent.models import AgentConfig, TaskResult, LLMConfig
from augagent.agent import AugAgent
from augagent.task import AugTask
from augagent.team import AugTeam
from augagent.audit import audit
from augagent.cache import rate_limiter
from augagent.execution_store import execution_store, ExecutionStatus
from augagent.tenant_store import tenant_store
from augagent.tools import DelegateWorkTool
from augagent.tools_file import view_file, create_file, list_directory, replace_file_content
from augagent.tools_terminal import run_terminal_command, list_open_windows
from augagent.jsonl_checkpointer import JsonlCheckpointer

# RBAC Configuration
JWT_SECRET = os.getenv("AUGAGENT_JWT_SECRET", "super-secret-default-key")
JWT_ALGORITHM = "HS256"

security = HTTPBearer()

class UserUser(BaseModel):
    username: str
    roles: List[str]
    tenant_id: str
    permissions: List[str] = Field(default_factory=list)

def get_current_user(credentials: HTTPAuthorizationCredentials = Security(security)) -> UserUser:
    token = credentials.credentials
    try:
        payload = jwt.decode(token, JWT_SECRET, algorithms=[JWT_ALGORITHM])
        return UserUser(
            username=payload.get("sub", ""),
            roles=payload.get("roles", []),
            tenant_id=payload.get("tenant_id", "default"),
            permissions=payload.get("permissions", [])
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

def require_permission(resource: str, action: str):
    def permission_checker(user: UserUser = Depends(get_current_user)):
        if "admin" in user.roles:
            return user
        perm = f"{resource}:{action}"
        if perm not in user.permissions:
            raise HTTPException(status_code=403, detail=f"Permission '{perm}' required.")
        return user
    return permission_checker

app = FastAPI(title="AugAgent API with RBAC", description="REST and WebSocket interfaces for AugAgent orchestration")

RATE_LIMIT_PER_MINUTE = int(os.getenv("AUGAGENT_RATE_LIMIT_PER_MINUTE", "100"))

@app.middleware("http")
async def tenant_rate_limit_middleware(request: Request, call_next):
    auth_header = request.headers.get("Authorization")
    if auth_header and auth_header.startswith("Bearer "):
        token = auth_header.split(" ")[1]
        try:
            payload = jwt.decode(token, JWT_SECRET, algorithms=[JWT_ALGORITHM])
            tenant_id = payload.get("tenant_id", "default")
            roles = payload.get("roles", [])
            
            # Admin role bypasses rate limit (or gets higher limit)
            if "admin" not in roles:
                endpoint = request.url.path
                if await rate_limiter.is_rate_limited(tenant_id, endpoint, RATE_LIMIT_PER_MINUTE, 60):
                    return JSONResponse(
                        status_code=429, 
                        content={"error": "RateLimitExceeded", "detail": "Rate limit exceeded for tenant", "code": 429}
                    )
        except Exception:
            pass
            
    response = await call_next(request)
    return response

class TaskRequest(BaseModel):
    description: str
    expected_output: str = ""
    agent_name: Optional[str] = None
    async_execution: bool = False

class KickoffRequest(BaseModel):
    agents: List[AgentConfig]
    tasks: List[TaskRequest]
    process: str = "sequential"
    inputs: Dict[str, Any] = {}
    
class ExecuteRequest(BaseModel):
    prompt: str
    inputs: Dict[str, Any] = {}

class AgentStreamRequest(BaseModel):
    prompt: str
    agent_id: Optional[str] = None
    agent_config: Optional[AgentConfig] = None
    inputs: Dict[str, Any] = Field(default_factory=dict)

# Global in-memory storage for demo purposes
active_agents: Dict[str, Any] = {}
active_executions: Dict[str, Any] = {}

@app.post("/api/v1/agents", response_model=Dict[str, Any])
async def create_agent(config: AgentConfig, user: UserUser = Depends(require_role("operator"))):
    """Create an agent instance."""
    agent = AugAgent.from_config(config)
    active_agents[agent.id] = agent
    return {"status": "success", "agent_id": agent.id}


@app.post("/kickoff", response_model=Dict[str, Any])
async def kickoff(request: KickoffRequest, user: UserUser = Depends(get_current_user)):
    """Trigger team execution with validation."""
    return {"status": "success", "message": "Kickoff accepted"}


@app.post("/v1/agent/stream")
@app.post("/api/v1/agent/stream")
async def stream_agent_execution(request: AgentStreamRequest, req: Request):
    """
    POST /v1/agent/stream with SSE response.
    Events: token, tool_call, done
    """
    auth_header = req.headers.get("Authorization")
    if auth_header and auth_header.startswith("Bearer "):
        token = auth_header.split(" ")[1]
        try:
            jwt.decode(token, JWT_SECRET, algorithms=[JWT_ALGORITHM])
        except jwt.PyJWTError:
            raise HTTPException(status_code=401, detail="Invalid token")

    agent: AugAgent | None = None
    if request.agent_id:
        agent = active_agents.get(request.agent_id)
        if not agent:
            raise HTTPException(status_code=404, detail=f"Agent '{request.agent_id}' not found")
    elif request.agent_config:
        agent = AugAgent.from_config(request.agent_config)
    else:
        agent = AugAgent(
            name="StreamingAssistant",
            role="AI Assistant",
            goal="Assist user with code and queries",
            llm_config=LLMConfig(),
        )

    async def sse_generator():
        queue: asyncio.Queue[tuple[str, dict[str, Any]] | None] = asyncio.Queue()

        async def stream_callback(event: Any):
            if isinstance(event, str):
                await queue.put(("token", {"token": event, "content": event}))
            elif isinstance(event, dict):
                event_type = event.get("type", "")
                if event_type in ("tool_call_start", "tool_call_end", "tool_call"):
                    payload = {
                        "tool": event.get("tool_name") or event.get("tool") or event.get("name", ""),
                        "name": event.get("tool_name") or event.get("tool") or event.get("name", ""),
                        "arguments": event.get("tool_args") or event.get("arguments", {}),
                        "result": event.get("result"),
                        "status": "start" if event_type == "tool_call_start" else ("completed" if event_type == "tool_call_end" else event.get("status", "call")),
                    }
                    await queue.put(("tool_call", payload))
                elif event_type in ("token", "chunk"):
                    chunk_text = event.get("content") or event.get("token", "")
                    await queue.put(("token", {"token": chunk_text, "content": chunk_text}))
                else:
                    await queue.put(("tool_call", event))

        async def run_agent():
            try:
                result = await agent.execute(  # type: ignore
                    request.prompt,
                    stream_callback=stream_callback,
                    stream=True,
                )
                done_payload = {
                    "output": result.output,
                    "status": result.status.value if hasattr(result.status, "value") else str(result.status),
                    "token_usage": result.token_usage,
                    "iterations": result.iterations,
                }
                await queue.put(("done", done_payload))
            except Exception as exc:
                await queue.put(("done", {
                    "output": f"Execution error: {exc}",
                    "status": "failed",
                    "error": str(exc),
                }))
            finally:
                await queue.put(None)

        task = asyncio.create_task(run_agent())
        try:
            while True:
                item = await queue.get()
                if item is None:
                    break
                event_name, data = item
                yield f"event: {event_name}\ndata: {json.dumps(data)}\n\n"
        finally:
            if not task.done():
                task.cancel()

    return StreamingResponse(
        sse_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )

async def execute_task_wrapper(thread_id: str, agent: AugAgent, prompt: str):
    try:
        await agent.execute(prompt)
        await execution_store.update_status(thread_id, ExecutionStatus.COMPLETED)
    except Exception as e:
        await execution_store.update_status(thread_id, ExecutionStatus.FAILED)

@app.post("/api/v1/agents/{agent_id}/execute", response_model=Dict[str, Any])
async def execute_task(agent_id: str, request: ExecuteRequest, background_tasks: BackgroundTasks, user: UserUser = Depends(require_permission("agents", "execute"))):
    """Execute a task using an existing agent."""
    if len(request.prompt) > 10000:
        raise HTTPException(400, "Prompt exceeds maximum length of 10000 characters.")
        
    agent = active_agents.get(agent_id)
    if not agent:
        raise HTTPException(status_code=404, detail="Agent not found")
        
    thread_id = str(uuid.uuid4())
    await execution_store.save_state(
        thread_id=thread_id,
        agent_config=agent.model_dump(),
        message_history=[{"role": "user", "content": request.prompt}],
        current_step=0,
        status=ExecutionStatus.RUNNING
    )
    
    background_tasks.add_task(execute_task_wrapper, thread_id, agent, request.prompt)
    return {"status": "started", "agent_id": agent_id, "thread_id": thread_id}

@app.get("/api/v1/agents/{agent_id}/status", response_model=Dict[str, Any])
async def check_status(agent_id: str, user: UserUser = Depends(require_role("operator"))):
    """Check execution status of an agent."""
    execution = active_executions.get(agent_id)
    if not execution:
        raise HTTPException(status_code=404, detail="Execution not found")
        
    task = execution["task"]
    if task.done():
        try:
            result = task.result()
            return {"status": "completed", "result": result.model_dump()}
        except Exception as e:
            return {"status": "failed", "error": str(e)}
    return {"status": "running"}

@app.post("/api/v1/teams/kickoff", response_model=List[TaskResult])
async def kickoff(request: KickoffRequest, user: UserUser = Depends(require_role("operator"))):
    """Execute a team of agents sequentially or hierarchically."""
    try:
        agents_map = {}
        agents = []
        for config in request.agents:
            agent = AugAgent.from_config(config)
            agents_map[agent.name] = agent
            agents.append(agent)
            
        tasks = []
        for tr in request.tasks:
            agent_to_use = agents_map.get(tr.agent_name) if tr.agent_name else agents[0]
            if not agent_to_use:
                raise HTTPException(status_code=400, detail=f"Agent '{tr.agent_name}' not defined in agents list.")
            task = AugTask(
                description=tr.description,
                expected_output=tr.expected_output,
                agent=agent_to_use,
                async_execution=tr.async_execution
            )
            tasks.append(task)
            
        team = AugTeam(agents=agents, tasks=tasks, process=request.process)  # type: ignore
        request.inputs["tenant_id"] = user.tenant_id
        results = await team.akickoff(request.inputs)
        return results
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/v1/audit/logs")
async def get_audit_logs(user: UserUser = Depends(require_role("admin"))):
    """Query audit logs (admin only)."""
    logs = []
    if audit.log_file.exists():
        with open(audit.log_file, "r", encoding="utf-8") as f:
            for line in f:
                logs.append(json.loads(line))
    return {"logs": logs}

@app.get("/api/v1/admin/rate-limits")
async def get_rate_limits(user: UserUser = Depends(require_role("admin"))):
    """View rate limits (admin only)."""
    return {
        "RATE_LIMIT_PER_MINUTE": RATE_LIMIT_PER_MINUTE,
        "type": rate_limiter.__class__.__name__
    }

class TenantCreate(BaseModel):
    id: str
    name: str

@app.post("/api/v1/admin/tenants")
async def create_tenant(request: TenantCreate, user: UserUser = Depends(require_role("admin"))):
    """Create a new tenant."""
    success = await tenant_store.create_tenant(request.id, request.name)
    if not success:
        raise HTTPException(400, "Tenant already exists")
    tenant = await tenant_store.get_tenant(request.id)
    return tenant

@app.get("/api/v1/admin/tenants")
async def list_tenants(user: UserUser = Depends(require_role("admin"))):
    """List all tenants."""
    tenants = await tenant_store.list_tenants()
    return {"tenants": tenants}

@app.get("/api/v1/admin/tenants/{tenant_id}/usage")
async def get_tenant_usage(tenant_id: str, user: UserUser = Depends(require_role("admin"))):
    """Get usage metering for a tenant."""
    tenant = await tenant_store.get_tenant(tenant_id)
    if not tenant:
        raise HTTPException(404, "Tenant not found")
    return {"usage": tenant}

@app.delete("/api/v1/admin/tenants/{tenant_id}")
async def deactivate_tenant(tenant_id: str, user: UserUser = Depends(require_role("admin"))):
    """Deactivate a tenant."""
    success = await tenant_store.deactivate_tenant(tenant_id)
    if not success:
        raise HTTPException(404, "Tenant not found")
    return {"status": "deactivated"}

@app.get("/api/v1/executions/pending")
async def get_pending_executions():
    """List all waiting-for-human executions."""
    return await execution_store.get_pending_executions()

@app.post("/api/v1/executions/{thread_id}/approve")
async def approve_execution(thread_id: str, background_tasks: BackgroundTasks):
    state = await execution_store.load_state(thread_id)
    if not state or state["status"] != ExecutionStatus.WAITING_HUMAN_INPUT:
        raise HTTPException(404, "Pending execution not found or not waiting for input.")
    
    await execution_store.update_status(thread_id, ExecutionStatus.RUNNING)
    
    # We would run this in background
    # background_tasks.add_task(agent.resume_pending_action, state)
    return {"status": "approved", "thread_id": thread_id}

@app.post("/api/v1/executions/{thread_id}/reject")
async def reject_execution(thread_id: str):
    state = await execution_store.load_state(thread_id)
    if not state or state["status"] != ExecutionStatus.WAITING_HUMAN_INPUT:
        raise HTTPException(404, "Pending execution not found.")
    
    await execution_store.update_status(thread_id, ExecutionStatus.FAILED)
    return {"status": "rejected", "thread_id": thread_id}

@app.get("/api/v1/executions/{thread_id}/state")
async def get_execution_state(thread_id: str):
    state = await execution_store.load_state(thread_id)
    if not state:
        raise HTTPException(404, "Execution state not found.")
    return state

@app.get("/metrics")
async def metrics():
    """Prometheus metrics endpoint."""
    if not HAS_PROMETHEUS:
        raise HTTPException(501, "Prometheus client not installed. `pip install prometheus_client`")
    from fastapi.responses import Response
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)  # type: ignore

@app.websocket("/ws/v1/stream")
async def websocket_stream(websocket: WebSocket):
    """
    WebSocket endpoint for real-time streaming of an agent's reasoning process.
    """
    from fastapi import WebSocketDisconnect
    
    await websocket.accept()
    token = websocket.query_params.get("token")
    
    try:
        if token:
            payload = jwt.decode(token, JWT_SECRET, algorithms=[JWT_ALGORITHM])
            roles = payload.get("roles", [])
            if "viewer" not in roles and "admin" not in roles and "operator" not in roles:
                await websocket.send_text(json.dumps({"type": "error", "message": "Insufficient permissions"}))
                await websocket.close(code=1008)
                return
                
        agent = None
        message_history: List[Dict[str, Any]] = []
        
        while True:
            data = await websocket.receive_text()
            req = json.loads(data)
            
            prompt = req.get("prompt", "")
            agent_config_dict = req.get("agent_config", {})
            
            if agent is None:
                config = AgentConfig.model_validate(agent_config_dict)
                tools_list = [view_file, create_file, list_directory, replace_file_content, run_terminal_command, list_open_windows]
                agent = AugAgent.from_config(config, tools=tools_list)
                agent.checkpointer = JsonlCheckpointer()
            
            async def stream_callback(chunk: str | dict):
                if isinstance(chunk, dict):
                    await websocket.send_text(json.dumps(chunk))
                else:
                    await websocket.send_text(json.dumps({"type": "chunk", "content": chunk}))
                
            result = await agent.execute(prompt, message_history=message_history, stream_callback=stream_callback)
            await websocket.send_text(json.dumps({"type": "result", "output": result.output}))
            
            message_history.append({"role": "user", "content": prompt})
            message_history.append({"role": "assistant", "content": result.output})
            
    except jwt.PyJWTError:
        try:
            await websocket.send_text(json.dumps({"type": "error", "message": "Invalid token"}))
            await websocket.close(code=1008)
        except Exception:
            pass
    except WebSocketDisconnect:
        pass
    except Exception as e:
        try:
            await websocket.send_text(json.dumps({"type": "error", "message": str(e)}))
        except Exception:
            pass
    finally:
        try:
            await websocket.close()
        except Exception:
            pass

# --- IDE File Explorer Endpoints ---

class FileContentRequest(BaseModel):
    path: str
    content: str

class FileCreateRequest(BaseModel):
    path: str
    is_dir: bool = False

@app.get("/api/v1/files/tree")
async def get_file_tree(root: str = "."):
    """Fetch the directory tree starting from root."""
    if not os.path.exists(root):
        raise HTTPException(status_code=404, detail="Directory not found")
        
    def _build_tree(dir_path):
        tree = []
        try:
            for item in sorted(os.listdir(dir_path)):
                # Skip hidden folders
                if item.startswith('.') and item not in ['.github', '.gitignore']:
                    continue
                if item in ['__pycache__', 'node_modules']:
                    continue
                    
                item_path = os.path.join(dir_path, item)
                is_dir = os.path.isdir(item_path)
                
                node = {
                    "name": item,
                    "path": item_path,
                    "isDir": is_dir
                }
                if is_dir:
                    # Don't recurse too deep to prevent performance issues, just one level
                    # The frontend should fetch lazily if needed, but for now we fetch it all
                    node["children"] = _build_tree(item_path)
                tree.append(node)
        except PermissionError:
            pass
        return tree
        
    return {"tree": _build_tree(root)}

@app.get("/api/v1/files/content")
async def get_file_content(path: str):
    """Get the text content of a file."""
    if not os.path.exists(path) or not os.path.isfile(path):
        raise HTTPException(status_code=404, detail="File not found")
        
    try:
        with open(path, "r", encoding="utf-8") as f:
            content = f.read()
        return {"content": content}
    except UnicodeDecodeError:
        raise HTTPException(status_code=400, detail="Cannot read binary files as text")

@app.put("/api/v1/files/content")
async def save_file_content(request: FileContentRequest):
    """Save the text content of a file."""
    try:
        with open(request.path, "w", encoding="utf-8") as f:
            f.write(request.content)
        return {"status": "success"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/v1/files/create")
async def create_file_or_dir(request: FileCreateRequest):
    """Create a new file or directory."""
    try:
        if request.is_dir:
            os.makedirs(request.path, exist_ok=True)
        else:
            with open(request.path, "w", encoding="utf-8") as f:
                f.write("")
        return {"status": "success"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.delete("/api/v1/files")
async def delete_file_or_dir(path: str):
    """Delete a file or directory."""
    import shutil
    if not os.path.exists(path):
        raise HTTPException(status_code=404, detail="Not found")
        
    try:
        if os.path.isdir(path):
            shutil.rmtree(path)
        else:
            os.remove(path)
        return {"status": "success"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# --- IDE Terminal Endpoint ---
from augagent.pty_server import handle_terminal_ws

@app.websocket("/ws/v1/terminal")
async def terminal_ws(websocket: WebSocket):
    await websocket.accept()
    
    # Disable auth token requirement for local IDE usage
    # token = websocket.query_params.get("token")
    # if not token:
    #     await websocket.send_text("Error: Authentication token required.\r\n")
    #     await websocket.close(code=1008)
    #     return
        
    await handle_terminal_ws(websocket)
