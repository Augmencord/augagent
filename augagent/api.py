from fastapi import FastAPI, HTTPException, WebSocket, Security, Depends
from fastapi.security import APIKeyHeader
from pydantic import BaseModel
from typing import List, Dict, Any
import asyncio
import json
import os

from augagent.models import AgentConfig, TaskResult, LLMConfig
from augagent.agent import AugAgent
from augagent.task import AugTask
from augagent.team import AugTeam

api_key_header = APIKeyHeader(name="X-API-Key", auto_error=False)

def get_api_key(api_key: str = Security(api_key_header)) -> str:
    expected_key = os.getenv("AUGAGENT_API_KEY")
    if expected_key and api_key != expected_key:
        raise HTTPException(
            status_code=401,
            detail="Invalid or missing API Key",
        )
    return api_key or ""

app = FastAPI(title="AugAgent API", description="REST and WebSocket interfaces for AugAgent orchestration")

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
async def kickoff(request: KickoffRequest, api_key: str = Depends(get_api_key)):
    """
    Execute a team of agents sequentially or hierarchically based on the request.
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
    Authentication is done via the 'api_key' query parameter.
    """
    await websocket.accept()
    
    expected_key = os.getenv("AUGAGENT_API_KEY")
    api_key = websocket.query_params.get("api_key")
    if expected_key and api_key != expected_key:
        await websocket.send_text(json.dumps({"type": "error", "message": "Invalid API Key"}))
        await websocket.close(code=1008)
        return

    try:
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
        
    except Exception as e:
        await websocket.send_text(json.dumps({"type": "error", "message": str(e)}))
    finally:
        await websocket.close()
