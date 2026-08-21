import typer
import asyncio
from rich.console import Console
from rich.panel import Panel
import uvicorn
import json

from augagent.models import AgentConfig, LLMConfig
from augagent.agent import AugAgent

app = typer.Typer(help="AugAgent Command Line Interface")
console = Console()

@app.command()
def run(
    prompt: str = typer.Argument(..., help="The prompt to send to the agent"),
    name: str = typer.Option("CLI_Agent", help="Name of the agent"),
    role: str = typer.Option("Assistant", help="Role of the agent"),
    goal: str = typer.Option("Help the user", help="Goal of the agent"),
    model: str = typer.Option("qwen2.5-coder:7b", help="LLM model to use")
):
    """Run a single agent execution."""
    console.print(Panel(f"Starting {name} ({model})...", title="AugAgent CLI"))
    
    config = AgentConfig(
        name=name,
        role=role,
        goal=goal,
        llm_config=LLMConfig(model=model)
    )
    
    agent = AugAgent.from_config(config)
    
    async def _run():
        result = await agent.execute(prompt)
        console.print(Panel(result.output, title="Result", style="bold green"))
        
    asyncio.run(_run())

@app.command()
def serve(
    host: str = typer.Option("0.0.0.0", help="Host to bind the server to"),
    port: int = typer.Option(8000, help="Port to bind the server to"),
    reload: bool = typer.Option(False, help="Enable auto-reload")
):
    """Start the AugAgent REST API and WebSocket server."""
    console.print(Panel(f"Starting AugAgent API on {host}:{port}", title="AugAgent Server"))
    uvicorn.run("augagent.api:app", host=host, port=port, reload=reload)

if __name__ == "__main__":
    app()
