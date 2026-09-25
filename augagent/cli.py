"""AugAgent Command Line Interface."""

import asyncio
import typer
import uvicorn
from augagent.agent import AugAgent

app = typer.Typer(help="AugAgent Command Line Interface")


@app.command("api")
def start_server(
    host: str = typer.Option("0.0.0.0", help="Bind socket to this host"),
    port: int = typer.Option(8000, help="Bind socket to this port"),
    reload: bool = typer.Option(False, help="Enable auto-reload"),
) -> None:
    """Start the AugAgent FastAPI server."""
    print(f"Starting API server on {host}:{port}")
    uvicorn.run("augagent.api:app", host=host, port=port, reload=reload)


@app.command("run")
def run_agent(
    prompt: str = typer.Argument(..., help="The instruction for the agent"),
    name: str = typer.Option("CLI Agent", help="Agent name"),
    role: str = typer.Option("Assistant", help="Agent role"),
    goal: str = typer.Option("Fulfill the user request", help="Agent goal"),
) -> None:
    """Run a single agent execution."""
    async def _run():
        agent = AugAgent(
            name=name,
            role=role,
            goal=goal,
        )
        print(f"Executing: '{prompt}'...")
        result = await agent.execute(prompt)
        print("\n--- Result ---\n")
        print(result.output)
        print("\n--------------")

    asyncio.run(_run())


def main() -> None:
    """Entrypoint function for CLI execution."""
    app()


if __name__ == "__main__":
    main()
