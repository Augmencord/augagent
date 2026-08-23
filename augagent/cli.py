import argparse
import asyncio
import sys
import uvicorn
from augagent.agent import AugAgent

def start_server(args):
    """Start the AugAgent FastAPI server."""
    print(f"Starting API server on {args.host}:{args.port}")
    uvicorn.run("augagent.api:app", host=args.host, port=args.port, reload=args.reload)

def run_agent(args):
    """Run an agent with a specific prompt."""
    async def _run():
        agent = AugAgent(
            name=args.name,
            role=args.role,
            goal=args.goal
        )
        print(f"Executing: '{args.prompt}'...")
        result = await agent.execute(args.prompt)
        print("\n--- Result ---\n")
        print(result.output)
        print("\n--------------")
    
    asyncio.run(_run())

def main():
    parser = argparse.ArgumentParser(description="AugAgent CLI")
    subparsers = parser.add_subparsers(dest="command", help="Available commands", required=True)
    
    # API command
    api_parser = subparsers.add_parser("api", help="Start the FastAPI backend server")
    api_parser.add_argument("--host", type=str, default="0.0.0.0", help="Bind socket to this host")
    api_parser.add_argument("--port", type=int, default=8000, help="Bind socket to this port")
    api_parser.add_argument("--reload", action="store_true", help="Enable auto-reload")
    api_parser.set_defaults(func=start_server)
    
    # Run command
    run_parser = subparsers.add_parser("run", help="Run a single agent execution")
    run_parser.add_argument("prompt", type=str, help="The instruction for the agent")
    run_parser.add_argument("--name", type=str, default="CLI Agent", help="Agent name")
    run_parser.add_argument("--role", type=str, default="Assistant", help="Agent role")
    run_parser.add_argument("--goal", type=str, default="Fulfill the user request", help="Agent goal")
    run_parser.set_defaults(func=run_agent)
    
    args = parser.parse_args()
    args.func(args)

if __name__ == "__main__":
    main()
