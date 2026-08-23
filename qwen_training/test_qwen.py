import asyncio
from augagent.models import AgentConfig, LLMConfig
from augagent.agent import AugAgent

async def main():
    config = AgentConfig(
        name="TestAgent",
        role="Tester",
        goal="Test the model",
        backstory="You are a test agent.",
        # Using default LLMConfig which defaults to qwen2.5-coder:7b and ollama
    )
    agent = AugAgent.from_config(config)
    result = await agent.execute("Say hello world!")
    print(f"Result: {result.output}")

if __name__ == "__main__":
    asyncio.run(main())
