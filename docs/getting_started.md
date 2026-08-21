# Getting Started with AugAgent

Welcome to AugAgent! This guide will help you set up and build your first autonomous agent.

## Prerequisites
- Python 3.10+
- An OpenAI, Ollama, or LiteLLM compatible API Key

## Installation
Install AugAgent using pip with all recommended extras:

```bash
pip install augagent[api,telemetry,mcp]
```

## Building Your First Agent

Let's build a simple Weather Assistant that can use a tool to fetch weather.

```python
import asyncio
from augagent.agent import AugAgent
from augagent.models import AgentConfig, LLMConfig
from augagent.tools import aug_tool

# 1. Define a tool
@aug_tool
def get_weather(location: str) -> str:
    """Get the current weather for a specific location."""
    # In reality, this would call a real API
    return f"The weather in {location} is 72°F and sunny."

async def main():
    # 2. Configure the LLM
    config = LLMConfig(
        model="gpt-4o",
        api_key="your-api-key"
    )
    
    # 3. Initialize the Agent
    agent = AugAgent(
        name="WeatherBot",
        role="Assistant",
        goal="Help the user with weather questions",
        llm_config=config,
    )
    
    # 4. Register the tool
    agent.tools.append(get_weather)
    
    # 5. Execute a task
    result = await agent.execute("What is the weather in Seattle?")
    
    print(result.output)
    print(f"Tokens used: {result.token_usage['total_tokens']}")

if __name__ == "__main__":
    asyncio.run(main())
```

## Next Steps
- Read about [Architecture](architecture.md) to understand state management and multi-model fallback.
- Explore the [API Reference](api_reference.md) for full configuration details.
