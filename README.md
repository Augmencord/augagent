# AugAgent

An enterprise-grade autonomous ReAct agent framework with token streaming, hierarchical orchestration, and vector memory.

## Features
- **Real-Time Token Streaming**: Built heavily on `httpx.AsyncClient` for blazing fast, token-by-token UI updates.
- **Hierarchical Orchestration**: Includes `DelegateWorkTool` out-of-the-box, allowing "Manager" agents to spin up specialized sub-agents.
- **RAG & Memory**: Natively integrates ChromaDB for sliding-window and long-term vector memory abstractions.
- **Human-in-the-Loop**: Supports `require_human_approval` to yield payloads to an external UI for approval before code execution.
- **Provider Agnostic**: Use OpenAI, Anthropic, or local open-source models via Ollama.

## Installation
```bash
pip install augagent
```

## Quick Start
```python
import asyncio
from augagent import Agent, LLMConfig

async def main():
    config = LLMConfig(model="qwen2.5-coder:7b")
    agent = Agent(
        name="AugHome-Root", 
        role="Senior AI Engineer", 
        goal="Handle IDE requests", 
        llm_config=config
    )
    
    result = await agent.execute("Write a python script to calculate fibonacci numbers.")
    print(result.output)

if __name__ == "__main__":
    asyncio.run(main())
```
