# AugAgent

[![PyPI version](https://badge.fury.io/py/augagent.svg)](https://badge.fury.io/py/augagent)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)

**AugAgent** is an enterprise-grade multi-agent orchestration framework designed for resilience, observability, and compliance.

## Features

- **Resilience**: Built-in multi-model fallback routing and retries across OpenAI, Ollama, and LiteLLM compatible APIs.
- **Persistence**: Pluggable state checkpointers (SQLite & ChromaDB) for durable agent memory.
- **Extensibility**: Native MCP (Model Context Protocol) client wrapper to inject dynamic tools.
- **Assurance**: OpenTelemetry logging and strict Pydantic token budgets.
- **Security**: Immutable JSONL audit logging with automatic PII redaction and FastAPI RBAC.

## Quickstart

### Installation

```bash
pip install augagent[api,telemetry,mcp]
```

### Basic Usage

```python
import asyncio
from augagent.agent import AugAgent
from augagent.models import AgentConfig, LLMConfig

async def main():
    agent = AugAgent(
        config=AgentConfig(
            name="ResearchBot",
            role="Analyst",
            goal="Analyze technical data",
            llm_config=LLMConfig(model="gpt-4o")
        )
    )
    
    result = await agent.run("What are the benefits of MCP?")
    print(result.output)

if __name__ == "__main__":
    asyncio.run(main())
```

## Documentation
Please refer to the `docs/` directory for our complete API reference and architectural guides.

## Contributing
See [CONTRIBUTING.md](CONTRIBUTING.md) for how to help out.

## Security
See [SECURITY.md](SECURITY.md) for vulnerability reporting.

## License
This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
