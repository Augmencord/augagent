# Getting Started

Welcome to AugAgent! This tutorial covers the absolute basics of getting your first agent running.

## Installation

```bash
pip install augagent
```

## Your First Agent

```python
from augagent import AugAgent, LLMConfig

agent = AugAgent(
    name="Assistant",
    role="Helpful bot",
    goal="Answer questions",
    llm_config=LLMConfig(model="gpt-4o-mini")
)
```
