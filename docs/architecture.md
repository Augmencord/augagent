# Architecture

AugAgent is designed from the ground up for enterprise resilience, audibility, and compliance.

## Core ReAct Loop
At the heart of `AugAgent` is a ReAct (Reasoning + Acting) loop. The agent iterates through observing its environment, planning its actions, executing tools, and evaluating the result.

## Checkpointers and Memory
The framework persists conversations and execution state via Checkpointers.
- **MemoryCheckpointer**: In-memory state for ephemeral tasks.
- **SQLiteCheckpointer**: Durable state backed by SQLite.

Checkpointers enable pausing the ReAct loop for Human-In-The-Loop (HITL) approval flows and resuming later seamlessly.

## Multi-Model Fallback
In production, APIs can go down. AugAgent allows configuring fallback models. If a primary model (e.g., GPT-4o) fails or is rate-limited beyond retries, the agent automatically falls back to secondary models (e.g., Claude 3.5 Sonnet or Ollama local models).

## Tooling and MCP
Tools are heavily strictly typed using Pydantic. AugAgent can auto-generate OpenAI compatible JSON schemas from any Python function using the `@aug_tool` decorator.

Additionally, AugAgent ships with a native Model Context Protocol (MCP) client to dynamically inject tools from external MCP servers.

## Audit and Compliance
All tool executions and agent reasoning steps can be funneled into the `AuditLogger`, which creates immutable JSONL logs with automated PII (Personally Identifiable Information) redaction, ensuring compliance with strict data privacy laws.
