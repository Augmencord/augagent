# Welcome to AugAgent

AugAgent is an enterprise-grade multi-agent orchestration framework, featuring:
- **Resilience**: Multi-model fallback routing and retries.
- **Persistence**: SQLite and ChromaDB checkpointers.
- **Extensibility**: Plugin registry and MCP client wrapper.
- **Assurance**: OpenTelemetry logging and strict token budgets.
- **Security**: Immutable audit logging and RBAC API endpoints.

## Getting Started

```bash
pip install augagent[api,telemetry,mcp]
```

See the [API Reference](api_reference.md) for more details.
