# Changelog

All notable changes to this project will be documented in this file.

## [1.0.0] - 2026-08-21
### Added
- Multi-model fallback routing and retry mechanisms.
- Support for OpenAI, Ollama, vLLM, and LiteLLM compatible endpoints.
- SQLite and ChromaDB checkpointers for persistent state.
- MCP client wrapper (`mcp_client.py`) for tool integrations.
- Role-Based Access Control (RBAC) via API Key headers.
- Immutable, PII-redacted JSONL Audit Logging for compliance.
- FastAPI backend with WebSocket streaming capabilities.
- OpenTelemetry observability metrics.

### Changed
- Standardized `AgentConfig` and `LLMConfig` structures using Pydantic.
- Refactored internal architecture for enterprise-readiness and zero data leakage.
