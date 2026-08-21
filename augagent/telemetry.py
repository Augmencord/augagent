"""Telemetry & logging primitives.

Provides a lightweight logger and OpenTelemetry integration.
"""

from __future__ import annotations

import logging
import sys
from typing import Any

_logger = logging.getLogger("augagent")
_logger.setLevel(logging.INFO)

if not _logger.handlers:
    handler = logging.StreamHandler(sys.stdout)
    formatter = logging.Formatter("%(message)s")
    handler.setFormatter(formatter)
    _logger.addHandler(handler)

try:
    from opentelemetry import trace
    HAS_OTEL = True
    tracer = trace.get_tracer("augagent")
except ImportError:
    HAS_OTEL = False
    tracer = None

class DummySpan:
    def __enter__(self): return self
    def __exit__(self, *args): pass
    def set_attribute(self, key: str, val: Any) -> None: pass
    def record_exception(self, exc: Exception) -> None: pass
    def set_status(self, status: Any) -> None: pass

class AgentLogger:
    """A lightweight logger for tracking agent orchestration."""

    def __init__(self, logger: logging.Logger = _logger):
        self._logger = logger

    def log_handoff(self, from_agent: str, to_agent: str, task_desc: str) -> None:
        self._logger.info(f"\n[{from_agent} -> {to_agent}] Starting Task: {task_desc}")

    def log_tool_execution(self, agent_name: str, tool_name: str, args: dict[str, Any] | str) -> None:
        self._logger.info(f"  [Tool Execution] {agent_name} is using '{tool_name}' with args: {args}")

    def log_info(self, message: str) -> None:
        self._logger.info(f"  [Info] {message}")

    def log_error(self, message: str) -> None:
        self._logger.error(f"  [Error] {message}")

    def start_span(self, name: str, attributes: dict[str, Any] | None = None) -> Any:
        """Start an OpenTelemetry span if available, else a dummy context."""
        if HAS_OTEL:
            return tracer.start_as_current_span(name, attributes=attributes)
        return DummySpan()

_DEFAULT_LOGGER = AgentLogger()

def get_logger() -> AgentLogger:
    return _DEFAULT_LOGGER
