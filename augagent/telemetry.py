"""Telemetry & logging primitives.

Provides a lightweight logger that prints agent handoffs and tool executions
to the console in a clean, readable format.
"""

from __future__ import annotations

import logging
import sys

# Configure a basic console logger
_logger = logging.getLogger("augagent")
_logger.setLevel(logging.INFO)

if not _logger.handlers:
    handler = logging.StreamHandler(sys.stdout)
    formatter = logging.Formatter("%(message)s")
    handler.setFormatter(formatter)
    _logger.addHandler(handler)


class AgentLogger:
    """A lightweight logger for tracking agent orchestration."""

    def __init__(self, logger: logging.Logger = _logger):
        self._logger = logger

    def log_handoff(self, from_agent: str, to_agent: str, task_desc: str) -> None:
        """Log a task handoff to an agent."""
        self._logger.info(f"\n[{from_agent} -> {to_agent}] Starting Task: {task_desc}")

    def log_tool_execution(self, agent_name: str, tool_name: str, args: dict[str, Any] | str) -> None:
        """Log when an agent executes a tool."""
        self._logger.info(f"  [Tool Execution] {agent_name} is using '{tool_name}' with args: {args}")

    def log_info(self, message: str) -> None:
        """Log a general info message."""
        self._logger.info(f"  [Info] {message}")

    def log_error(self, message: str) -> None:
        """Log an error message."""
        self._logger.error(f"  [Error] {message}")


_DEFAULT_LOGGER = AgentLogger()

def get_logger() -> AgentLogger:
    """Return the global agent logger instance."""
    return _DEFAULT_LOGGER
