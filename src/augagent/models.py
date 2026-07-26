"""Core Pydantic models for the augagent framework.

All data flowing through the system is strongly typed via Pydantic v2 models,
providing runtime validation, serialisation, and rich editor support.
"""

from __future__ import annotations

import os
import uuid
from datetime import datetime, timezone
from enum import Enum
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, SecretStr


# ---------------------------------------------------------------------------
# Enumerations
# ---------------------------------------------------------------------------

class Role(str, Enum):
    """Roles within a conversation turn."""

    SYSTEM = "system"
    USER = "user"
    ASSISTANT = "assistant"
    TOOL = "tool"


class TaskStatus(str, Enum):
    """Lifecycle status of a :class:`Task`."""

    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


# ═══════════════════════════════════════════════════════════════════════════
# LLM CONFIGURATION SCHEMAS
# ═══════════════════════════════════════════════════════════════════════════

class LLMConfig(BaseModel):
    """Declarative configuration for connecting to an LLM provider.

    Supports any **OpenAI-compatible** API (OpenAI, Azure OpenAI, Ollama,
    vLLM, LiteLLM, etc.) by pointing ``base_url`` at the right endpoint.

    Example::

        # Explicit key
        config = LLMConfig(model="gpt-4o", api_key="sk-...")

        # From environment (default: reads OPENAI_API_KEY)
        config = LLMConfig(model="gpt-4o")

        # Local Ollama
        config = LLMConfig(
            model="llama3",
            base_url="http://localhost:11434/v1",
            api_key="ollama",
        )
    """

    model_config = ConfigDict(populate_by_name=True)

    model: str = Field(
        default="gpt-4o",
        description="Model identifier (e.g. 'gpt-4o', 'claude-3-opus', 'llama3').",
    )
    api_key: SecretStr | None = Field(
        default=None,
        description=(
            "API key for the provider. If not set, the framework reads the "
            "environment variable specified by ``api_key_env_var``."
        ),
    )
    api_key_env_var: str = Field(
        default="OPENAI_API_KEY",
        description="Environment variable to read the API key from when ``api_key`` is None.",
    )
    base_url: str = Field(
        default="https://api.openai.com/v1",
        description="Base URL of the chat-completions API.",
    )
    temperature: float = Field(default=0.7, ge=0.0, le=2.0)
    max_tokens: int = Field(default=4096, ge=1)
    top_p: float = Field(default=1.0, ge=0.0, le=1.0)
    frequency_penalty: float = Field(default=0.0, ge=-2.0, le=2.0)
    presence_penalty: float = Field(default=0.0, ge=-2.0, le=2.0)
    stop: list[str] | None = Field(
        default=None,
        description="Up to 4 sequences where the model will stop generating.",
    )
    timeout: float = Field(
        default=120.0,
        gt=0,
        description="HTTP request timeout in seconds.",
    )
    max_retries: int = Field(
        default=3,
        ge=0,
        description="Retries on transient HTTP errors (429, 500, 502, 503).",
    )
    extra_headers: dict[str, str] = Field(
        default_factory=dict,
        description="Additional HTTP headers sent with every request.",
    )

    def resolve_api_key(self) -> str:
        """Return the API key, resolving from the environment if necessary.

        Raises
        ------
        ValueError
            If no key is available from either the field or the env var.
        """
        if self.api_key is not None:
            return self.api_key.get_secret_value()
        key = os.environ.get(self.api_key_env_var, "")
        if not key:
            raise ValueError(
                f"No API key provided and environment variable "
                f"'{self.api_key_env_var}' is not set."
            )
        return key


# ═══════════════════════════════════════════════════════════════════════════
# CHAT COMPLETION RESPONSE SCHEMAS  (OpenAI-compatible)
# ═══════════════════════════════════════════════════════════════════════════

class FunctionCall(BaseModel):
    """The function the model wants to invoke (name + raw JSON args)."""

    name: str
    arguments: str  # JSON-encoded string — parsed by the caller


class ChatToolCall(BaseModel):
    """A single tool-call entry inside an assistant message."""

    id: str
    type: str = "function"
    function: FunctionCall


class ChatMessage(BaseModel):
    """A message object as returned inside a ``Choice``."""

    model_config = ConfigDict(populate_by_name=True)

    role: str
    content: str | None = None
    tool_calls: list[ChatToolCall] | None = None
    refusal: str | None = None


class Choice(BaseModel):
    """One candidate completion returned by the API."""

    index: int = 0
    message: ChatMessage
    finish_reason: str | None = None


class TokenUsage(BaseModel):
    """Token-level usage statistics from the API response."""

    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0


class ChatCompletion(BaseModel):
    """Fully-parsed ``/chat/completions`` response.

    Using a Pydantic model here means malformed API responses are caught
    immediately with clear validation errors rather than producing silent
    ``KeyError`` s downstream.
    """

    model_config = ConfigDict(populate_by_name=True)

    id: str = ""
    object: str = "chat.completion"
    created: int | None = None
    model: str = ""
    choices: list[Choice] = Field(default_factory=list)
    usage: TokenUsage | None = None


# ---------------------------------------------------------------------------
# Internal message models
# ---------------------------------------------------------------------------

class ToolCall(BaseModel):
    """Framework-internal representation of a tool invocation request."""

    model_config = ConfigDict(frozen=True)

    id: str = Field(default_factory=lambda: f"call_{uuid.uuid4().hex[:12]}")
    tool_name: str = Field(..., description="Name of the tool to invoke.")
    arguments: dict[str, Any] = Field(
        default_factory=dict,
        description="Keyword arguments for the tool.",
    )


class ToolResponse(BaseModel):
    """Encapsulates the result returned by a tool invocation."""

    model_config = ConfigDict(frozen=True)

    tool_call_id: str = Field(..., description="ID of the originating ToolCall.")
    tool_name: str
    content: str = Field(..., description="Serialised output from the tool.")
    is_error: bool = False


class Message(BaseModel):
    """A single message within an agent conversation.

    Messages form the backbone of inter-agent and agent-LLM communication.
    """

    model_config = ConfigDict(frozen=True)

    id: str = Field(default_factory=lambda: uuid.uuid4().hex)
    role: Role
    content: str | None = None
    tool_calls: list[ToolCall] = Field(default_factory=list)
    tool_responses: list[ToolResponse] = Field(default_factory=list)
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    metadata: dict[str, Any] = Field(default_factory=dict)


# ---------------------------------------------------------------------------
# Agent configuration  (serialisable subset of AugAgent fields)
# ---------------------------------------------------------------------------

class AgentConfig(BaseModel):
    """Declarative, serialisable configuration for an
    :class:`~augagent.agent.AugAgent`.

    Useful for persisting, templating, or sharing agent definitions
    as JSON / YAML without carrying callable state.
    """

    model_config = ConfigDict(populate_by_name=True)

    name: str = Field(..., min_length=1, max_length=128)
    role: str = Field(..., description="A short descriptor of the agent's persona.")
    goal: str = Field(..., description="What this agent is trying to achieve.")
    backstory: str = Field(
        default="",
        description="Rich context that shapes the agent's behaviour.",
    )
    llm_config: LLMConfig = Field(default_factory=LLMConfig)
    max_iterations: int = Field(
        default=25,
        ge=1,
        description="Maximum ReAct loop iterations before the agent stops.",
    )
    allow_delegation: bool = Field(
        default=False,
        description="Whether this agent may delegate sub-tasks to teammates.",
    )
    verbose: bool = False


# ---------------------------------------------------------------------------
# Task result
# ---------------------------------------------------------------------------

class TaskResult(BaseModel):
    """The output produced when a :class:`~augagent.task.Task` completes."""

    model_config = ConfigDict(frozen=True)

    task_id: str
    agent_name: str
    status: TaskStatus = TaskStatus.COMPLETED
    output: str = ""
    raw_output: Any = None
    token_usage: dict[str, int] = Field(default_factory=dict)
    elapsed_seconds: float = 0.0
    iterations: int = 0
    metadata: dict[str, Any] = Field(default_factory=dict)
