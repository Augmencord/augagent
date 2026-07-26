"""AugAgent — the core autonomous entity in augagent.

:class:`AugAgent` encapsulates a persona (role, goal, backstory), an
:class:`~augagent.models.LLMConfig`, and a set of
:class:`~augagent.tools.AugTool` instances.  It implements a **ReAct**
(Reason + Act) loop that interleaves LLM reasoning with tool execution,
communicating with the LLM provider over HTTP via :mod:`httpx`.

Example::

    from augagent import AugAgent, aug_tool, LLMConfig
    from pydantic import Field

    @aug_tool
    def lookup_db(query: str = Field(description="SQL query")) -> str:
        \"\"\"Run a database lookup.\"\"\"
        return "42"

    agent = AugAgent(
        name="Analyst",
        role="Data Analyst",
        goal="Answer data questions accurately",
        backstory="You have 10 years of experience with SQL and analytics.",
        llm_config=LLMConfig(model="gpt-4o", temperature=0.2),
        tools=[lookup_db],
    )

    result = await agent.execute("What is the total revenue for Q3?")
"""

from __future__ import annotations

import asyncio
import json
import time
from typing import Any

import httpx
from pydantic import BaseModel, ConfigDict, Field, PrivateAttr

from augagent.models import (
    AgentConfig,
    ChatCompletion,
    ChatMessage,
    ChatToolCall,
    LLMConfig,
    TaskResult,
    TaskStatus,
)
from augagent.telemetry import get_logger
from augagent.tools import AugTool


class AugAgent(BaseModel):
    """A single autonomous agent with a built-in ReAct loop.

    Parameters
    ----------
    name:
        Unique, human-readable identifier for this agent.
    role:
        The agent's persona (e.g. ``"Senior Python Developer"``).
    goal:
        One-sentence objective guiding behaviour.
    backstory:
        Rich context that shapes reasoning style and domain knowledge.
    llm_config:
        Connection and generation settings for the LLM provider.
    tools:
        :class:`~augagent.tools.AugTool` instances the agent may invoke.
    max_iterations:
        Safety cap on ReAct loop iterations (Reason → Act → Observe).
    verbose:
        Enable detailed step-by-step logging.
    allow_delegation:
        Whether this agent may delegate sub-tasks to teammates.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    # ── identity ──────────────────────────────────────────────────────────
    name: str = Field(..., min_length=1, max_length=128)
    role: str
    goal: str
    backstory: str = ""

    # ── LLM configuration ────────────────────────────────────────────────
    llm_config: LLMConfig = Field(default_factory=LLMConfig)

    # ── capabilities ─────────────────────────────────────────────────────
    tools: list[AugTool] = Field(default_factory=list)

    # ── behaviour ────────────────────────────────────────────────────────
    max_iterations: int = Field(default=25, ge=1)
    verbose: bool = False
    allow_delegation: bool = False

    # ── internal state (private, excluded from serialisation) ─────────────
    _message_history: list[dict[str, Any]] = PrivateAttr(default_factory=list)

    # ══════════════════════════════════════════════════════════════════════
    # PUBLIC API
    # ══════════════════════════════════════════════════════════════════════

    async def execute(self, prompt: str) -> TaskResult:
        """Run the agent's ReAct loop on the given prompt."""
        logger = get_logger()
        start = time.time()

        if self.verbose:
            logger.log_info(f"[{self.name}] starting execution with {self.llm_config.model}")

        try:
            output, usage, iterations = await self._react_loop(prompt, logger)
            elapsed = time.time() - start

            result = TaskResult(
                task_id="",
                agent_name=self.name,
                status=TaskStatus.COMPLETED,
                output=output,
                token_usage=usage,
                elapsed_seconds=round(elapsed, 3),
                iterations=iterations,
            )

            if self.verbose:
                logger.log_info(
                    f"[{self.name}] completed in {elapsed:.2f}s "
                    f"({iterations} iterations, {usage.get('total_tokens', 0)} tokens)"
                )
            return result

        except Exception as exc:
            elapsed = time.time() - start
            logger.log_error(f"[{self.name}] execution failed: {exc}")

            return TaskResult(
                task_id="",
                agent_name=self.name,
                status=TaskStatus.FAILED,
                output=f"Agent execution failed: {exc}",
                elapsed_seconds=round(elapsed, 3),
            )

    # ══════════════════════════════════════════════════════════════════════
    # REACT LOOP  (Reason → Act → Observe)
    # ══════════════════════════════════════════════════════════════════════

    async def _react_loop(
        self,
        prompt: str,
        logger: Any,
    ) -> tuple[str, dict[str, int], int]:
        """Core ReAct loop."""
        messages: list[dict[str, Any]] = [
            {"role": "system", "content": self._build_system_prompt()},
            {"role": "user", "content": prompt},
        ]
        self._message_history = list(messages)

        total_usage: dict[str, int] = {
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "total_tokens": 0,
        }

        for iteration in range(1, self.max_iterations + 1):
            if self.verbose:
                logger.log_info(f"[{self.name}] --- iteration {iteration} / {self.max_iterations} ---")

            # ── REASON: call the LLM ─────────────────────────────────────
            completion = await self._call_llm(messages, logger)

            if completion.usage:
                total_usage["prompt_tokens"] += completion.usage.prompt_tokens
                total_usage["completion_tokens"] += completion.usage.completion_tokens
                total_usage["total_tokens"] += completion.usage.total_tokens

            if not completion.choices:
                raise RuntimeError("LLM returned an empty choices array.")

            assistant_msg = completion.choices[0].message
            messages.append(self._chat_message_to_dict(assistant_msg))

            # ── Final answer (no tool calls) ─────────────────────────────
            if not assistant_msg.tool_calls:
                final_output = assistant_msg.content or ""
                if self.verbose:
                    logger.log_info(f"[{self.name}] final answer ({len(final_output)} chars)")
                self._message_history = list(messages)
                return final_output, total_usage, iteration

            # ── ACT: execute each tool call ──────────────────────────────
            for tc in assistant_msg.tool_calls:
                logger.log_tool_execution(
                    agent_name=self.name,
                    tool_name=tc.function.name,
                    args=tc.function.arguments[:200]
                )
                tool_output = await self._execute_tool_call(tc, logger)
                messages.append({
                    "role": "tool",
                    "tool_call_id": tc.id,
                    "content": tool_output,
                })

            self._message_history = list(messages)

        # ── Max iterations exhausted ─────────────────────────────────────
        logger.log_error(f"[{self.name}] reached max iterations ({self.max_iterations}).")
        
        last_content = ""
        for msg in reversed(messages):
            if msg.get("role") == "assistant" and msg.get("content"):
                last_content = msg["content"]
                break

        return (
            last_content or f"[Max iterations ({self.max_iterations}) reached without a final answer]",
            total_usage,
            self.max_iterations,
        )

    # ══════════════════════════════════════════════════════════════════════
    # LLM COMMUNICATION  (httpx)
    # ══════════════════════════════════════════════════════════════════════

    async def _call_llm(
        self,
        messages: list[dict[str, Any]],
        logger: Any,
    ) -> ChatCompletion:
        """POST to ``/chat/completions`` with retry + exponential backoff."""
        cfg = self.llm_config
        url = f"{cfg.base_url.rstrip('/')}/chat/completions"
        api_key = cfg.resolve_api_key()

        headers: dict[str, str] = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
            **cfg.extra_headers,
        }

        payload: dict[str, Any] = {
            "model": cfg.model,
            "messages": messages,
            "temperature": cfg.temperature,
            "max_tokens": cfg.max_tokens,
            "top_p": cfg.top_p,
        }
        if cfg.frequency_penalty != 0.0:
            payload["frequency_penalty"] = cfg.frequency_penalty
        if cfg.presence_penalty != 0.0:
            payload["presence_penalty"] = cfg.presence_penalty
        if cfg.stop:
            payload["stop"] = cfg.stop

        if self.tools:
            payload["tools"] = [t.to_openai_schema() for t in self.tools]

        last_exc: Exception | None = None

        for attempt in range(cfg.max_retries + 1):
            try:
                async with httpx.AsyncClient(
                    timeout=httpx.Timeout(cfg.timeout),
                ) as client:
                    resp = await client.post(url, headers=headers, json=payload)

                    if resp.status_code == 429:
                        retry_after = float(resp.headers.get("retry-after", str(2 ** attempt)))
                        logger.log_error(f"Rate-limited (429). Retrying in {retry_after:.1f}s")
                        await asyncio.sleep(retry_after)
                        continue

                    resp.raise_for_status()
                    return ChatCompletion.model_validate(resp.json())

            except httpx.TimeoutException as exc:
                last_exc = exc
                logger.log_error(f"Timeout [attempt {attempt + 1}/{cfg.max_retries + 1}]: {exc}")
            except httpx.HTTPStatusError as exc:
                if exc.response.status_code in {500, 502, 503}:
                    last_exc = exc
                    logger.log_error(f"Server error {exc.response.status_code} [attempt {attempt + 1}/{cfg.max_retries + 1}]")
                else:
                    raise

            if attempt < cfg.max_retries:
                backoff = min(2 ** attempt, 30)
                await asyncio.sleep(backoff)

        raise RuntimeError(f"LLM request failed after {cfg.max_retries + 1} attempts: {last_exc}")

    # ══════════════════════════════════════════════════════════════════════
    # TOOL EXECUTION
    # ══════════════════════════════════════════════════════════════════════

    async def _execute_tool_call(self, tool_call: ChatToolCall, logger: Any) -> str:
        """Dispatch a single tool call to the matching :class:`AugTool`."""
        tool_map = {t.name: t for t in self.tools}
        tool_impl = tool_map.get(tool_call.function.name)

        if tool_impl is None:
            error = f"Tool '{tool_call.function.name}' not found."
            logger.log_error(error)
            return json.dumps({"error": error})

        try:
            arguments = json.loads(tool_call.function.arguments)
        except json.JSONDecodeError as exc:
            error = f"Failed to parse tool arguments: {exc}"
            logger.log_error(error)
            return json.dumps({"error": error})

        try:
            result = await tool_impl.run(**arguments)
            return result
        except Exception as exc:
            error = f"Tool error ({type(exc).__name__}): {exc}"
            logger.log_error(f"[{tool_call.function.name}] {error}")
            return json.dumps({"error": error})

    # ══════════════════════════════════════════════════════════════════════
    # PROMPT CONSTRUCTION
    # ══════════════════════════════════════════════════════════════════════

    def _build_system_prompt(self) -> str:
        """Assemble the system prompt from the agent's persona fields."""
        parts: list[str] = [
            f"You are {self.name}, a {self.role}.",
            f"\nYour goal: {self.goal}",
        ]

        if self.backstory:
            parts.append(f"\nBackstory: {self.backstory}")

        if self.tools:
            tool_lines = "\n".join(f"  • {t.name} — {t.description}" for t in self.tools)
            parts.append(f"\nYou have access to the following tools:\n{tool_lines}")
            parts.append(
                "\nUse tools when they would help accomplish your goal.  "
                "When you have gathered enough information to provide a "
                "final answer, respond directly without making tool calls."
            )

        return "\n".join(parts)

    # ══════════════════════════════════════════════════════════════════════
    # SERIALISATION HELPERS
    # ══════════════════════════════════════════════════════════════════════

    @staticmethod
    def _chat_message_to_dict(msg: ChatMessage) -> dict[str, Any]:
        """Convert a parsed :class:`ChatMessage` back to an API-ready dict."""
        d: dict[str, Any] = {"role": msg.role}
        if msg.content is not None:
            d["content"] = msg.content
        if msg.tool_calls:
            d["tool_calls"] = [
                {
                    "id": tc.id,
                    "type": tc.type,
                    "function": {
                        "name": tc.function.name,
                        "arguments": tc.function.arguments,
                    },
                }
                for tc in msg.tool_calls
            ]
        return d

    def to_config(self) -> AgentConfig:
        """Export the agent's declarative settings as an :class:`AgentConfig`."""
        return AgentConfig(
            name=self.name,
            role=self.role,
            goal=self.goal,
            backstory=self.backstory,
            llm_config=self.llm_config,
            max_iterations=self.max_iterations,
            allow_delegation=self.allow_delegation,
            verbose=self.verbose,
        )

    @classmethod
    def from_config(
        cls,
        config: AgentConfig,
        tools: list[AugTool] | None = None,
    ) -> AugAgent:
        """Construct an ``AugAgent`` from a serialised :class:`AgentConfig`."""
        return cls(
            name=config.name,
            role=config.role,
            goal=config.goal,
            backstory=config.backstory,
            llm_config=config.llm_config,
            max_iterations=config.max_iterations,
            allow_delegation=config.allow_delegation,
            verbose=config.verbose,
            tools=tools or [],
        )

    @property
    def message_history(self) -> list[dict[str, Any]]:
        """Read-only view of the conversation history from the last execution."""
        return list(self._message_history)

    def __repr__(self) -> str:
        return (
            f"AugAgent(name={self.name!r}, role={self.role!r}, "
            f"model={self.llm_config.model!r}, "
            f"tools={[t.name for t in self.tools]})"
        )


# Backward-compatible alias
Agent = AugAgent
