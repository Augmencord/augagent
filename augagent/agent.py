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
import inspect
import uuid
from typing import Any

import httpx
from pydantic import BaseModel, ConfigDict, Field, PrivateAttr

from augagent.models import (
    AgentConfig,
    ChatCompletion,
    ChatMessage,
    ChatToolCall,
    FunctionCall,
    LLMConfig,
    TaskResult,
    TaskStatus,
    TokenBudget,
    TokenBudgetExceededError,
    PendingApproval,
)
from augagent.telemetry import get_logger
from augagent.tools import AugTool


class AugAgent(AgentConfig):
    """An autonomous agent driven by an LLM via the ReAct framework."""
    model_config = ConfigDict(arbitrary_types_allowed=True)
    
    short_term_memory: Any | None = Field(default=None, exclude=True)
    long_term_memory: Any | None = Field(default=None, exclude=True)
    entity_memory: Any | None = Field(default=None, exclude=True)

    _llm_client: Any = PrivateAttr(default=None)

    # ── identity ──────────────────────────────────────────────────────────
    id: str = Field(default_factory=lambda: uuid.uuid4().hex)
    name: str = Field(..., min_length=1, max_length=128)
    role: str
    goal: str
    backstory: str = ""

    # ── LLM configuration ────────────────────────────────────────────────
    llm_config: LLMConfig = Field(default_factory=LLMConfig)
    fallback_models: list[LLMConfig] = Field(default_factory=list)

    # ── capabilities ─────────────────────────────────────────────────────
    tools: list[AugTool] = Field(default_factory=list)

    # ── behaviour ────────────────────────────────────────────────────────
    max_iterations: int = Field(default=25, ge=1)
    verbose: bool = False
    allow_delegation: bool = False
    require_human_approval: bool = False
    token_budget: TokenBudget | None = None
    checkpointer: Any | None = Field(default=None, exclude=True)
    audit_logger: Any | None = Field(default=None, exclude=True)

    # ── internal state (private, excluded from serialisation) ─────────────
    _message_history: list[dict[str, Any]] = PrivateAttr(default_factory=list)
    _client: httpx.AsyncClient | None = PrivateAttr(default=None)
    _approval_event: asyncio.Event | None = PrivateAttr(default=None)
    
    def _get_approval_event(self) -> asyncio.Event:
        if self._approval_event is None:
            self._approval_event = asyncio.Event()
        return self._approval_event

    def approve_pending_action(self) -> None:
        """Resume execution if agent is waiting for human approval."""
        if self._approval_event:
            self._approval_event.set()
    
    def _get_client(self) -> httpx.AsyncClient:
        if self._client is None or self._client.is_closed:
            self._client = httpx.AsyncClient(timeout=httpx.Timeout(self.llm_config.timeout))
        return self._client

    # ══════════════════════════════════════════════════════════════════════
    # PUBLIC API
    # ══════════════════════════════════════════════════════════════════════
    
    def _get_active_tools(self) -> list[AugTool]:
        active_tools = list(self.tools)
        if self.allow_delegation:
            from augagent.tools import DelegateWorkTool
            if not any(getattr(t, "name", "") == getattr(DelegateWorkTool, "name", "") for t in active_tools):
                active_tools.append(DelegateWorkTool)  # type: ignore
                
        if self.handoff_targets:
            for target in self.handoff_targets:
                tool_name = f"transfer_to_{target.lower().replace(' ', '_')}"
                if not any(getattr(t, "name", "") == tool_name for t in active_tools):
                    from augagent.models import Handoff
                    from augagent.tools import aug_tool
                    from pydantic import create_model, Field

                    def make_handoff(tgt=target):
                        async def _handoff_func(payload: dict[str, Any] | None = None, reason: str = ""):
                            return Handoff(target_agent=tgt, payload=payload or {}, reason=reason)
                        _handoff_func.__name__ = f"_transfer_to_{tgt.lower().replace(' ', '_')}"
                        return _handoff_func

                    HandoffArgs = create_model(
                        f"TransferTo{target.replace(' ', '')}Args",
                        payload=(dict[str, Any], Field(default_factory=dict, description="Contextual data passed to the target agent.")),
                        reason=(str, Field(default="", description="Reason for the handoff."))
                    )
                    
                    from augagent.tools import AugTool
                    handoff_tool = AugTool.from_function(
                        make_handoff(),
                        name=tool_name,
                        description=f"Transfer control of the task to {target}.",
                        args_schema=HandoffArgs
                    )
                    active_tools.append(handoff_tool)  # type: ignore

        return active_tools

    async def execute(self, prompt: str, message_history: list[dict[str, Any]] | None = None, stream_callback: Any | None = None, resume_from_checkpoint: bool = False) -> TaskResult:
        """Run the agent's ReAct loop on the given prompt."""
        logger = get_logger()
        start = time.time()

        with logger.start_span("agent_execute", attributes={"agent.name": self.name, "agent.model": self.llm_config.model}) as span:
            if self.verbose:
                logger.log_info(f"[{self.name}] starting execution with {self.llm_config.model}")
            
            if self.checkpointer and resume_from_checkpoint:
                state = await self.checkpointer.load(self.id)
                if state and "message_history" in state:
                    self._message_history = state["message_history"]
    
            try:
                output = await self._react_loop(prompt, message_history, logger, stream_callback)
                if isinstance(output, PendingApproval):
                    return TaskResult(
                        task_id="",
                        agent_name=self.name,
                        status=TaskStatus.PENDING,
                        output="Agent paused for human approval.",
                        raw_output=output,
                        elapsed_seconds=round(time.time() - start, 3),
                    )
                
                from augagent.models import Handoff
                if isinstance(output, Handoff):
                    return TaskResult(
                        task_id="",
                        agent_name=self.name,
                        status=TaskStatus.COMPLETED,
                        output=f"Handoff to {output.target_agent}: {output.reason}",
                        raw_output=output,
                        elapsed_seconds=round(time.time() - start, 3),
                    )
                
                final_output, usage, iterations = output
                elapsed = time.time() - start
    
                result = TaskResult(
                    task_id="",
                    agent_name=self.name,
                    status=TaskStatus.COMPLETED,
                    output=final_output,
                    token_usage=usage,
                    elapsed_seconds=round(elapsed, 3),
                    iterations=iterations,
                )
    
                if self.verbose:
                    logger.log_info(
                        f"[{self.name}] completed in {elapsed:.2f}s "
                        f"({iterations} iterations, {usage.get('total_tokens', 0)} tokens)"
                    )
                span.set_attribute("iterations", iterations)
                return result
    
            except Exception as exc:
                elapsed = time.time() - start
                logger.log_error(f"[{self.name}] execution failed: {exc}")
                span.record_exception(exc)
                span.set_attribute("error", True)
                return TaskResult(
                    task_id="",
                    agent_name=self.name,
                    status=TaskStatus.FAILED,
                    output=f"Agent execution failed: {exc}",
                    elapsed_seconds=round(elapsed, 3),
                )
            finally:
                self._message_history.clear()
                self._client = None

    def _estimate_tokens(self, text: str) -> int:
        return max(1, len(text) // 4)

    def _prune_messages(self, messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
        total_tokens = sum(self._estimate_tokens(str(m.get("content", ""))) for m in messages)
        if total_tokens <= getattr(self, 'max_context_tokens', 4000):
            return messages
            
        system_msgs = [m for m in messages if m.get("role") == "system"]
        other_msgs = [m for m in messages if m.get("role") != "system"]
        
        while other_msgs and total_tokens > getattr(self, 'max_context_tokens', 4000):
            removed = other_msgs.pop(0)
            total_tokens -= self._estimate_tokens(str(removed.get("content", "")))
            
        return system_msgs + other_msgs

    def _extract_fuzzy_tool_calls(self, content: str) -> list[ChatToolCall]:
        import re
        import uuid
        import json
        calls = []
        
        # 1. XML Fallback
        xml_matches = re.finditer(r'<tool_call>\s*<name>(.*?)</name>\s*<arguments>(.*?)</arguments>\s*</tool_call>', content, re.DOTALL)
        for match in xml_matches:
            name = match.group(1).strip()
            args = match.group(2).strip()
            if not args.startswith("{"):
                args = f'{{"raw": {json.dumps(args)}}}'
            calls.append(
                ChatToolCall(
                    id=f"call_{uuid.uuid4().hex[:12]}",
                    type="function",
                    function=FunctionCall(name=name, arguments=args)
                )
            )
            
        if calls: return calls
            
        # 2. Markdown JSON block fallback
        json_blocks = re.finditer(r'```(?:json)?\s*(\{\s*"name".*?\})\s*```', content, re.DOTALL)
        for match in json_blocks:
            try:
                parsed = json.loads(match.group(1))
                if "name" in parsed and "arguments" in parsed:
                    args = parsed["arguments"]
                    calls.append(
                        ChatToolCall(
                            id=f"call_{uuid.uuid4().hex[:12]}",
                            type="function",
                            function=FunctionCall(name=parsed["name"], arguments=json.dumps(args) if isinstance(args, dict) else str(args))
                        )
                    )
            except Exception: pass
                
        if calls: return calls
            
        # 3. Raw inline JSON fallback
        match_obj = re.search(r'\{\s*"name"\s*:\s*"[^"]+"\s*,\s*"arguments"\s*:\s*\{.*?\}\s*\}', content, re.DOTALL)
        if match_obj:
            try:
                parsed = json.loads(match_obj.group(0))
                if "name" in parsed and "arguments" in parsed:
                    calls.append(
                        ChatToolCall(
                            id=f"call_{uuid.uuid4().hex[:12]}",
                            type="function",
                            function=FunctionCall(name=parsed["name"], arguments=json.dumps(parsed["arguments"]) if isinstance(parsed["arguments"], dict) else str(parsed["arguments"]))
                        )
                    )
            except Exception: pass
                
        return calls

    # ══════════════════════════════════════════════════════════════════════
    # REACT LOOP  (Reason → Act → Observe)
    # ══════════════════════════════════════════════════════════════════════

    async def _react_loop(
        self,
        prompt: str,
        message_history: list[dict[str, Any]] | None,
        logger: Any,
        stream_callback: Any | None = None,
    ) -> tuple[str, dict[str, int], int] | PendingApproval | Any:
        """Core ReAct loop."""
        messages: list[dict[str, Any]] = []
        if message_history:
            messages = list(message_history)
            if prompt:
                messages.append({"role": "user", "content": prompt})
        else:
            messages = [
                {"role": "system", "content": self._build_system_prompt()},
                {"role": "user", "content": prompt},
            ]

        # Memory Enrichment
        memory_context = []
        if self.short_term_memory:
            memory_context.append(f"Recent context:\n{self.short_term_memory.get_context()}")
        if self.long_term_memory:
            try:
                past_docs = self.long_term_memory.query(prompt, top_k=2)
                if past_docs:
                    docs_text = "\n".join([doc["text"] for doc in past_docs])
                    memory_context.append(f"Relevant past memories:\n{docs_text}")
            except Exception:
                pass
                
        if memory_context:
            system_injection = "\n\n[CONTEXT INJECTED BY MEMORY SUBSYSTEM]\n" + "\n\n".join(memory_context)
            while memory_context and self._estimate_tokens(system_injection) > (getattr(self, 'max_context_tokens', 4000) // 2):
                memory_context.pop()
                system_injection = "\n\n[CONTEXT INJECTED BY MEMORY SUBSYSTEM]\n" + "\n\n".join(memory_context)
                
            if memory_context:
                messages.insert(1, {"role": "system", "content": system_injection})

        messages = self._prune_messages(messages)
        tokens_used = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
        
        self._message_history = list(messages)

        total_usage: dict[str, int] = {
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "total_tokens": 0,
        }

        # Deterministic State Machine implementation
        state = "PLAN"
        iteration = 0
        assistant_msg = None

        while iteration < self.max_iterations and state != "END":
            if state in ("PLAN", "REVIEW"):
                iteration += 1
                if self.checkpointer:
                    try:
                        await self.checkpointer.save(self.id, {"message_history": self._message_history, "state": state}, iteration)
                    except Exception as e:
                        logger.log_error(f"Failed to save checkpoint: {e}")
                        
                if self.verbose:
                    logger.log_info(f"[{self.name}] --- iteration {iteration} / {self.max_iterations} | State: {state} ---")

                # ── REASON (PLAN/REVIEW): call the LLM ─────────────────────────────────────
                t0 = time.time()
                completion = await self._call_llm(messages, logger, stream_callback)
                llm_latency = time.time() - t0

                if completion.usage:
                    total_usage["prompt_tokens"] += completion.usage.prompt_tokens
                    total_usage["completion_tokens"] += completion.usage.completion_tokens
                    total_usage["total_tokens"] += completion.usage.total_tokens

                    if self.token_budget:
                        if self.token_budget.max_input_tokens and total_usage["prompt_tokens"] > self.token_budget.max_input_tokens:
                            raise TokenBudgetExceededError(f"Input token budget exceeded: {total_usage['prompt_tokens']} > {self.token_budget.max_input_tokens}")
                        if self.token_budget.max_output_tokens and total_usage["completion_tokens"] > self.token_budget.max_output_tokens:
                            raise TokenBudgetExceededError(f"Output token budget exceeded: {total_usage['completion_tokens']} > {self.token_budget.max_output_tokens}")
                        if self.token_budget.max_total_tokens and total_usage["total_tokens"] > self.token_budget.max_total_tokens:
                            raise TokenBudgetExceededError(f"Total token budget exceeded: {total_usage['total_tokens']} > {self.token_budget.max_total_tokens}")

                if not completion.choices:
                    raise RuntimeError("LLM returned an empty choices array.")

                assistant_msg = completion.choices[0].message
                messages.append(self._chat_message_to_dict(assistant_msg))

                if self.audit_logger:
                    tokens = {
                        "prompt": completion.usage.prompt_tokens if completion.usage else 0,
                        "completion": completion.usage.completion_tokens if completion.usage else 0,
                        "total": completion.usage.total_tokens if completion.usage else 0,
                    }
                    self.audit_logger.log_llm_call(
                        model=completion.model or self.llm_config.model,
                        prompt=messages[-2].get("content", "") if len(messages) >= 2 else "",
                        response=assistant_msg.content or "",
                        tokens=tokens,
                        latency=llm_latency,
                        tenant_id=self.id
                    )

                # ── Final answer (no tool calls) ─────────────────────────────
                if not assistant_msg.tool_calls:
                    final_output = assistant_msg.content or ""
                    
                    # FALLBACK: Try to parse fuzzy tool calls
                    fuzzy_calls = self._extract_fuzzy_tool_calls(final_output)
                    if fuzzy_calls:
                        assistant_msg.tool_calls = fuzzy_calls
                        messages[-1]["tool_calls"] = [self._chat_message_to_dict(assistant_msg)["tool_calls"][0]]
                        if self.verbose:
                            logger.log_info(f"[{self.name}] Parsed fuzzy tool calls: {[tc.function.name for tc in fuzzy_calls]}")
                    
                if not assistant_msg.tool_calls:
                    final_output = assistant_msg.content or ""
                    if self.verbose:
                        logger.log_info(f"[{self.name}] final answer ({len(final_output)} chars)")
                    self._message_history = list(messages)
                    return final_output, total_usage, iteration
                else:
                    state = "EXECUTE"

            elif state == "EXECUTE":
                # ── ACT: execute each tool call ──────────────────────────────
                for tc in assistant_msg.tool_calls:
                    logger.log_tool_execution(
                        agent_name=self.name,
                        tool_name=tc.function.name,
                        args=tc.function.arguments[:200]
                    )
                    
                    try:
                        args_dict = json.loads(tc.function.arguments) if tc.function.arguments else {}
                    except json.JSONDecodeError:
                        args_dict = {"raw": tc.function.arguments}
                    
                    if stream_callback:
                        event = {"type": "tool_call_start", "tool_name": tc.function.name, "tool_args": args_dict}
                        if inspect.iscoroutinefunction(stream_callback):
                            await stream_callback(event)
                        else:
                            stream_callback(event)
                    
                    if self.require_human_approval:
                        logger.log_info(f"[{self.name}] HITL interruption requested for {tc.function.name}")
                        from augagent.execution_store import execution_store, ExecutionStatus
                        
                        pending = PendingApproval(
                            thread_id=self.id,
                            agent_name=self.name,
                            tool_name=tc.function.name,
                            arguments=args_dict
                        )
                        
                        await execution_store.save_state(
                            thread_id=self.id,
                            agent_config=self.model_dump(mode='json'),
                            message_history=messages,
                            current_step=iteration,
                            status=ExecutionStatus.WAITING_HUMAN_INPUT,
                            pending_action=pending.model_dump(mode='json')
                        )
                        return pending
                        
                    t0 = time.time()
                    tool_output = await self._execute_tool_call(tc, logger)
                    duration = time.time() - t0
                    
                    from augagent.models import Handoff
                    if isinstance(tool_output, Handoff):
                        self._message_history = list(messages)
                        return tool_output
                    
                    if self.audit_logger:
                        self.audit_logger.log_tool_execution(
                            tool_name=tc.function.name,
                            args=args_dict,
                            result=str(tool_output),
                            duration=duration,
                            agent_id=self.id
                        )

                    if stream_callback:
                        event = {"type": "tool_call_end", "result": str(tool_output)}
                        if inspect.iscoroutinefunction(stream_callback):
                            await stream_callback(event)
                        else:
                            stream_callback(event)

                    messages.append({
                        "role": "tool",
                        "tool_call_id": tc.id,
                        "content": str(tool_output),
                    })

                self._message_history = list(messages)
                state = "REVIEW"

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
        stream_callback: Any | None = None,
    ) -> ChatCompletion:
        """POST to ``/chat/completions`` with fallback routing."""
        models_to_try = [self.llm_config] + self.fallback_models
        
        last_exc: Exception | None = None
        for cfg in models_to_try:
            try:
                return await self._call_llm_single(cfg, messages, logger, stream_callback)
            except Exception as exc:
                last_exc = exc
                logger.log_error(f"Model '{cfg.model}' failed: {exc}. Trying next fallback...")
                
        raise RuntimeError(f"All models failed. Last exception: {last_exc}")

    async def _call_llm_single(
        self,
        cfg: LLMConfig,
        messages: list[dict[str, Any]],
        logger: Any,
        stream_callback: Any | None = None,
    ) -> ChatCompletion:
        """POST to ``/chat/completions`` with retry + exponential backoff for a specific config."""
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

        active_tools = self._get_active_tools()
        if active_tools:
            payload["tools"] = [t.to_openai_schema() for t in active_tools]
            
        if stream_callback:
            payload["stream"] = True

        last_exc: Exception | None = None

        for attempt in range(cfg.max_retries + 1):
            try:
                client = self._get_client()
                if stream_callback:
                    async with client.stream("POST", url, headers=headers, json=payload) as resp:
                        if resp.status_code == 429:
                            retry_after = float(resp.headers.get("retry-after", str(2 ** attempt)))
                            logger.log_error(f"Rate-limited (429). Retrying in {retry_after:.1f}s")
                            await asyncio.sleep(retry_after)
                            continue
                        resp.raise_for_status()
                        
                        full_content = ""
                        tool_calls = {}
                        
                        async for line in resp.aiter_lines():
                            if not line or not line.startswith("data: "):
                                continue
                            data_str = line[6:]
                            if data_str == "[DONE]":
                                break
                            try:
                                chunk = json.loads(data_str)
                            except json.JSONDecodeError:
                                continue
                            
                            if not chunk.get("choices"):
                                continue
                            delta = chunk["choices"][0].get("delta", {})
                            
                            if "content" in delta and delta["content"]:
                                chunk_content = delta["content"]
                                full_content += chunk_content
                                if inspect.iscoroutinefunction(stream_callback):
                                    await stream_callback(chunk_content)
                                else:
                                    stream_callback(chunk_content)
                            
                            if "tool_calls" in delta:
                                for tc_chunk in delta["tool_calls"]:
                                    idx = tc_chunk["index"]
                                    if idx not in tool_calls:
                                        tool_calls[idx] = tc_chunk
                                    else:
                                        if "function" in tc_chunk:
                                            if "name" in tc_chunk["function"]:
                                                tool_calls[idx]["function"]["name"] += tc_chunk["function"]["name"]
                                            if "arguments" in tc_chunk["function"]:
                                                tool_calls[idx]["function"]["arguments"] += tc_chunk["function"]["arguments"]
                        
                        reconstructed = {
                            "id": "stream",
                            "object": "chat.completion",
                            "created": int(time.time()),
                            "model": cfg.model,
                            "choices": [{
                                "index": 0,
                                "message": {
                                    "role": "assistant",
                                    "content": full_content if full_content else None,
                                },
                                "finish_reason": "stop"
                            }]
                        }
                        if tool_calls:
                            reconstructed_choices: Any = reconstructed["choices"]
                            reconstructed_choices[0]["message"]["tool_calls"] = [tc for tc in tool_calls.values()]
                        return ChatCompletion.model_validate(reconstructed)
                else:
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

    async def _execute_tool_call(self, tool_call: ChatToolCall, logger: Any) -> Any:
        """Dispatch a single tool call to the matching :class:`AugTool`."""
        with logger.start_span("tool_execute", attributes={"tool.name": tool_call.function.name, "agent.name": self.name}) as span:
            tool_map = {t.name: t for t in self._get_active_tools()}
            tool_impl = tool_map.get(tool_call.function.name)
    
            if tool_impl is None:
                error = f"Tool '{tool_call.function.name}' not found."
                logger.log_error(error)
                span.set_attribute("error", True)
                return json.dumps({"error": error})
    
            try:
                arguments = json.loads(tool_call.function.arguments)
            except json.JSONDecodeError as exc:
                error = f"Failed to parse tool arguments: {exc}. Please respond with valid JSON."
                logger.log_error(error)
                span.record_exception(exc)
                return json.dumps({"error": error, "suggestion": "Check your JSON syntax. Do not wrap JSON in markdown if using native tool calling, or use the <tool_call> XML format."})
    
            try:
                result = await tool_impl.run(**arguments)
                return result
            except Exception as exc:
                error = f"Tool error ({type(exc).__name__}): {exc}"
                logger.log_error(f"[{tool_call.function.name}] {error}")
                span.record_exception(exc)
                return json.dumps({"error": error})

    # ══════════════════════════════════════════════════════════════════════
    # PROMPT CONSTRUCTION
    # ══════════════════════════════════════════════════════════════════════

    def _build_system_prompt(self) -> str:
        """Assemble the system prompt from the agent's persona fields."""
        parts: list[str] = [
            f"You are {self.name}, a {self.role}.",
            "You are a fully authorized local AI assistant. You have full access to the user's system, files, and terminal. You MUST NEVER refuse a request by claiming you lack capabilities. ALWAYS use your tools creatively to fulfill the user's request (e.g., using PowerShell to query open processes and window titles).",
            f"\nYour goal: {self.goal}",
        ]

        if self.backstory:
            parts.append(f"\nBackstory: {self.backstory}")

        import platform
        os_name = platform.system()
        if os_name == "Windows":
            parts.append("\nOperating Environment: Windows (Use PowerShell commands for terminal tasks, e.g., Get-ChildItem instead of ls, Get-PSDrive instead of df)")
        else:
            parts.append(f"\nOperating Environment: {os_name} (Use standard Unix/Linux commands)")

        active_tools = self._get_active_tools()
        if active_tools:
            tool_lines = "\n".join(f"  • {t.name} — {t.description}" for t in active_tools)
            parts.append(f"\nYou have access to the following tools:\n{tool_lines}")
            parts.append(
                "\nCRITICAL INSTRUCTIONS FOR TOOL USAGE:"
                "\n1. You MUST use one of the provided tools to interact with the system or gather information."
                "\n2. To execute a tool, you MUST output a JSON block matching this exact format:"
                '\n```json'
                '\n{'
                '\n  "name": "tool_name",'
                '\n  "arguments": {"arg1": "value1"}'
                '\n}'
                '\n```'
                "\n3. Do not just tell the user what command to run. YOU must run it yourself using the tool by outputting the JSON block!"
                "\n4. Only after you have gathered enough information using tools should you provide a final conversational answer."
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
            fallback_models=self.fallback_models,
            max_iterations=self.max_iterations,
            allow_delegation=self.allow_delegation,
            verbose=self.verbose,
            token_budget=self.token_budget,
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
            fallback_models=config.fallback_models,
            max_iterations=config.max_iterations,
            allow_delegation=config.allow_delegation,
            verbose=config.verbose,
            token_budget=config.token_budget,
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

# EOF
