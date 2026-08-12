"""Task — a discrete unit of work assigned to an AugAgent.

Tasks describe *what* needs to be done, *what* the expected output looks
like, and *which* agent is responsible.  They are the currency of work
within an :class:`~augagent.team.AugTeam`.

Example::

    from augagent import AugAgent, AugTask, LLMConfig

    writer = AugAgent(
        name="Writer",
        role="Content Writer",
        goal="Write great copy",
        llm_config=LLMConfig(model="gpt-4o"),
    )

    task = AugTask(
        description="Write a blog post about AI agents",
        expected_output="A 500-word blog post in Markdown format",
        agent=writer,
    )
"""

from __future__ import annotations

import uuid
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from augagent.agent import AugAgent
from augagent.models import TaskResult, TaskStatus
from augagent.telemetry import get_logger


class AugTask(BaseModel):
    """A unit of work to be executed by an :class:`AugAgent`.

    Parameters
    ----------
    description:
        A natural-language description of what needs to be done.
    expected_output:
        A description of the ideal output format and content.
    agent:
        The agent responsible for executing this task.
    context:
        Optional list of predecessor :class:`AugTask` instances whose outputs
        are prepended to this task's prompt, enabling information flow
        across sequential tasks.
    async_execution:
        If ``True``, the task is eligible for concurrent execution.
    output_json:
        Optional Pydantic model class to validate and parse the output.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    id: str = Field(default_factory=lambda: uuid.uuid4().hex[:12])
    description: str = Field(..., min_length=1)
    expected_output: str = Field(default="")
    agent: AugAgent | None = None
    context: list["AugTask"] = Field(default_factory=list)
    async_execution: bool = False
    output_json: Any = Field(default=None, exclude=True, repr=False)

    # -- runtime state (excluded from serialisation) ------------------------
    status: TaskStatus = Field(default=TaskStatus.PENDING, exclude=True)
    result: TaskResult | None = Field(default=None, exclude=True)

    # -- public API ---------------------------------------------------------

    async def execute(self, agent: AugAgent | None = None) -> TaskResult:
        """Run the task using its assigned agent (or an override).

        Raises
        ------
        ValueError
            If no agent is assigned or provided.
        """
        executor = agent or self.agent
        if executor is None:
            raise ValueError(
                f"Task '{self.id}' has no assigned agent. "
                "Pass an agent explicitly or set task.agent."
            )

        logger = get_logger()
        self.status = TaskStatus.IN_PROGRESS

        # Build the full prompt, incorporating context from upstream tasks
        prompt = self._build_prompt()

        try:
            result = await executor.execute(prompt)
            # Re-stamp with this task's id
            result = result.model_copy(update={"task_id": self.id})
            self.result = result
            self.status = TaskStatus.COMPLETED

            # Optional JSON output validation
            if self.output_json is not None:
                try:
                    self.output_json.model_validate_json(result.output)
                except Exception as exc:
                    logger.log_error(f"JSON validation failed: {exc}")
                    
            try:
                from augagent.memory import global_long_term_memory
                global_long_term_memory.add_document(
                    text=f"Task: {self.description}\\nResult: {result.output}",
                    metadata={"task_id": self.id, "agent": executor.name}
                )
            except ImportError:
                pass

            return result

        except Exception as exc:
            self.status = TaskStatus.FAILED
            logger.log_error(f"Task failed: {exc}")
            return TaskResult(
                task_id=self.id,
                agent_name=executor.name,
                status=TaskStatus.FAILED,
                output=f"Task failed: {exc}",
            )

    # -- internals ----------------------------------------------------------

    def _build_prompt(self) -> str:
        """Compose the task prompt, prepending upstream context if present."""
        parts: list[str] = []

        if self.context:
            parts.append("## Context from previous tasks\n")
            for ctx_task in self.context:
                if ctx_task.result and ctx_task.result.output:
                    parts.append(
                        f"### Output from '{ctx_task.description}'\n{ctx_task.result.output}\n"
                    )
            parts.append("---\n")
            
        try:
            from augagent.memory import global_long_term_memory
            historical_context = global_long_term_memory.search(self.description)
            if historical_context:
                parts.append("## Relevant Historical Context (RAG)\n")
                for item in historical_context:
                    parts.append(f"- {item['text']}\n")
                parts.append("---\n")
        except ImportError:
            pass

        parts.append(f"## Task\n{self.description}\n")
        if self.expected_output:
            parts.append(f"## Expected Output\n{self.expected_output}\n")

        return "\n".join(parts)

    def interpolate_inputs(self, inputs: dict[str, Any]) -> None:
        """Interpolate inputs into description and expected_output."""
        self.description = self.description.format(**inputs)
        if self.expected_output:
            self.expected_output = self.expected_output.format(**inputs)

    def __repr__(self) -> str:
        agent_name = self.agent.name if self.agent else "unassigned"
        return (
            f"AugTask(id={self.id!r}, agent={agent_name!r}, "
            f"status={self.status.value!r})"
        )

# Backward compatibility alias
Task = AugTask
