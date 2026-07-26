"""Team — multi-agent orchestration.

An :class:`AugTeam` groups agents and tasks together. In sequential mode,
it executes tasks one by one, automatically chaining the output of one task
as context for the next.

Example::

    from augagent import AugAgent, AugTask, AugTeam, LLMConfig

    config = LLMConfig(model="gpt-4o")
    researcher = AugAgent(name="Researcher", role="Researcher", goal="Find facts", llm_config=config)
    writer     = AugAgent(name="Writer", role="Writer", goal="Write articles", llm_config=config)

    t1 = AugTask(description="Research AI trends", expected_output="Bullet points", agent=researcher)
    t2 = AugTask(description="Write blog post", expected_output="Markdown article", agent=writer)

    # Output from t1 is automatically passed as context to t2 in sequential mode
    team = AugTeam(
        agents=[researcher, writer],
        tasks=[t1, t2],
    )
    results = team.kickoff()
"""

from __future__ import annotations

import asyncio
import time
from enum import Enum
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from augagent.agent import AugAgent
from augagent.models import TaskResult, TaskStatus
from augagent.task import AugTask
from augagent.telemetry import get_logger


class Process(str, Enum):
    """Orchestration strategy for a :class:`AugTeam`."""

    SEQUENTIAL = "sequential"


class AugTeam(BaseModel):
    """A coordinated group of :class:`AugAgent` instances executing :class:`AugTask` objects.

    Parameters
    ----------
    agents:
        The roster of agents available to this team.
    tasks:
        An ordered list of tasks to execute.
    process:
        The orchestration strategy (currently only ``"sequential"``).
    verbose:
        Enable detailed logging.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    agents: list[AugAgent] = Field(..., min_length=1)
    tasks: list[AugTask] = Field(..., min_length=1)
    process: Process = Process.SEQUENTIAL
    verbose: bool = False

    # -- public API ---------------------------------------------------------

    def kickoff(self, inputs: dict[str, Any] | None = None) -> list[TaskResult]:
        """Execute all tasks sequentially and return results.

        Passes the output of each completed task as context to the next task.
        Handles both sync and async contexts transparently.
        """
        if inputs:
            for task in self.tasks:
                task.interpolate_inputs(inputs)

        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            loop = None

        if loop and loop.is_running():
            import concurrent.futures

            with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
                return pool.submit(asyncio.run, self._run()).result()
        return asyncio.run(self._run())

    async def akickoff(self, inputs: dict[str, Any] | None = None) -> list[TaskResult]:
        """Async version of :meth:`kickoff`."""
        if inputs:
            for task in self.tasks:
                task.interpolate_inputs(inputs)
        return await self._run()

    # -- orchestration strategies -------------------------------------------

    async def _run(self) -> list[TaskResult]:
        logger = get_logger()
        start = time.time()

        if self.verbose:
            logger.log_info(f"Team kickoff started with {len(self.tasks)} tasks.")

        results = await self._run_sequential(logger)

        elapsed = time.time() - start
        completed = sum(1 for r in results if r.status == TaskStatus.COMPLETED)
        failed = sum(1 for r in results if r.status == TaskStatus.FAILED)

        if self.verbose:
            logger.log_info(f"Team finished in {elapsed:.2f}s ({completed} completed, {failed} failed).")

        return results

    async def _run_sequential(self, logger: Any) -> list[TaskResult]:
        """Execute tasks one-by-one, passing outputs as context to the next."""
        results: list[TaskResult] = []

        for i, task in enumerate(self.tasks):
            # Automatically chain context: if this isn't the first task, and the previous task
            # successfully generated output, append it to this task's context.
            if i > 0 and results[-1].status == TaskStatus.COMPLETED:
                prev_task = self.tasks[i - 1]
                if prev_task not in task.context:
                    task.context.append(prev_task)

            executor = task.agent or self.agents[0]

            logger.log_handoff(from_agent="Manager", to_agent=executor.name, task_desc=task.description)

            result = await task.execute(agent=executor)
            results.append(result)

            if result.status == TaskStatus.COMPLETED:
                logger.log_info(f"Task completed successfully. Output length: {len(result.output)} chars.")
            else:
                logger.log_error(f"Task failed: {result.output}")

        return results

    # -- utilities ----------------------------------------------------------

    def get_agent(self, name: str) -> AugAgent | None:
        """Look up an agent by name."""
        for agent in self.agents:
            if agent.name == name:
                return agent
        return None

    def __repr__(self) -> str:
        return (
            f"AugTeam(agents={[a.name for a in self.agents]}, "
            f"tasks={len(self.tasks)}, process={self.process.value!r})"
        )


# Backward compatibility alias
Team = AugTeam
