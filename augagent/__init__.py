"""AugAgent — A multi-agent framework with intuitive DX, powered by Pydantic.

Quick Start::

    from augagent import AugAgent, AugTask, AugTeam, aug_tool, LLMConfig
    from pydantic import Field

    @aug_tool
    def greet(name: str = Field(description="Person to greet")) -> str:
        \"\"\"Greet someone by name.\"\"\"
        return f"Hello, {name}!"

    assistant = AugAgent(
        name="Assistant",
        role="Friendly Helper",
        goal="Help users with their requests",
        llm_config=LLMConfig(model="gpt-4o"),
        tools=[greet],
    )

    task = AugTask(
        description="Greet the user warmly",
        expected_output="A friendly greeting message",
        agent=assistant,
    )

    team = AugTeam(agents=[assistant], tasks=[task], verbose=True)
    result = team.kickoff()
"""

from augagent.agent import Agent, AugAgent
from augagent.models import (
    AgentConfig,
    ChatCompletion,
    ChatMessage,
    ChatToolCall,
    FunctionCall,
    LLMConfig,
    Message,
    Role,
    TaskResult,
    TaskStatus,
    TokenUsage,
    ToolCall,
    ToolResponse,
)
from augagent.task import AugTask, Task
from augagent.team import AugTeam, Process, Team
from augagent.telemetry import AgentLogger, get_logger
from augagent.tools import AugTool, Tool, aug_tool, tool

__version__ = "1.0.0"

__all__ = [
    # Core orchestration
    "AugAgent",
    "AugTask",
    "AugTeam",
    "Process",
    # Tools
    "AugTool",
    "aug_tool",
    # Backward compatibility aliases
    "Agent",
    "Task",
    "Team",
    "Tool",
    "tool",
    # Models — LLM configuration
    "LLMConfig",
    # Models — chat completion response
    "ChatCompletion",
    "ChatMessage",
    "ChatToolCall",
    "FunctionCall",
    "TokenUsage",
    # Models — internal
    "AgentConfig",
    "Message",
    "Role",
    "TaskResult",
    "TaskStatus",
    "ToolCall",
    "ToolResponse",
    # Telemetry
    "AgentLogger",
    "get_logger",
    # Metadata
    "__version__",
]
