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
from augagent.tools import AugTool, Tool, aug_tool, tool, PluginRegistry
from augagent.mcp_client import MCPToolAdapter
from augagent.diff_engine import (
    apply_patch,
    generate_unified_diff,
    get_workspace_root,
    resolve_sandboxed_path,
)
from augagent.tools_file_write import (
    create_file,
    delete_lines,
    insert_at_line,
    replace_in_file,
    write_file,
)
from augagent.tools_search import (
    find_files,
    grep_search,
    list_directory,
)
from augagent.tools_git import (
    git_add,
    git_branch,
    git_commit,
    git_diff,
    git_log,
    git_status,
)

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
    "PluginRegistry",
    "MCPToolAdapter",
    # File write, edit & diff tools
    "write_file",
    "create_file",
    "replace_in_file",
    "insert_at_line",
    "delete_lines",
    "generate_unified_diff",
    "apply_patch",
    "resolve_sandboxed_path",
    "get_workspace_root",
    # Code search tools
    "grep_search",
    "find_files",
    "list_directory",
    # Git integration tools
    "git_status",
    "git_diff",
    "git_commit",
    "git_log",
    "git_branch",
    "git_add",
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
