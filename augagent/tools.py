"""Tool abstraction layer with Pydantic-powered schema generation.

:class:`AugTool` wraps plain Python functions and **automatically generates
OpenAI-compatible function-calling JSON schemas** from the function's type
hints.  If a parameter's default is a :func:`pydantic.Field`, its
``description``, ``ge``/``le`` constraints, and other metadata are carried
through into the generated schema.

Quick start::

    from augagent import aug_tool
    from pydantic import Field

    @aug_tool
    def search_web(
        query: str = Field(description="The search query"),
        max_results: int = Field(default=5, description="Max results to return"),
    ) -> str:
        \"\"\"Search the web and return relevant results.\"\"\"
        return f"Results for: {query}"

    # Inspect the generated OpenAI schema
    print(search_web.to_openai_schema())

    # Or bring your own Pydantic model for maximum control:
    from pydantic import BaseModel

    class CalcArgs(BaseModel):
        expression: str = Field(description="A mathematical expression")

    @aug_tool(args_schema=CalcArgs)
    def calculate(expression: str) -> str:
        \"\"\"Evaluate a mathematical expression safely.\"\"\"
        return str(eval(expression))
"""

from __future__ import annotations

import inspect
import json
from typing import Any, Callable, get_type_hints

from pydantic import BaseModel, ConfigDict, Field, create_model
from pydantic.fields import FieldInfo


# ═══════════════════════════════════════════════════════════════════════════
# AugTool
# ═══════════════════════════════════════════════════════════════════════════

class AugTool(BaseModel):
    """A callable tool backed by a Pydantic args schema.

    Wraps a Python function and derives an OpenAI-compatible
    function-calling JSON schema from its :attr:`args_schema` (a Pydantic
    model).  Arguments are **validated through Pydantic before invocation**,
    giving you runtime type safety for free.

    There are three ways to create an ``AugTool``:

    1. **Decorator** — ``@aug_tool`` (simplest, recommended)
    2. **Factory** — ``AugTool.from_function(fn)``
    3. **Manual** — ``AugTool(name=..., args_schema=MyModel, func=fn)``
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    name: str = Field(..., description="Unique tool name (sent to the LLM).")
    description: str = Field(default="", description="Human-readable purpose of the tool.")
    args_schema: type[BaseModel] = Field(
        ...,
        description="Pydantic model class that describes the tool's parameters.",
        exclude=True,
    )
    func: Callable[..., Any] = Field(..., exclude=True, repr=False)

    # ── OpenAI schema generation ──────────────────────────────────────────

    def to_openai_schema(self) -> dict[str, Any]:
        """Generate an OpenAI-compatible function-calling tool definition.

        Returns a ``dict`` that can be passed directly inside the ``tools``
        array of a ``/chat/completions`` request::

            {
                "type": "function",
                "function": {
                    "name": "search_web",
                    "description": "Search the web …",
                    "parameters": { <JSON Schema> }
                }
            }

        The ``parameters`` block is produced by
        :meth:`pydantic.BaseModel.model_json_schema`, so every ``Field``
        constraint (``description``, ``ge``, ``le``, ``pattern``, …) is
        automatically propagated.
        """
        raw_schema = self.args_schema.model_json_schema()

        # Strip keys that Pydantic adds but OpenAI does not expect.
        raw_schema.pop("title", None)
        raw_schema.pop("$defs", None)
        raw_schema.pop("definitions", None)

        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": raw_schema,
            },
        }

    # ── Execution ─────────────────────────────────────────────────────────

    async def run(self, **kwargs: Any) -> str:
        """Validate *kwargs* through :attr:`args_schema`, then call the function.

        Coroutine functions are automatically awaited.  Non-string return
        values are JSON-serialised.
        """
        validated = self.args_schema.model_validate(kwargs)
        result = self.func(**validated.model_dump())
        if inspect.isawaitable(result):
            result = await result
        return result if isinstance(result, str) else json.dumps(result, default=str)

    def run_sync(self, **kwargs: Any) -> str:
        """Synchronous variant of :meth:`run` (no coroutine support)."""
        validated = self.args_schema.model_validate(kwargs)
        result = self.func(**validated.model_dump())
        return result if isinstance(result, str) else json.dumps(result, default=str)

    # ── Factory ───────────────────────────────────────────────────────────

    @classmethod
    def from_function(
        cls,
        func: Callable[..., Any],
        *,
        name: str | None = None,
        description: str | None = None,
        args_schema: type[BaseModel] | None = None,
    ) -> AugTool:
        """Construct an ``AugTool`` from a plain Python function.

        If *args_schema* is ``None``, a Pydantic model is **dynamically
        generated** from the function's type hints and defaults
        (including ``pydantic.Field`` defaults).
        """
        tool_name = name or func.__name__
        tool_desc = description or inspect.getdoc(func) or ""
        schema = args_schema or _create_args_model(func, tool_name)

        return cls(
            name=tool_name,
            description=tool_desc,
            args_schema=schema,
            func=func,
        )

    def __repr__(self) -> str:
        return f"AugTool(name={self.name!r})"


# ═══════════════════════════════════════════════════════════════════════════
# Dynamic Pydantic model creation from function signatures
# ═══════════════════════════════════════════════════════════════════════════

def _create_args_model(func: Callable[..., Any], tool_name: str) -> type[BaseModel]:
    """Build a Pydantic model from a function's signature + type hints.

    Handles three default-value styles:

    ==================== =============================================
    Signature            Pydantic field
    ==================== =============================================
    ``x: int``           ``(int, ...)``                — required
    ``x: int = 5``       ``(int, 5)``                  — optional
    ``x: int = Field()`` ``(int, <FieldInfo>)``        — rich metadata
    ==================== =============================================
    """
    sig = inspect.signature(func)
    hints = get_type_hints(func)
    field_definitions: dict[str, Any] = {}

    for param_name, param in sig.parameters.items():
        if param_name in ("self", "cls", "return"):
            continue

        annotation = hints.get(param_name, Any)
        default = param.default

        if default is inspect.Parameter.empty:
            # Required — no default value
            field_definitions[param_name] = (annotation, ...)
        elif isinstance(default, FieldInfo):
            # pydantic.Field(...) used as the default
            field_definitions[param_name] = (annotation, default)
        else:
            # Plain Python default (int, str, None, …)
            field_definitions[param_name] = (annotation, default)

    model_name = f"{_pascal_case(tool_name)}Args"
    return create_model(model_name, **field_definitions)  # type: ignore[call-overload]


def _pascal_case(snake: str) -> str:
    """Convert ``snake_case`` → ``PascalCase``."""
    return "".join(word.capitalize() for word in snake.replace("-", "_").split("_"))


# ═══════════════════════════════════════════════════════════════════════════
# @aug_tool decorator
# ═══════════════════════════════════════════════════════════════════════════

def aug_tool(
    func: Callable[..., Any] | None = None,
    *,
    name: str | None = None,
    description: str | None = None,
    args_schema: type[BaseModel] | None = None,
) -> AugTool | Callable[[Callable[..., Any]], AugTool]:
    """Decorator that converts a function into an :class:`AugTool`.

    Works bare or with keyword arguments::

        @aug_tool
        def my_tool(x: int) -> str: ...

        @aug_tool(description="Custom description")
        def my_tool(x: int) -> str: ...

        @aug_tool(args_schema=MyArgsModel)
        def my_tool(x: int) -> str: ...
    """

    def _wrap(fn: Callable[..., Any]) -> AugTool:
        return AugTool.from_function(
            fn,
            name=name,
            description=description,
            args_schema=args_schema,
        )

    if func is not None:
        # Bare ``@aug_tool`` (no parentheses)
        return _wrap(func)

    # Called with arguments: ``@aug_tool(...)``
    return _wrap


# ---------------------------------------------------------------------------
# Backward-compatible aliases
# ---------------------------------------------------------------------------
Tool = AugTool
"""Alias for :class:`AugTool` — kept for backward compatibility."""

tool = aug_tool
"""Alias for :func:`aug_tool` — kept for backward compatibility."""

_delegation_registry = {}

def register_agent(agent: Any):
    """Register an agent for delegation."""
    _delegation_registry[agent.name] = agent

class DelegateWorkArgs(BaseModel):
    agent_name: str = Field(description="The role/name of the sub-agent to delegate to.")
    task_description: str = Field(description="Detailed instructions of what the sub-agent needs to accomplish.")

@aug_tool(args_schema=DelegateWorkArgs)
async def DelegateWorkTool(agent_name: str, task_description: str) -> str:
    """Delegate a subtask to another specialized agent. The sub-agent will return its string result."""
    from augagent.task import Task
    
    sub_agent = _delegation_registry.get(agent_name)
    if not sub_agent:
        return f"Error: Agent '{agent_name}' not found. Available agents: {list(_delegation_registry.keys())}"
        
    sub_task = Task(description=task_description, agent=sub_agent)
    result = await sub_task.execute()
    return f"Delegation to {agent_name} complete. Result:\n{result.output}"


# ═══════════════════════════════════════════════════════════════════════════
# Plugin Architecture
# ═══════════════════════════════════════════════════════════════════════════

class PluginRegistry:
    """Dynamically loads AugTools from external packages."""
    
    @classmethod
    def load_plugins(cls) -> list[AugTool]:
        """Scan installed packages starting with 'augagent_plugin_' and load their tools."""
        import importlib
        import pkgutil
        
        tools = []
        # Discover all modules in the environment
        for module_info in pkgutil.iter_modules():
            if module_info.name.startswith("augagent_plugin_"):
                try:
                    module = importlib.import_module(module_info.name)
                    # Look for AugTool instances in the module
                    for attr_name in dir(module):
                        attr = getattr(module, attr_name)
                        if isinstance(attr, AugTool):
                            tools.append(attr)
                except Exception as e:
                    # Silently skip broken plugins or log them if a logger was available here
                    pass
        return tools

# EOF
