"""Smoke test for the augagent framework."""
import sys, json
sys.path.insert(0, "src")

from augagent import (
    AugAgent, AugTool, aug_tool, Agent, Tool, tool,  # aliases
    Task, Team, LLMConfig, ChatCompletion, AgentConfig,
    Process, TokenUsage, ChatMessage, ChatToolCall, FunctionCall,
)
from pydantic import Field, BaseModel

print("=" * 60)
print("1. All imports OK")
print("=" * 60)

# ── @aug_tool decorator ──────────────────────────────────────────────────

@aug_tool
def search_web(
    query: str = Field(description="The search query"),
    max_results: int = Field(default=5, description="Max results to return"),
) -> str:
    """Search the web and return relevant results."""
    return f"Results for: {query} (max={max_results})"

print(f"\n2. @aug_tool decorator: {search_web}")
print(f"   Name: {search_web.name}")
print(f"   Description: {search_web.description}")
schema = search_web.to_openai_schema()
print(f"   OpenAI Schema:\n{json.dumps(schema, indent=4)}")

# ── Validate tool execution ─────────────────────────────────────────────

result = search_web.run_sync(query="AI agents", max_results=3)
print(f"\n3. Tool sync execution: {result}")

# ── AugTool.from_function ───────────────────────────────────────────────

def calculate(expression: str) -> str:
    """Evaluate a math expression."""
    return str(eval(expression))

calc_tool = AugTool.from_function(calculate, name="calculator")
print(f"\n4. AugTool.from_function: {calc_tool}")
print(f"   Schema: {json.dumps(calc_tool.to_openai_schema(), indent=4)}")

# ── Custom Pydantic args_schema ─────────────────────────────────────────

class WeatherArgs(BaseModel):
    city: str = Field(description="City name")
    units: str = Field(default="celsius", description="Temperature units")

@aug_tool(args_schema=WeatherArgs)
def get_weather(city: str, units: str = "celsius") -> str:
    """Get the current weather for a city."""
    return f"Sunny, 22° {units} in {city}"

print(f"\n5. Custom args_schema: {get_weather}")
print(f"   Schema: {json.dumps(get_weather.to_openai_schema(), indent=4)}")

# ── LLMConfig ───────────────────────────────────────────────────────────

cfg = LLMConfig(model="gpt-4o-mini", temperature=0.3)
print(f"\n6. LLMConfig: model={cfg.model}, temp={cfg.temperature}, base_url={cfg.base_url}")

# ── AugAgent construction ───────────────────────────────────────────────

agent = AugAgent(
    name="Researcher",
    role="Senior Research Analyst",
    goal="Find accurate, comprehensive information",
    backstory="15 years of experience in investigative research.",
    llm_config=cfg,
    tools=[search_web, get_weather],
    max_iterations=10,
    verbose=True,
)
print(f"\n7. AugAgent: {agent}")

# ── AgentConfig roundtrip ───────────────────────────────────────────────

config = agent.to_config()
print(f"\n8. AgentConfig roundtrip:")
print(f"   {config.model_dump_json(indent=4)}")

restored = AugAgent.from_config(config, tools=[search_web, get_weather])
print(f"   Restored: {restored}")

# ── Backward-compat aliases ─────────────────────────────────────────────

assert Agent is AugAgent, "Agent alias broken"
assert Tool is AugTool, "Tool alias broken"
assert tool is aug_tool, "tool alias broken"
print(f"\n9. Backward-compat aliases: Agent, Tool, tool OK")

# ── ChatCompletion parsing ──────────────────────────────────────────────

raw_response = {
    "id": "chatcmpl-test123",
    "object": "chat.completion",
    "created": 1234567890,
    "model": "gpt-4o-mini",
    "choices": [{
        "index": 0,
        "message": {
            "role": "assistant",
            "content": None,
            "tool_calls": [{
                "id": "call_abc",
                "type": "function",
                "function": {
                    "name": "search_web",
                    "arguments": '{"query": "AI agents", "max_results": 3}'
                }
            }]
        },
        "finish_reason": "tool_calls"
    }],
    "usage": {
        "prompt_tokens": 50,
        "completion_tokens": 25,
        "total_tokens": 75
    }
}

completion = ChatCompletion.model_validate(raw_response)
print(f"\n10. ChatCompletion parsing:")
print(f"    Model: {completion.model}")
print(f"    Tool call: {completion.choices[0].message.tool_calls[0].function.name}")
print(f"    Arguments: {completion.choices[0].message.tool_calls[0].function.arguments}")
print(f"    Usage: {completion.usage}")

# ── Task + Team construction ────────────────────────────────────────────

t1 = Task(description="Research AI trends", expected_output="Bullet points", agent=agent)
t2 = Task(description="Summarise findings", expected_output="One paragraph", agent=agent, context=[t1])
team = Team(agents=[agent], tasks=[t1, t2], process=Process.SEQUENTIAL, verbose=True)
print(f"\n11. Task + Team: {team}")

print("\n" + "=" * 60)
print("ALL TESTS PASSED")
print("=" * 60)
