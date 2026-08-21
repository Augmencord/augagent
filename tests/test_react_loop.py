import pytest
import httpx
import json
import asyncio
from augagent import AugAgent, LLMConfig
from augagent.models import TokenBudget, TokenBudgetExceededError
from augagent.tools import aug_tool

@aug_tool
def get_weather(location: str) -> str:
    """Get the weather for a location."""
    return f"The weather in {location} is sunny."

def mock_response(content="", tool_calls=None, status_code=200, is_stream=False):
    if is_stream:
        # Simulate streaming response
        async def stream_content():
            chunks = [content[i:i+5] for i in range(0, len(content), 5)]
            for chunk in chunks:
                data = json.dumps({"choices": [{"delta": {"content": chunk}}]})
                yield f"data: {data}\n\n".encode("utf-8")
                await asyncio.sleep(0.01)
            yield b"data: [DONE]\n\n"
        return httpx.Response(status_code, content=stream_content())
    else:
        msg = {"role": "assistant"}
        if content:
            msg["content"] = content
        if tool_calls:
            msg["tool_calls"] = tool_calls
        
        data = {
            "choices": [{"message": msg}],
            "usage": {"prompt_tokens": 10, "completion_tokens": 20, "total_tokens": 30}
        }
        return httpx.Response(status_code, json=data)

@pytest.mark.asyncio
async def test_react_loop_basic_completion():
    def handler(request: httpx.Request):
        return mock_response(content="Hello, I am an AI.")

    transport = httpx.MockTransport(handler)
    client = httpx.AsyncClient(transport=transport)
    
    config = LLMConfig(model="test-model", api_key="test")
    agent = AugAgent(name="Test", role="Tester", goal="Test", llm_config=config)
    agent._client = client
    
    result = await agent.execute("Say hello.")
    
    assert result.output == "Hello, I am an AI."
    assert result.iterations == 1
    assert result.token_usage["total_tokens"] == 30

@pytest.mark.asyncio
async def test_react_loop_tool_execution():
    call_count = 0
    def handler(request: httpx.Request):
        nonlocal call_count
        call_count += 1
        if call_count == 1:
            # Return a tool call
            return mock_response(tool_calls=[{
                "id": "call_123",
                "type": "function",
                "function": {
                    "name": "get_weather",
                    "arguments": '{"location": "Seattle"}'
                }
            }])
        else:
            # Return final answer
            body = json.loads(request.content)
            messages = body["messages"]
            assert messages[-1]["role"] == "tool"
            assert messages[-1]["content"] == "The weather in Seattle is sunny."
            return mock_response(content="The weather in Seattle is sunny.")

    transport = httpx.MockTransport(handler)
    client = httpx.AsyncClient(transport=transport)
    
    config = LLMConfig(model="test-model", api_key="test")
    agent = AugAgent(name="Test", role="Tester", goal="Test", llm_config=config)
    agent.tools.append(get_weather)
    agent._client = client
    
    result = await agent.execute("What's the weather in Seattle?")
    
    assert result.output == "The weather in Seattle is sunny."
    assert result.iterations == 2

@pytest.mark.asyncio
async def test_react_loop_token_budget_exceeded():
    def handler(request: httpx.Request):
        return mock_response(content="Here is some output.")

    transport = httpx.MockTransport(handler)
    client = httpx.AsyncClient(transport=transport)
    
    config = LLMConfig(model="test-model", api_key="test")
    agent = AugAgent(
        name="Test", 
        role="Tester", 
        goal="Test", 
        llm_config=config,
        token_budget=TokenBudget(max_total_tokens=15) # Very small budget, response is 30
    )
    agent._client = client
    
    result = await agent.execute("This should fail due to tokens.")
    assert result.status.name == "FAILED"
    assert "budget exceeded" in result.output.lower() or "budget exceeded" in result.raw_output.lower() if isinstance(result.raw_output, str) else True

@pytest.mark.asyncio
async def test_react_loop_hitl_approval_approved():
    call_count = 0
    def handler(request: httpx.Request):
        nonlocal call_count
        call_count += 1
        if call_count == 1:
            return mock_response(tool_calls=[{
                "id": "call_123",
                "type": "function",
                "function": {
                    "name": "get_weather",
                    "arguments": '{"location": "Tokyo"}'
                }
            }])
        else:
            return mock_response(content="Weather in Tokyo is sunny.")

    transport = httpx.MockTransport(handler)
    client = httpx.AsyncClient(transport=transport)
    
    config = LLMConfig(model="test-model", api_key="test")
    agent = AugAgent(
        name="Test", 
        role="Tester", 
        goal="Test", 
        llm_config=config, 
        require_human_approval=True
    )
    agent.tools.append(get_weather)
    agent._client = client
    
    # Run agent in background
    execute_task = asyncio.create_task(agent.execute("Weather in Tokyo?"))
    
    # Wait for it to pause on HITL
    await asyncio.sleep(0.1)
    
    # Approve
    agent.approve_pending_action()
    
    result = await execute_task
    assert result.output == "Weather in Tokyo is sunny."

@pytest.mark.asyncio
async def test_react_loop_streaming():
    def handler(request: httpx.Request):
        return mock_response(content="Streaming is cool.", is_stream=True)

    transport = httpx.MockTransport(handler)
    client = httpx.AsyncClient(transport=transport)
    
    config = LLMConfig(model="test-model", api_key="test")
    agent = AugAgent(name="Test", role="Tester", goal="Test", llm_config=config)
    agent._client = client
    
    chunks = []
    async def callback(chunk: str):
        chunks.append(chunk)
        
    result = await agent.execute("Tell me about streaming.", stream_callback=callback)
    
    assert "".join(chunks) == "Streaming is cool."
    assert result.output == "Streaming is cool."

@pytest.mark.asyncio
async def test_react_loop_rate_limit_retry():
    call_count = 0
    def handler(request: httpx.Request):
        nonlocal call_count
        call_count += 1
        if call_count == 1:
            return httpx.Response(429, headers={"retry-after": "0.1"})
        return mock_response(content="Recovered from 429.")

    transport = httpx.MockTransport(handler)
    client = httpx.AsyncClient(transport=transport)
    
    config = LLMConfig(model="test-model", api_key="test")
    agent = AugAgent(name="Test", role="Tester", goal="Test", llm_config=config)
    agent._client = client
    
    result = await agent.execute("Will you fail?")
    
    assert result.output == "Recovered from 429."
    assert call_count == 2
