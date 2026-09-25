import asyncio
import json
import httpx
import pytest
from fastapi.testclient import TestClient

from augagent import AugAgent, LLMConfig
from augagent.api import app, active_agents
from augagent.models import ChatMessage, TaskStatus
from augagent.telemetry import get_logger
from augagent.tools import aug_tool


@aug_tool
def fetch_weather(location: str) -> str:
    """Get the current weather."""
    return f"Weather in {location} is 22C and sunny."


def make_sse_response(text: str, chunk_size: int = 4, status_code: int = 200) -> httpx.Response:
    """Helper to simulate an OpenAI-compatible SSE streaming HTTP response."""
    async def sse_stream():
        chunks = [text[i:i + chunk_size] for i in range(0, len(text), chunk_size)]
        for chunk in chunks:
            payload = {
                "choices": [{
                    "index": 0,
                    "delta": {"content": chunk},
                    "finish_reason": None,
                }]
            }
            yield f"data: {json.dumps(payload)}\n\n".encode("utf-8")
            await asyncio.sleep(0.005)
        yield b"data: [DONE]\n\n"

    return httpx.Response(status_code, content=sse_stream(), headers={"content-type": "text/event-stream"})


def make_json_response(text: str, status_code: int = 200) -> httpx.Response:
    """Helper to return standard non-streaming JSON response."""
    data = {
        "id": "chatcmpl-test",
        "object": "chat.completion",
        "created": 123456789,
        "model": "test-model",
        "choices": [{
            "index": 0,
            "message": {"role": "assistant", "content": text},
            "finish_reason": "stop",
        }],
        "usage": {
            "prompt_tokens": 10,
            "completion_tokens": 15,
            "total_tokens": 25,
        },
    }
    return httpx.Response(status_code, json=data)


@pytest.mark.asyncio
async def test_call_llm_streaming_sse_chunks():
    """Verify _call_llm(stream=True) parses SSE chunks and yields them via stream_callback."""
    expected_text = "Streaming responses in real-time."
    received_tokens: list[str] = []

    def handler(request: httpx.Request):
        assert request.url.path.endswith("/chat/completions")
        body = json.loads(request.read().decode("utf-8"))
        assert body.get("stream") is True
        return make_sse_response(expected_text, chunk_size=5)

    transport = httpx.MockTransport(handler)
    client = httpx.AsyncClient(transport=transport)

    config = LLMConfig(model="mock-gpt", api_key="test-key")
    agent = AugAgent(name="TestAgent", role="Tester", goal="Stream tokens", llm_config=config)
    agent._client = client

    async def callback(token: str):
        received_tokens.append(token)

    logger = get_logger()
    messages = [{"role": "user", "content": "Start streaming"}]
    completion = await agent._call_llm(messages, logger, stream_callback=callback, stream=True)

    assert "".join(received_tokens) == expected_text
    assert completion.choices[0].message.content == expected_text
    assert completion.usage is not None
    assert completion.usage.total_tokens > 0


@pytest.mark.asyncio
async def test_streaming_matches_non_streaming():
    """Verify final output of streaming execution matches non-streaming execution."""
    sample_text = "Augagent powers intelligent IDE capabilities with full fidelity."

    # 1. Non-streaming execution
    def handler_non_stream(request: httpx.Request):
        body = json.loads(request.read().decode("utf-8"))
        assert not body.get("stream")
        return make_json_response(sample_text)

    client_non_stream = httpx.AsyncClient(transport=httpx.MockTransport(handler_non_stream))
    agent_non_stream = AugAgent(
        name="NonStreamAgent",
        role="Tester",
        goal="Test parity",
        llm_config=LLMConfig(model="mock-gpt", api_key="k"),
    )
    agent_non_stream._client = client_non_stream

    result_non_stream = await agent_non_stream.execute("Explain Augagent.", stream=False)

    # 2. Streaming execution
    streamed_tokens: list[str] = []

    def handler_stream(request: httpx.Request):
        body = json.loads(request.read().decode("utf-8"))
        assert body.get("stream") is True
        return make_sse_response(sample_text, chunk_size=6)

    client_stream = httpx.AsyncClient(transport=httpx.MockTransport(handler_stream))
    agent_stream = AugAgent(
        name="StreamAgent",
        role="Tester",
        goal="Test parity",
        llm_config=LLMConfig(model="mock-gpt", api_key="k"),
    )
    agent_stream._client = client_stream

    async def token_collector(token: str):
        streamed_tokens.append(token)

    result_stream = await agent_stream.execute(
        "Explain Augagent.",
        stream_callback=token_collector,
        stream=True,
    )

    # Assert exact match of output
    assert result_stream.output == result_non_stream.output
    assert "".join(streamed_tokens) == result_non_stream.output
    assert result_stream.status == TaskStatus.COMPLETED
    assert result_non_stream.status == TaskStatus.COMPLETED


@pytest.mark.asyncio
async def test_streaming_unicode_and_emojis():
    """Verify streaming handles multibyte Unicode characters and emojis properly."""
    unicode_text = "🚀 AugHome IDE — 智能化 💻 Code Editor with full UTF-8: 🌟✓"
    received: list[str] = []

    def handler(request: httpx.Request):
        return make_sse_response(unicode_text, chunk_size=3)

    client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    agent = AugAgent(
        name="UnicodeAgent",
        role="Tester",
        goal="UTF8",
        llm_config=LLMConfig(model="mock-gpt", api_key="k"),
    )
    agent._client = client

    def sync_callback(chunk: str):
        received.append(chunk)

    result = await agent.execute("Give unicode", stream_callback=sync_callback, stream=True)
    assert "".join(received) == unicode_text
    assert result.output == unicode_text


@pytest.mark.asyncio
async def test_streaming_with_tool_call_sse():
    """Verify tool call flow works with streaming SSE chunks."""
    call_count = 0

    def handler(request: httpx.Request):
        nonlocal call_count
        call_count += 1
        if call_count == 1:
            # Tool call response
            payload = {
                "choices": [{
                    "index": 0,
                    "delta": {
                        "content": None,
                        "tool_calls": [{
                            "index": 0,
                            "id": "call_weather_1",
                            "type": "function",
                            "function": {
                                "name": "fetch_weather",
                                "arguments": '{"location": "Tokyo"}',
                            },
                        }],
                    },
                    "finish_reason": "tool_calls",
                }]
            }
            async def sse():
                yield f"data: {json.dumps(payload)}\n\n".encode("utf-8")
                yield b"data: [DONE]\n\n"
            return httpx.Response(200, content=sse())
        else:
            return make_sse_response("Tokyo is pleasant at 22C.")

    client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    agent = AugAgent(
        name="ToolAgent",
        role="Tester",
        goal="Tools",
        tools=[fetch_weather],
        llm_config=LLMConfig(model="mock-gpt", api_key="k"),
    )
    agent._client = client

    events: list[dict | str] = []

    async def stream_cb(event: dict | str):
        events.append(event)

    result = await agent.execute("Weather in Tokyo?", stream_callback=stream_cb, stream=True)
    assert result.output == "Tokyo is pleasant at 22C."
    # Verify tool events occurred
    tool_events = [e for e in events if isinstance(e, dict) and e.get("type") in ("tool_call_start", "tool_call_end")]
    assert len(tool_events) == 2


@pytest.mark.asyncio
async def test_api_v1_agent_stream_sse_endpoint():
    """Verify POST /v1/agent/stream returns SSE response with token and done events."""
    mock_answer = "Hello from streamed SSE API endpoint!"

    def handler(request: httpx.Request):
        return make_sse_response(mock_answer, chunk_size=4)

    mock_client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    test_agent = AugAgent(
        id="test-stream-agent",
        name="ApiStreamAgent",
        role="Assistant",
        goal="API Testing",
        llm_config=LLMConfig(model="mock-gpt", api_key="k"),
    )
    test_agent._client = mock_client
    active_agents["test-stream-agent"] = test_agent

    # Use httpx.AsyncClient with ASGITransport to test streaming endpoint directly
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app),
        base_url="http://testserver",
    ) as ac:
        resp = await ac.post("/v1/agent/stream", json={
            "prompt": "Say hello",
            "agent_id": "test-stream-agent",
        })
        assert resp.status_code == 200
        assert "text/event-stream" in resp.headers.get("content-type", "")

        # Parse SSE stream
        events: list[tuple[str, dict]] = []
        current_event: str | None = None
        for line in resp.text.splitlines():
            line = line.strip()
            if not line:
                continue
            if line.startswith("event: "):
                current_event = line[7:]
            elif line.startswith("data: ") and current_event:
                data = json.loads(line[6:])
                events.append((current_event, data))
                current_event = None

        event_names = [e[0] for e in events]
        assert "token" in event_names
        assert "done" in event_names

        # Concatenate tokens
        reconstructed = "".join([e[1]["token"] for e in events if e[0] == "token"])
        assert reconstructed == mock_answer

        # Check done event
        done_event = [e[1] for e in events if e[0] == "done"][0]
        assert done_event["output"] == mock_answer
        assert done_event["status"].lower() == "completed"


def test_api_v1_agent_stream_errors():
    """Verify failure modes on /v1/agent/stream."""
    test_client = TestClient(app)

    # 1. Nonexistent agent id
    res_404 = test_client.post("/v1/agent/stream", json={
        "prompt": "Hello",
        "agent_id": "nonexistent-agent-id-12345",
    })
    assert res_404.status_code == 404
    assert "not found" in res_404.json()["detail"].lower()

    # 2. Invalid JWT token
    res_401 = test_client.post(
        "/v1/agent/stream",
        json={"prompt": "Hello"},
        headers={"Authorization": "Bearer invalid-tampered-token"},
    )
    assert res_401.status_code == 401
    assert "invalid token" in res_401.json()["detail"].lower()
