import pytest
from augagent import AugAgent, LLMConfig

@pytest.mark.asyncio
async def test_agent_initialization():
    config = LLMConfig(model="test-model", api_key="test")
    agent = AugAgent(
        name="TestAgent",
        role="Tester",
        goal="Test things",
        llm_config=config
    )
    assert agent.name == "TestAgent"
    assert agent.role == "Tester"
