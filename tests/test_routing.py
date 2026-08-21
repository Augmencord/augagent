import pytest
import asyncio
from unittest.mock import AsyncMock, patch, MagicMock
from augagent.agent import AugAgent
from augagent.models import LLMConfig

@pytest.mark.asyncio
async def test_llm_fallback_routing():
    config1 = LLMConfig(model="model-1")
    config2 = LLMConfig(model="model-2")
    
    agent = AugAgent(
        name="TestAgent",
        role="Tester",
        goal="Test fallback",
        llm_config=config1,
        fallback_models=[config2]
    )
    
    # Mock _call_llm_single to fail on first call, succeed on second
    mock_call = AsyncMock(side_effect=[Exception("503 Service Unavailable"), {"content": "success", "tool_calls": []}])
    
    with patch.object(agent, "_call_llm_single", mock_call):
        res = await agent._call_llm([{"role": "user", "content": "hi"}], MagicMock())
        assert res["content"] == "success"
        assert mock_call.call_count == 2
        
        # Check that it tried config1 then config2
        call_args = mock_call.call_args_list
        assert call_args[0][0][0] == config1
        assert call_args[1][0][0] == config2
