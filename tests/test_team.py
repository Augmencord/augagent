import pytest
from augagent.team import AugTeam, Process

from augagent import AugAgent, LLMConfig

@pytest.mark.asyncio
async def test_team_graph_inputs():
    agent = AugAgent(name="A", role="A", goal="A", llm_config=LLMConfig(model="test"))
    team = AugTeam(name="TestTeam", process=Process.GRAPH, agents=[agent])
    
    # We are testing that passing inputs to _run does not cause a NameError
    try:
        # Without tasks or a valid graph, this might raise another error, 
        # but we specifically want to ensure 'inputs' is defined and passed correctly.
        try:
            results = await team._run(inputs={"test": "value"})
        except Exception as e:
            # We allow other exceptions related to missing graph config
            if isinstance(e, NameError) and "inputs" in str(e):
                pytest.fail(f"NameError on inputs: {e}")
    except Exception:
        pass
