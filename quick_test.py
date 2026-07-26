from augagent import AugAgent, AugTask, AugTeam

# 1. The Heavy Lifter (Assigned to Qwen 8B)
researcher = AugAgent(
    name="Lead Researcher",
    role="Senior Analyst",
    goal="Solve complex problems and gather detailed data.",
    backstory="You are a meticulous researcher who thinks deeply.",
    tools=[], 
    llm_config={
        "base_url": "http://localhost:11434/v1", 
        "model": "qwen3:8b",  
        "api_key": "ollama"     
    }
)

# 2. The Fast Formatter (Assigned to Qwen 7B)
writer = AugAgent(
    name="Speedy Writer",
    role="Content Creator",
    goal="Take research and write a clean, readable summary.",
    backstory="You are a fast writer who excels at formatting text.",
    tools=[],
    llm_config={
        "base_url": "http://localhost:11434/v1", 
        "model": "qwen2:7b",  
        "api_key": "ollama"    
    }
)

# 3. Create Tasks
task_1 = AugTask(
    description="Analyze the history of quantum computing. List 3 key breakthroughs.",
    expected_output="A bulleted list of 3 major milestones.",
    agent=researcher
)

task_2 = AugTask(
    description="Take the researcher's list and write a 1-paragraph summary.",
    expected_output="A simple, readable paragraph.",
    agent=writer
)

# 4. Run the Multi-Model Team
team = AugTeam(agents=[researcher, writer], tasks=[task_1, task_2])
result = team.kickoff()

print("\n--- FINAL RESULT ---")
print(result)