import os
import sys

# Ensure src/ is in the python path if running locally
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))

import asyncio
from augagent import AugAgent, AugTask, AugTeam, aug_tool, LLMConfig
from pydantic import Field

# 1. Define tools

@aug_tool
def search_web(query: str = Field(description="Search query")) -> str:
    """Search the web for information."""
    print(f"  [Tool Executing] Searching the web for: {query}")
    # Mock search results for demonstration
    if "quantum" in query.lower():
        return "Quantum computing uses qubits. It can solve complex problems faster than classical computers."
    return f"Found generic results for {query}."

@aug_tool
def fetch_article(url: str = Field(description="URL to fetch")) -> str:
    """Fetch the contents of an article."""
    print(f"  [Tool Executing] Fetching article: {url}")
    return "This is the content of the article regarding " + url

# 2. Define agents

llm_config = LLMConfig(model="gpt-4o-mini", temperature=0.2)

researcher = AugAgent(
    name="Researcher",
    role="Senior Technology Researcher",
    goal="Find comprehensive and accurate information on emerging technologies.",
    backstory="You have 10 years of experience researching deep tech.",
    llm_config=llm_config,
    tools=[search_web, fetch_article],
    verbose=True
)

writer = AugAgent(
    name="Writer",
    role="Technical Content Writer",
    goal="Write engaging, accurate articles about technology.",
    backstory="You write clear, accessible articles for a wide audience.",
    llm_config=llm_config,
    verbose=True
)

# 3. Define tasks

research_task = AugTask(
    description="Research the topic: {topic}. Find key facts and recent advancements.",
    expected_output="A bulleted list of key facts about {topic}.",
    agent=researcher
)

writing_task = AugTask(
    description="Write a short, engaging article about {topic} based on the research provided.",
    expected_output="A 2-paragraph article in Markdown format.",
    agent=writer
)

# 4. Assemble the team and kickoff

team = AugTeam(
    agents=[researcher, writer],
    tasks=[research_task, writing_task],
    verbose=True
)

if __name__ == "__main__":
    print(f"=== Kicking off AugTeam ===")
    results = team.kickoff(inputs={"topic": "Quantum Computing"})
    
    print("\n=== Final Results ===")
    for idx, result in enumerate(results):
        print(f"\nTask {idx + 1} ({result.agent_name}):")
        print("-" * 40)
        print(result.output)
