"""Web Search and Scraping tools for AugAgent."""

import requests
from bs4 import BeautifulSoup
import markdownify
from pydantic import BaseModel, Field
from augagent.tools import aug_tool

class SearchWebArgs(BaseModel):
    query: str = Field(description="The search query.")
    num_results: int = Field(default=3, description="Number of results to return.")

@aug_tool(args_schema=SearchWebArgs)
def search_web(query: str, num_results: int = 3) -> str:
    """Search the web using a local SearxNG instance."""
    url = "http://localhost:8080/search"
    params = {
        "q": query,
        "format": "json"
    }
    
    try:
        response = requests.get(url, params=params, timeout=10)
        response.raise_for_status()
        data = response.json()
        
        results = data.get("results", [])[:num_results]
        if not results:
            return "No results found."
            
        output = [f"Search Results for '{query}':"]
        for i, res in enumerate(results):
            output.append(f"{i+1}. {res.get('title')}")
            output.append(f"   URL: {res.get('url')}")
            output.append(f"   Snippet: {res.get('content')}")
            output.append("")
            
        return "\n".join(output)
        
    except requests.exceptions.ConnectionError:
        return "Error: Could not connect to SearxNG at http://localhost:8080. Is the Docker container running?"
    except Exception as e:
        return f"Error performing search: {e}"

class ReadUrlArgs(BaseModel):
    url: str = Field(description="The URL of the webpage to read.")

@aug_tool(args_schema=ReadUrlArgs)
def read_url_content(url: str) -> str:
    """Fetch a webpage and extract its main content as clean Markdown."""
    try:
        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
        }
        response = requests.get(url, headers=headers, timeout=15)
        response.raise_for_status()
        
        soup = BeautifulSoup(response.text, 'html.parser')
        
        # Remove noisy elements
        for element in soup(["script", "style", "nav", "footer", "header", "aside"]):
            element.decompose()
            
        # Try to find the main content article if it exists
        main_content = soup.find('article') or soup.find('main') or soup.body
        if not main_content:
            return "Error: Could not extract body content from the page."
            
        # Convert to Markdown
        md_content = markdownify.markdownify(str(main_content), heading_style="ATX").strip()
        
        # Truncate if massive (context limit protection)
        max_chars = 30000
        if len(md_content) > max_chars:
            md_content = md_content[:max_chars] + f"\n\n[Content truncated at {max_chars} characters]"
            
        return md_content
        
    except Exception as e:
        return f"Error reading URL: {e}"
