# ============================================================
# tools.py — The "superpowers" our agent can use
# Each function = one tool the LLM can decide to call
# ============================================================

from langchain.tools import tool          # Decorator that turns a function into a LangChain tool
from duckduckgo_search import DDGS        # Free web search (no API key needed!)
import requests                           # HTTP requests to fetch web pages
from bs4 import BeautifulSoup            # HTML parser to extract text from web pages
from datetime import datetime            # For timestamps in saved files
import json                              # For structured data handling


@tool  # This decorator tells LangChain: "this function is a tool the agent can use"
def search_web(query: str) -> str:
    """
    Search the internet for current information about any topic.
    Use this when you need up-to-date facts, news, or research.
    Returns: Top 5 search results with titles, links, and snippets.
    """
    try:
        # DuckDuckGo is FREE and doesn't need an API key
        with DDGS() as ddgs:
            # Search and get top 5 results
            results = list(ddgs.text(query, max_results=5))
        
        if not results:
            return "No results found for this query."
        
        # Format results as clean text the LLM can understand
        formatted = []
        for i, r in enumerate(results, 1):
            formatted.append(
                f"Result {i}:\n"
                f"  Title: {r.get('title', 'No title')}\n"
                f"  URL: {r.get('href', 'No URL')}\n"
                f"  Summary: {r.get('body', 'No summary')}\n"
            )
        
        return "\n".join(formatted)
    
    except Exception as e:
        return f"Search failed: {str(e)}. Try a different query."


@tool
def fetch_webpage(url: str) -> str:
    """
    Fetch and read the full text content of a webpage.
    Use this after search_web to get complete article content.
    Returns: The main text content of the page (cleaned HTML).
    """
    try:
        # Set headers to look like a real browser (avoids being blocked)
        headers = {
            "User-Agent": "Mozilla/5.0 (compatible; ResearchBot/1.0)"
        }
        
        # Download the webpage
        response = requests.get(url, headers=headers, timeout=10)
        response.raise_for_status()  # Raises error if page not found
        
        # Parse HTML and extract just the text
        soup = BeautifulSoup(response.content, 'html.parser')
        
        # Remove navigation, ads, scripts (we only want article content)
        for tag in soup(['nav', 'header', 'footer', 'script', 'style', 'ads']):
            tag.decompose()  # Remove these tags from the DOM
        
        # Get the main text content
        text = soup.get_text(separator='\n', strip=True)
        
        # Limit to 3000 characters (LLMs have context limits — like a BigQuery scan limit)
        return text[:3000] + "...[truncated]" if len(text) > 3000 else text
    
    except requests.exceptions.Timeout:
        return "Webpage timed out. Try a different URL."
    except Exception as e:
        return f"Could not fetch webpage: {str(e)}"


@tool
def save_report(content: str) -> str:
    """
    Save research findings to a markdown file on disk.
    Use this as the FINAL step after completing research.
    The content should be a well-formatted research report.
    Returns: The filepath where the report was saved.
    """
    # Create filename with timestamp (like partitioned BigQuery tables!)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"research_report_{timestamp}.md"
    
    # Add a header to the report
    full_content = f"""# Research Report
Generated: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}
---

{content}
"""
    
    # Write to file
    with open(filename, 'w', encoding='utf-8') as f:
        f.write(full_content)
    
    return f"Report saved successfully to: {filename}"


@tool
def calculate(expression: str) -> str:
    """
    Safely evaluate a mathematical expression.
    Use this for any calculations needed during research.
    Example: '(150 * 0.23) + 45' → '79.5'
    """
    try:
        # eval() is dangerous with untrusted input, but we restrict it here
        # In production, use a proper math parser
        allowed_chars = set('0123456789+-*/().% ')
        if not all(c in allowed_chars for c in expression):
            return "Error: Only basic math operators allowed"
        
        result = eval(expression)  # nosec - restricted to math only
        return str(round(result, 4))
    except Exception as e:
        return f"Calculation error: {str(e)}"