# tools.py
# ─────────────────────────────────────────────────────
# 6 tools that give the agent its superpowers.
# The docstring of each @tool function is READ BY THE LLM
# to decide when and how to call it. Write them carefully!
# ─────────────────────────────────────────────────────

from langchain.tools import tool
from langchain_ollama import ChatOllama
from langchain_chroma import Chroma
from langchain_ollama import OllamaEmbeddings
from duckduckgo_search import DDGS
import requests
from bs4 import BeautifulSoup
from datetime import datetime
from config import LLM_MODEL, NOTES_DB_PATH, EMBEDDING_MODEL


# ── Tool 1: Web Search ─────────────────────────────────────────
@tool
def search_web(query: str) -> str:
    """
    Search the internet for current, up-to-date information on any topic.
    Use this for: recent news, facts you don't know, current events,
    technology comparisons, documentation lookups.
    Returns top 5 results with titles, URLs, and summaries.
    DO NOT use this for calculations or saved notes.
    """
    try:
        with DDGS() as ddgs:
            results = list(ddgs.text(query, max_results=5))
        
        if not results:
            return f"No results found for '{query}'. Try different keywords."
        
        output = []
        for i, r in enumerate(results, 1):
            output.append(
                f"[{i}] {r.get('title', 'No title')}\n"
                f"    URL: {r.get('href', '')}\n"
                f"    {r.get('body', 'No description')[:200]}"
            )
        return "\n\n".join(output)
    
    except Exception as e:
        return f"Search error: {e}. Try rephrasing the query."


# ── Tool 2: Fetch Webpage ──────────────────────────────────────
@tool
def fetch_webpage(url: str) -> str:
    """
    Fetch and read the full text content of a specific webpage or article.
    Use this AFTER search_web when you need the complete content of a result.
    Best for: reading full articles, documentation pages, blog posts.
    Input must be a valid URL starting with http:// or https://
    Returns: cleaned text content (first 3000 characters).
    """
    try:
        headers = {"User-Agent": "Mozilla/5.0 (compatible; AssistantBot/1.0)"}
        resp = requests.get(url, headers=headers, timeout=10)
        resp.raise_for_status()
        
        soup = BeautifulSoup(resp.content, 'html.parser')
        # Remove non-content tags
        for tag in soup(['nav', 'header', 'footer', 'script', 'style',
                         'aside', 'advertisement', 'iframe']):
            tag.decompose()
        
        text = soup.get_text(separator='\n', strip=True)
        # Collapse multiple blank lines
        import re
        text = re.sub(r'\n{3,}', '\n\n', text)
        
        return text[:3000] + "\n...[content truncated]" if len(text) > 3000 else text
    
    except requests.Timeout:
        return "Webpage timed out after 10 seconds. Try a different URL."
    except Exception as e:
        return f"Could not fetch '{url}': {e}"


# ── Tool 3: Save Note ──────────────────────────────────────────
@tool
def save_note(content: str) -> str:
    """
    Save important information, facts, or reminders to long-term memory.
    Use this when the user says 'remember', 'save', 'note down', or 
    when you learn something important about the user's preferences/context.
    The note is saved permanently and can be found later with search_notes.
    Input: the full text content to remember.
    Returns: confirmation with the note ID.
    """
    try:
        # Load the vector store (creates it if it doesn't exist)
        embeddings = OllamaEmbeddings(model=EMBEDDING_MODEL)
        vectorstore = Chroma(
            collection_name="notes",
            embedding_function=embeddings,
            persist_directory=NOTES_DB_PATH,
        )
        
        # Add the note with a timestamp in the metadata
        # Metadata lets us filter notes later (by date, category, etc.)
        note_id = f"note_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        vectorstore.add_texts(
            texts=[content],
            metadatas=[{
                "id": note_id,
                "timestamp": datetime.now().isoformat(),
                "source": "user_conversation",
            }],
            ids=[note_id]
        )
        
        return f"Note saved successfully (ID: {note_id}).\nContent: '{content[:100]}...'" \
               if len(content) > 100 else f"Note saved (ID: {note_id}).\nContent: '{content}'"
    
    except Exception as e:
        return f"Failed to save note: {e}"


# ── Tool 4: Search Notes ───────────────────────────────────────
@tool
def search_notes(query: str) -> str:
    """
    Search through previously saved notes using semantic similarity.
    Use this when: user asks about something you might have saved earlier,
    user says 'what did I tell you about...', 'do you remember...', 
    'what are my notes on...', or when context from past sessions is needed.
    Input: what to search for (can be a topic, keyword, or full question).
    Returns: the most relevant saved notes with their timestamps.
    """
    try:
        embeddings = OllamaEmbeddings(model=EMBEDDING_MODEL)
        vectorstore = Chroma(
            collection_name="notes",
            embedding_function=embeddings,
            persist_directory=NOTES_DB_PATH,
        )
        
        # Semantic search — finds notes by MEANING not exact keywords
        results = vectorstore.similarity_search_with_score(query, k=3)
        
        if not results:
            return "No saved notes found related to that topic."
        
        output = ["Found relevant notes:"]
        for doc, score in results:
            # score is distance (lower = more similar in ChromaDB)
            relevance = round((1 - score) * 100, 1)  # Convert to % similarity
            timestamp = doc.metadata.get("timestamp", "Unknown time")
            output.append(
                f"\n[{relevance}% match | Saved: {timestamp[:10]}]\n"
                f"{doc.page_content}"
            )
        
        return "\n".join(output)
    
    except Exception as e:
        return f"Search failed: {e}. Note: Run `ollama pull {EMBEDDING_MODEL}` if embeddings error."


# ── Tool 5: Summarize Text ─────────────────────────────────────
@tool
def summarize_text(text: str) -> str:
    """
    Summarize a long piece of text into 5 clear bullet points.
    Use this when: the user provides a long article/document to summarize,
    after fetching a webpage that's too long to include directly,
    or when condensing search results for a cleaner answer.
    Input: the full text to summarize (can be very long).
    Returns: 5 bullet-point summary.
    """
    try:
        # We call the LLM directly here as a "sub-agent" for summarization
        # This is perfectly fine — tools can use LLMs internally!
        llm = ChatOllama(model=LLM_MODEL, temperature=0)
        
        # Truncate very long inputs to avoid context overflow
        truncated = text[:4000] if len(text) > 4000 else text
        
        prompt = f"""Summarize the following text into exactly 5 bullet points.
Each bullet must be one clear, informative sentence.
Start each bullet with •

Text to summarize:
{truncated}

5-bullet summary:"""
        
        response = llm.invoke(prompt)
        return response.content
    
    except Exception as e:
        return f"Summarization failed: {e}"


# ── Tool 6: Get Date & Time ────────────────────────────────────
@tool
def get_datetime(format: str = "full") -> str:
    """
    Get the current date and/or time.
    Use this whenever the user asks: what time is it, what day is today,
    what's today's date, what year/month/week is it.
    Input: 'full' (default), 'date', 'time', or 'day'
    Returns: formatted current date/time string.
    """
    now = datetime.now()
    
    formats = {
        "full": now.strftime("%A, %B %d %Y — %I:%M %p"),  # Tuesday, March 15 2026 — 09:30 AM
        "date": now.strftime("%B %d, %Y"),                 # March 15, 2026
        "time": now.strftime("%I:%M %p"),                  # 09:30 AM
        "day":  now.strftime("%A"),                        # Tuesday
    }
    
    return formats.get(format, formats["full"])


# ── Tool 7 (BONUS for GCP Engineers): Calculate ────────────────
@tool
def calculate(expression: str) -> str:
    """
    Safely evaluate a mathematical expression and return the result.
    Use this for ALL calculations — never try to do math in your head.
    Handles: arithmetic (+,-,*,/), percentages, powers (**), parentheses.
    Examples: '150 * 0.15', '(2400 / 30) * 365', '2 ** 10'
    Input: a valid math expression as a string.
    Returns: the numerical result.
    """
    try:
        # Only allow safe math characters — prevents code injection
        import re
        if re.search(r'[^0-9+\-*/().\s%]', expression):
            return f"Error: expression contains invalid characters. Only numbers and +-*/()%. are allowed."
        
        result = eval(expression, {"__builtins__": {}}, {})  # Safe eval
        return f"{expression} = {round(result, 6)}"
    
    except ZeroDivisionError:
        return "Error: division by zero"
    except Exception as e:
        return f"Calculation error: {e}"


# Export all tools as a list for the agent to use
ALL_TOOLS = [
    search_web,
    fetch_webpage,
    save_note,
    search_notes,
    summarize_text,
    get_datetime,
    calculate,
]