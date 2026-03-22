# tools.py
# ─────────────────────────────────────────────────────────────
# Tools for the Document AI Agent
# The key new tool here is search_documents — the RAG tool.
# ─────────────────────────────────────────────────────────────

from langchain.tools import tool
from langchain_community.vectorstores import FAISS
from duckduckgo_search import DDGS
from datetime import datetime
import re

# Global vectorstore reference — set by the agent after indexing
_vectorstore: FAISS = None

def set_vectorstore(vs: FAISS):
    """Called once during startup to inject the vectorstore into tools."""
    global _vectorstore
    _vectorstore = vs


# ── Tool 1: Search Documents (the RAG tool) ───────────────────
@tool
def search_documents(query: str) -> str:
    """
    Search through the indexed PDF documents using semantic similarity.
    Use this as your PRIMARY tool for ANY question about the uploaded documents.
    This tool finds the most relevant passages from the PDFs and returns them
    with source citations including file name and page number.
    Always use this BEFORE search_web for document-related questions.
    Input: a natural language question or keyword phrase.
    Returns: relevant text passages with source citations.
    """
    if _vectorstore is None:
        return ("No documents have been indexed yet. "
                "Please upload PDFs to the sample_docs/ folder and restart.")

    from rag import retrieve_relevant_chunks
    return retrieve_relevant_chunks(_vectorstore, query)


# ── Tool 2: Web Search ────────────────────────────────────────
@tool
def search_web(query: str) -> str:
    """
    Search the internet for information NOT found in the indexed documents.
    Use this for: current events, general knowledge, topics outside the PDFs.
    Do NOT use this for questions that the uploaded documents can answer —
    use search_documents first for document questions.
    Input: search query string.
    Returns: top 5 web results with titles, URLs, and summaries.
    """
    try:
        with DDGS() as ddgs:
            results = list(ddgs.text(query, max_results=5))
        if not results:
            return f"No results found for '{query}'."
        output = []
        for i, r in enumerate(results, 1):
            output.append(
                f"[{i}] {r.get('title','')}\n"
                f"    {r.get('href','')}\n"
                f"    {r.get('body','')[:200]}"
            )
        return "\n\n".join(output)
    except Exception as e:
        return f"Search error: {e}"


# ── Tool 3: Summarize Document Section ───────────────────────
@tool
def summarize_section(section_content: str) -> str:
    """
    Summarize a long section of document content into clear bullet points.
    Use this when search_documents returns a very long passage and you need
    to condense it before including in your answer.
    Input: the text content to summarize (can be long).
    Returns: 5 concise bullet points covering the key information.
    """
    from langchain_ollama import ChatOllama
    from config import LLM_MODEL
    llm = ChatOllama(model=LLM_MODEL, temperature=0)
    truncated = section_content[:4000]
    response = llm.invoke(
        f"Summarize the following document section into exactly 5 bullet points.\n"
        f"Each bullet starts with •\n\nContent:\n{truncated}\n\nSummary:"
    )
    return response.content


# ── Tool 4: List Indexed Documents ───────────────────────────
@tool
def list_indexed_documents() -> str:
    """
    List all PDF documents that have been indexed and are available to search.
    Use this when the user asks 'what documents do you have?' or 
    'what files are loaded?' or before answering to confirm the right docs exist.
    Returns: list of indexed document filenames with page counts.
    """
    if _vectorstore is None:
        return "No documents indexed yet."

    # Get all documents from the FAISS store's docstore
    all_docs = _vectorstore.docstore._dict
    # Group by source file
    files: dict = {}
    for doc_id, doc in all_docs.items():
        source = doc.metadata.get("source_file", "unknown")
        files[source] = files.get(source, 0) + 1

    if not files:
        return "No documents found in the index."

    lines = ["Indexed documents:"]
    for fname, chunk_count in sorted(files.items()):
        lines.append(f"  • {fname} ({chunk_count} chunks indexed)")
    return "\n".join(lines)


# ── Tool 5: Calculate ─────────────────────────────────────────
@tool
def calculate(expression: str) -> str:
    """
    Safely evaluate a mathematical expression.
    Use for ALL numerical calculations — never guess numbers.
    Input: math expression using +, -, *, /, (, ), **.
    Example: '(1024 * 0.023) * 730'
    Returns: the calculated result.
    """
    try:
        if re.search(r'[^0-9+\-*/().\s%\*]', expression):
            return "Error: only numbers and +-*/()** allowed."
        result = eval(expression, {"__builtins__": {}}, {})
        return f"{expression} = {round(result, 6)}"
    except Exception as e:
        return f"Calculation error: {e}"


# ── Tool 6: Get Date/Time ─────────────────────────────────────
@tool
def get_datetime(format: str = "full") -> str:
    """
    Get the current date and/or time.
    Use when user asks about date, time, day, or year.
    Input: 'full', 'date', 'time', or 'day'.
    """
    now = datetime.now()
    return {
        "full": now.strftime("%A, %B %d %Y — %I:%M %p"),
        "date": now.strftime("%B %d, %Y"),
        "time": now.strftime("%I:%M %p"),
        "day":  now.strftime("%A"),
    }.get(format, now.strftime("%A, %B %d %Y — %I:%M %p"))


ALL_TOOLS = [
    search_documents,
    search_web,
    summarize_section,
    list_indexed_documents,
    calculate,
    get_datetime,
]