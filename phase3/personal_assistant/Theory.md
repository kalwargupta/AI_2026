Explore all 4 tabs — click every layer in "LangChain layers" and every tool card in "Tool registry." Now the full lesson:

---

# ⚙️ PHASE 3 — Build Your First Real AI Agent

> **Goal:** By the end of this phase you'll have a fully working Personal AI Assistant with a web UI, persistent memory, 6 tools, and zero paid APIs.

---

## LESSON 3.1 — LangChain Deep Dive

LangChain is a framework that gives you pre-built pieces to assemble agents — like Lego for AI. Here's every piece you'll use today:

```
LangChain Component          What It Does                  Your GCP Parallel
──────────────────────────────────────────────────────────────────────────
ChatOllama                   Connect to local LLM          BigQuery client
ChatPromptTemplate           Structure LLM requests        SQL query template
@tool decorator              Register agent tools          Cloud Function
AgentExecutor                Run the agent loop            Airflow DAGRun
ConversationBufferMemory     Remember chat history         Dataflow stateful op
Chroma (VectorStore)         Persist notes semantically    BigQuery VECTOR_SEARCH
create_tool_calling_agent    Build the decision maker      Composer operator
```

---

## Final Project Structure

```
ai-agents-course/
└── phase3/
    └── personal_assistant/
        ├── app.py              ← Streamlit web UI
        ├── agent.py            ← Agent assembly
        ├── tools.py            ← All 6 tools
        ├── memory.py           ← Memory setup
        ├── config.py           ← Settings
        ├── requirements.txt
        ├── .env                ← (empty for now, keys go here later)
        └── notes_db/           ← Auto-created by ChromaDB
```

---

## Step 1 — Setup

```bash
mkdir -p ai-agents-course/phase3/personal_assistant
cd ai-agents-course/phase3/personal_assistant
python -m venv venv && source venv/bin/activate   # Windows: venv\Scripts\activate

pip install langchain langchain-community langchain-ollama \
            langchain-chroma chromadb \
            duckduckgo-search requests beautifulsoup4 \
            streamlit python-dotenv rich

# Pull a model that handles tools well (qwen2.5 is best for free local use)
ollama pull qwen2.5       # 4.7GB — best tool-calling
# OR if storage is tight:
ollama pull llama3.1      # 4.7GB — good alternative
```

---

## Step 2 — `config.py`

```python
# config.py
# ─────────────────────────────────────────────────────
# Central config file. Change your model here and it
# updates everywhere. Like a Terraform variables file.
# ─────────────────────────────────────────────────────

# Which local LLM to use (must be pulled via `ollama pull <name>`)
LLM_MODEL = "qwen2.5"        # Best for tool-calling. Change to "llama3.1" if needed.
LLM_TEMPERATURE = 0          # 0 = precise/factual. Increase for creative tasks.

# Memory settings
MEMORY_WINDOW_SIZE = 10      # Remember last 10 conversation turns in RAM
NOTES_DB_PATH = "./notes_db" # Where ChromaDB saves notes on disk
NOTES_COLLECTION = "assistant_notes"  # ChromaDB collection name

# Embedding model for vector search (runs locally via Ollama)
EMBEDDING_MODEL = "nomic-embed-text"  # Pull with: ollama pull nomic-embed-text

# Agent settings
MAX_ITERATIONS = 12          # Max tool calls before forced finish
MAX_EXEC_TIME  = 90          # Seconds before timeout
```

---

## Step 3 — `memory.py`

```python
# memory.py
# ─────────────────────────────────────────────────────
# Two memory systems working together:
#   1. ConversationBufferWindowMemory  → short-term (RAM)
#   2. ChromaDB                        → long-term (disk)
# ─────────────────────────────────────────────────────

from langchain.memory import ConversationBufferWindowMemory
from langchain_chroma import Chroma
from langchain_ollama import OllamaEmbeddings
from config import MEMORY_WINDOW_SIZE, NOTES_DB_PATH, NOTES_COLLECTION, EMBEDDING_MODEL


def get_short_term_memory() -> ConversationBufferWindowMemory:
    """
    Short-term memory: remembers last k conversation turns.
    Lives in RAM — cleared when the program restarts.
    
    Like a Dataflow pipeline's in-memory state window.
    """
    return ConversationBufferWindowMemory(
        k=MEMORY_WINDOW_SIZE,   # Keep last 10 turns (20 messages: 10 human + 10 AI)
        memory_key="chat_history",  # Key name used in the prompt template
        return_messages=True,       # Return Message objects (not plain strings)
    )


def get_vector_store() -> Chroma:
    """
    Long-term memory: vector database that persists to disk.
    Survives restarts. Searchable by semantic meaning.
    
    Like a BigQuery table with VECTOR_SEARCH — but local and free.
    """
    # The embedding model converts text → vectors for storage and search
    # nomic-embed-text is a free, fast embedding model that runs via Ollama
    embeddings = OllamaEmbeddings(model=EMBEDDING_MODEL)
    
    # Chroma creates the ./notes_db/ folder on first run
    # On subsequent runs it loads existing notes from disk
    vectorstore = Chroma(
        collection_name=NOTES_DB_PATH,
        embedding_function=embeddings,
        persist_directory=NOTES_DB_PATH,  # Save to disk (not just RAM)
    )
    
    return vectorstore
```

---

## Step 4 — `tools.py` — All 6 Tools

```python
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
```

---

## Step 5 — `agent.py` — Assembly

```python
# agent.py
# ─────────────────────────────────────────────────────
# Assembles all pieces into a working agent.
# Analogy: this is your Airflow DAG definition file —
# it wires up all the tasks (tools) and sets the schedule
# (AgentExecutor config).
# ─────────────────────────────────────────────────────

from langchain_ollama import ChatOllama
from langchain.agents import AgentExecutor, create_tool_calling_agent
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.memory import ConversationBufferWindowMemory

from tools import ALL_TOOLS
from memory import get_short_term_memory
from config import (LLM_MODEL, LLM_TEMPERATURE,
                    MAX_ITERATIONS, MAX_EXEC_TIME)


def build_system_prompt() -> str:
    """
    The system prompt is the agent's identity card.
    It tells the LLM: who you are, what you can do, how to behave.
    
    Keep it focused. The more specific, the better the agent behaves.
    """
    return """You are a helpful Personal AI Assistant with expertise in:
- Data Engineering (BigQuery, Airflow, Spark, GCP, Python ETL)
- General knowledge and research
- Summarizing articles and documents
- Taking and retrieving personal notes

Your available tools:
- search_web: find current information online
- fetch_webpage: read full articles from URLs
- save_note: permanently save important info to memory
- search_notes: find previously saved notes semantically
- summarize_text: condense long content to bullet points
- get_datetime: current date and time
- calculate: any mathematical calculation

Behavior rules:
1. Always use calculate for math — never guess numbers
2. Always use get_datetime when asked about date/time — never guess
3. When user says "remember" or "save", ALWAYS call save_note
4. When user asks "what did I tell you" or "do you remember", call search_notes FIRST
5. Give concise, direct answers. Use bullet points for lists.
6. If you don't know something, say so and offer to search for it.
7. For GCP/data engineering questions, be specific and practical."""


def create_agent(memory: ConversationBufferWindowMemory = None) -> AgentExecutor:
    """
    Builds and returns a fully configured AgentExecutor.
    
    Args:
        memory: optional pre-built memory object
                (pass existing memory to preserve chat history)
    
    Returns:
        AgentExecutor ready to handle requests
    """
    
    # ── 1. LLM ────────────────────────────────────────────────
    llm = ChatOllama(
        model=LLM_MODEL,
        temperature=LLM_TEMPERATURE,
    )
    
    # ── 2. Prompt Template ────────────────────────────────────
    # MessagesPlaceholder("chat_history") is where LangChain
    # automatically injects the conversation history from memory.
    # MessagesPlaceholder("agent_scratchpad") is where tool
    # call results get injected during the ReAct loop.
    prompt = ChatPromptTemplate.from_messages([
        ("system", build_system_prompt()),
        MessagesPlaceholder("chat_history"),     # ← memory injected here
        ("human", "{input}"),
        MessagesPlaceholder("agent_scratchpad"), # ← tool loop injected here
    ])
    
    # ── 3. Memory (use provided or create fresh) ──────────────
    if memory is None:
        memory = get_short_term_memory()
    
    # ── 4. Agent (decision maker) ─────────────────────────────
    # create_tool_calling_agent uses the LLM's native function-calling
    # instead of text-based ReAct parsing → much more reliable!
    agent = create_tool_calling_agent(
        llm=llm,
        tools=ALL_TOOLS,
        prompt=prompt,
    )
    
    # ── 5. AgentExecutor (the loop engine) ───────────────────
    agent_executor = AgentExecutor(
        agent=agent,
        tools=ALL_TOOLS,
        memory=memory,                    # Injects chat_history into prompt
        verbose=True,                     # Show tool calls in terminal
        max_iterations=MAX_ITERATIONS,
        max_execution_time=MAX_EXEC_TIME,
        handle_parsing_errors=True,
        early_stopping_method="generate", # Write best answer if limit hit
        return_intermediate_steps=True,   # For debugging in the UI
    )
    
    return agent_executor


def chat(agent_executor: AgentExecutor, user_input: str) -> dict:
    """
    Send one message to the agent and get back the full result.
    
    Returns a dict with:
      - output: the agent's final answer (string)
      - steps:  list of (action, observation) tool call pairs
    """
    result = agent_executor.invoke({"input": user_input})
    
    return {
        "output": result.get("output", "No response generated."),
        "steps":  result.get("intermediate_steps", []),
    }
```

---

## Step 6 — `app.py` — Streamlit Web UI

```python
# app.py
# ─────────────────────────────────────────────────────
# Streamlit web interface for the Personal AI Assistant.
# Run with: streamlit run app.py
#
# Streamlit works by re-running this entire file on every
# user interaction. We use st.session_state to persist
# data (like the agent and chat history) between reruns.
# ─────────────────────────────────────────────────────

import streamlit as st
from agent import create_agent, chat

# ── Page config (must be first Streamlit call) ────────────────
st.set_page_config(
    page_title="Personal AI Assistant",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Initialize session state ──────────────────────────────────
# session_state persists values across Streamlit reruns
# (like module-level variables but Streamlit-aware)

if "agent" not in st.session_state:
    # Build the agent ONCE and reuse it — preserves memory between messages
    with st.spinner("🔧 Loading AI Assistant (first load may take 30s)..."):
        st.session_state.agent = create_agent()
    st.success("✅ Assistant ready!")

if "messages" not in st.session_state:
    # Chat display history (separate from LangChain's internal memory)
    st.session_state.messages = []

if "tool_calls" not in st.session_state:
    # Track tool calls for the debug sidebar
    st.session_state.tool_calls = []


# ── Sidebar ───────────────────────────────────────────────────
with st.sidebar:
    st.title("🤖 Personal Assistant")
    st.caption("Phase 3 · Local LLM · No paid APIs")
    
    st.divider()
    
    # Show available tools
    st.subheader("🛠️ Available Tools")
    tools_info = {
        "🔍 search_web":     "Live internet search",
        "📄 fetch_webpage":  "Read full articles",
        "💾 save_note":      "Remember things forever",
        "🔎 search_notes":   "Find saved notes",
        "📝 summarize_text": "Bullet-point summaries",
        "📅 get_datetime":   "Current date & time",
        "🧮 calculate":      "Safe math calculations",
    }
    for tool_name, desc in tools_info.items():
        st.markdown(f"**{tool_name}** — {desc}")
    
    st.divider()
    
    # Quick test prompts
    st.subheader("💡 Try these prompts")
    quick_prompts = [
        "What is today's date?",
        "What is 15% of 4500?",
        "Remember: I prefer Python over Java",
        "What do you know about my preferences?",
        "What is LangGraph in one paragraph?",
        "Search for the latest news about Gemini AI",
    ]
    for prompt in quick_prompts:
        if st.button(prompt, use_container_width=True):
            st.session_state.quick_input = prompt
    
    st.divider()
    
    # Tool call history
    if st.session_state.tool_calls:
        st.subheader("🔧 Recent Tool Calls")
        for tc in st.session_state.tool_calls[-5:]:  # Show last 5
            with st.expander(f"→ {tc['tool']}"):
                st.code(str(tc['input'])[:200], language="text")
                st.caption(str(tc['output'])[:300])
    
    # Clear conversation button
    st.divider()
    if st.button("🗑️ Clear conversation", use_container_width=True):
        st.session_state.messages = []
        st.session_state.tool_calls = []
        # Rebuild agent with fresh memory
        st.session_state.agent = create_agent()
        st.rerun()


# ── Main chat area ────────────────────────────────────────────
st.title("🤖 Personal AI Assistant")
st.caption("Powered by local LLM (Ollama) · Persistent notes · Web search")

# Display all previous messages
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        # Show which tools were called (for assistant messages)
        if msg.get("tools_used"):
            with st.expander(f"🔧 Used {len(msg['tools_used'])} tool(s)", expanded=False):
                for t in msg["tools_used"]:
                    st.caption(f"→ **{t['tool']}** | input: `{str(t['input'])[:80]}`")

# Handle quick input from sidebar buttons
if "quick_input" in st.session_state:
    user_input = st.session_state.pop("quick_input")
else:
    user_input = None

# Chat input box
typed_input = st.chat_input("Ask me anything... (try: 'Remember my standup is at 9am')")
if typed_input:
    user_input = typed_input

# Process the message
if user_input:
    # Show user message
    st.session_state.messages.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.markdown(user_input)
    
    # Get agent response
    with st.chat_message("assistant"):
        with st.spinner("🤔 Thinking..."):
            try:
                result = chat(st.session_state.agent, user_input)
                
                output     = result["output"]
                steps      = result["steps"]
                
                # Display the answer
                st.markdown(output)
                
                # Collect tool usage info for display
                tools_used = []
                for action, observation in steps:
                    tools_used.append({
                        "tool":   action.tool,
                        "input":  action.tool_input,
                        "output": str(observation)[:200],
                    })
                    # Also store in sidebar history
                    st.session_state.tool_calls.append({
                        "tool":   action.tool,
                        "input":  action.tool_input,
                        "output": str(observation),
                    })
                
                # Show tool calls inline if any were made
                if tools_used:
                    with st.expander(f"🔧 Used {len(tools_used)} tool(s)", expanded=False):
                        for t in tools_used:
                            st.caption(f"→ **{t['tool']}** | `{str(t['input'])[:80]}`")
                
                # Save to display history
                st.session_state.messages.append({
                    "role":       "assistant",
                    "content":    output,
                    "tools_used": tools_used,
                })
            
            except Exception as e:
                error_msg = f"⚠️ Error: {str(e)}\n\nMake sure Ollama is running: `ollama serve`"
                st.error(error_msg)
                st.session_state.messages.append({"role": "assistant", "content": error_msg})
```

---

## Step 7 — `requirements.txt`

```text
langchain==0.3.7
langchain-community==0.3.7
langchain-ollama==0.2.1
langchain-chroma==0.1.4
chromadb==0.5.18
duckduckgo-search==6.3.5
requests==2.32.3
beautifulsoup4==4.12.3
streamlit==1.40.0
python-dotenv==1.0.1
rich==13.9.4
```

---

## Step 8 — Run It

```bash
# Terminal 1 — keep Ollama running
ollama serve

# Terminal 2 — launch the web app
cd ai-agents-course/phase3/personal_assistant
source venv/bin/activate

# Pull the embedding model (needed for save_note / search_notes)
ollama pull nomic-embed-text

# Run!
streamlit run app.py
# Opens at: http://localhost:8501
```

---

## Step 9 — Test Sequence (Run These in Order)

```
# Test 1 — No tool needed
"What is a GCP Dataflow pipeline?"

# Test 2 — Uses calculate tool
"If I process 3.7 TB in BigQuery at $5 per TB, what's my cost?"

# Test 3 — Uses get_datetime tool
"What day of the week is today?"

# Test 4 — Uses search_web tool
"What are the newest features in LangChain 0.3?"

# Test 5 — Uses save_note tool (IMPORTANT - test memory!)
"Remember: I'm a GCP Data Engineer learning AI Agents in 2026"

# Test 6 — Uses search_notes tool (should find what you saved)
"What do you know about me from my notes?"

# Test 7 — Uses search_web + summarize_text tools (2 tools!)
"Search for what CrewAI is and summarize it in bullet points"

# Test 8 — Multi-turn memory test
"My favourite cloud is GCP"
"What cloud did I just say I like?"  ← should remember from context

# Test 9 — Cross-session memory test
# Close and reopen the browser tab, then:
"What do you know about me?"  ← should recall from ChromaDB notes
```

---

## Debugging Cheatsheet

```bash
# Problem: "ollama: command not found"
# Fix: make sure Ollama daemon is running in another terminal
ollama serve

# Problem: "nomic-embed-text not found"
# Fix:
ollama pull nomic-embed-text

# Problem: ChromaDB error on first run
# Fix: delete the db folder and restart
rm -rf ./notes_db && streamlit run app.py

# Problem: Agent gives wrong answer without using tools
# Fix: make the tool docstrings more directive, e.g.
#   "ALWAYS use this for date/time questions — never guess the date"

# Problem: slow responses
# Fix: try a smaller/faster model
#   ollama pull llama3.2:1b    ← 1.3GB, very fast
#   Change LLM_MODEL in config.py

# View what's stored in ChromaDB (useful for debugging notes)
python3 -c "
from langchain_chroma import Chroma
from langchain_ollama import OllamaEmbeddings
db = Chroma(collection_name='notes', embedding_function=OllamaEmbeddings(model='nomic-embed-text'), persist_directory='./notes_db')
print(db.get())
"
```

---

## 🎯 Coding Task — 3 Extensions

**Task 1 — Add a `list_all_notes` tool:**
```python
@tool
def list_all_notes() -> str:
    """
    List ALL saved notes with their timestamps.
    Use when user asks 'what have I saved?' or 'show all my notes'.
    Returns: numbered list of all notes ordered by date.
    """
    # Hint: use vectorstore.get() to retrieve all documents
    # Then format them as a numbered list with timestamps
    pass
```

**Task 2 — Add a GCP BigQuery tool (use your real skills!):**
```python
@tool
def explain_sql(sql_query: str) -> str:
    """
    Explain what a SQL or BigQuery query does in plain English.
    Also suggest one optimization if possible.
    Use when user pastes a SQL query and asks what it does.
    """
    # Hint: call the LLM directly (like summarize_text does)
    # with a prompt like: "Explain this BigQuery SQL step by step:"
    pass
```

**Task 3 — Add conversation export:**
```python
# In app.py sidebar, add a download button:
if st.session_state.messages:
    import json
    export = json.dumps(st.session_state.messages, indent=2)
    st.download_button(
        label="⬇️ Export conversation",
        data=export,
        file_name="conversation.json",
        mime="application/json"
    )
```

---

## 📝 Phase 3 Quiz

1. Why do we keep `st.session_state.agent` instead of recreating the agent on every message?
2. What is `MessagesPlaceholder("chat_history")` doing in the prompt template?
3. Why does the `save_note` tool need an embedding model but `search_web` doesn't?
4. What happens if you change `k=10` in `ConversationBufferWindowMemory` to `k=2`?
5. Why is `temperature=0` used for this assistant? When would you increase it?
6. In `calculate`, why do we use `eval(expression, {"__builtins__": {}}, {})` instead of plain `eval(expression)`?
7. What's the difference between `st.session_state.messages` and LangChain's `ConversationBufferWindowMemory`?

---

## 💼 Interview Questions

> "How would you make an AI agent's memory persist across user sessions?"

> "Explain the difference between `create_react_agent` and `create_tool_calling_agent` — when would you pick each?"

> "How do you prevent prompt injection attacks in a production AI agent?"

> "A user's conversation has 200 turns. How does memory work and what breaks first?"

> "How would you add authentication to this Streamlit agent app?"

---

## Phase 4 Preview

```
PHASE 3 (DONE ✅):             PHASE 4 (NEXT 🔜):
  ├── LangChain deep dive        ├── ReAct agents (deep)
  ├── 7 custom tools             ├── RAG — chat with your own PDFs
  ├── Short + long-term memory   ├── FAISS vs ChromaDB benchmarked
  ├── Streamlit web UI           ├── Autonomous agents (no hand-holding)
  └── Personal Assistant ✅      └── Document AI Agent:
                                       reads PDFs → answers questions
                                       from YOUR documents 🔜
```

Phase 4 is where it gets seriously powerful — you'll build an agent that can read any PDF you give it (GCP docs, Airflow docs, your company runbook) and answer questions from it. That's the thing that gets you hired.

**Run the assistant, complete the 3 coding tasks, then say "ready for Phase 4" when done!** 🚀