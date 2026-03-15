Let's build your Phase 2 interactive diagram first — then I'll teach every concept with full code.Now let me build the interactive Phase 2 visuals and full lesson.Explore all 4 tabs — click every step in "The Agent Loop" to see what happens at each stage. Now let's go deep on every concept.

---

# 🤖 PHASE 2 — Understanding AI Agents

> **Your mental model shift:** A chatbot answers questions. An agent **completes goals**. This is the single biggest difference you need to internalize.

---

## LESSON 2.1 — What is an AI Agent?

### 📖 What it is
An AI Agent is an AI system that can **perceive its environment, make decisions, take actions, and work towards a goal** — often over multiple steps — without you specifying every step.

```
CHATBOT (what you built in Phase 1):
  You:  "What is the capital of France?"
  AI:   "Paris"
  Done. One round trip. No autonomy.

AI AGENT:
  You:  "Research the top 3 cloud providers, compare their BigQuery 
         equivalents, and save a report to my Google Drive."

  Agent: Step 1 → Search web for "Google BigQuery alternatives 2024"
         Step 2 → Search web for "AWS Athena vs BigQuery"
         Step 3 → Search web for "Azure Synapse vs BigQuery"
         Step 4 → Synthesize all results
         Step 5 → Format as a structured report
         Step 6 → Call Google Drive API to save the file
         Step 7 → Report back: "Done! Report saved to your Drive."

  You did nothing after the first instruction. The agent figured out Steps 1-7 itself.
```

### 🔌 GCP Analogy You Know
```
Chatbot          ≈   A single BigQuery SELECT query
AI Agent         ≈   A full Airflow DAG
                     - Has multiple tasks
                     - Tasks depend on each other
                     - Can branch and loop
                     - Can call external APIs
                     - Runs autonomously until done
```

---

## LESSON 2.2 — Agent vs Chatbot (Deep Comparison)

| Feature | Chatbot | AI Agent |
|---|---|---|
| **Input** | One question | A goal or complex task |
| **Output** | One response | Actions + final answer |
| **Steps** | Single turn | Multi-step, autonomous |
| **Tools** | None | Web, code, APIs, databases |
| **Memory** | Forgets (or just chat history) | Short-term + Long-term |
| **Planning** | No | Yes — breaks goals into subtasks |
| **Example** | "What is Spark?" | "Optimize my Spark job and email me the report" |
| **Frameworks** | OpenAI API directly | LangChain, CrewAI, AutoGen |

---

## LESSON 2.3 — Agent Architecture (The ReAct Loop)

The most important pattern in AI Agents is **ReAct** — **Re**asoning + **Act**ing. It was introduced in a 2022 Google/Princeton paper and is the foundation of nearly every agent today.

```
┌─────────────────────────────────────────────────────────┐
│                   THE REACT LOOP                         │
│                                                          │
│  Thought:  "The user wants current Bitcoin price.        │
│             I should search the web."                    │
│                                                          │
│  Action:   web_search("Bitcoin price today")            │
│                                                          │
│  Observation: "$67,432 as of 2pm EST"                   │
│                                                          │
│  Thought:  "I have the price. I can now answer."        │
│                                                          │
│  Final Answer: "Bitcoin is currently $67,432."          │
│                                                          │
│         ↑ This Thought→Action→Observation loop          │
│           repeats until the agent decides it's done.    │
└─────────────────────────────────────────────────────────┘
```

This is exactly what you see when you use ChatGPT's "browse" feature or Claude's web search. The LLM generates the "Thought" text, then decides whether to call a tool.

---

## LESSON 2.4 — Tools

### 📖 What it is
Tools are **functions the agent can call** to interact with the outside world. Without tools, an agent is just a fancy chatbot. Tools give it **superpowers**.

```python
# A tool is literally just a Python function with a description
# The LLM reads the description and decides WHEN and HOW to call it

def search_web(query: str) -> str:
    """Search the internet for current information."""
    # ... calls DuckDuckGo or SerpAPI
    return results

def run_python(code: str) -> str:
    """Execute Python code and return the output."""
    # ... runs code in a sandbox
    return output

def read_file(filepath: str) -> str:
    """Read a file from disk."""
    return open(filepath).read()

def query_bigquery(sql: str) -> str:
    """Run a SQL query on BigQuery and return results."""
    # ... calls BigQuery API (YOUR SUPERPOWER!)
    return results
```

### Common Tool Categories
```
┌─────────────────────────────────────────────────────┐
│                   AGENT TOOLS                        │
│                                                      │
│  📡 INFORMATION          🔧 EXECUTION                │
│  ├── Web search          ├── Python executor         │
│  ├── Wikipedia           ├── Terminal/bash           │
│  ├── News APIs           └── SQL runner              │
│  └── Vector DB search                                │
│                          🌐 INTEGRATION              │
│  📁 FILE SYSTEM          ├── REST APIs               │
│  ├── Read/write files    ├── Google Drive            │
│  ├── Parse PDFs          ├── Slack/Email             │
│  └── Process CSVs        └── GitHub                  │
└─────────────────────────────────────────────────────┘
```

---

## LESSON 2.5 — Memory

### 📖 What it is
Memory lets agents **remember things** across steps and sessions. There are 4 types:

```
TYPE 1: SENSORY (In-context)
  What: The current prompt + everything in the LLM's context window
  How long: Until the conversation ends
  Like: RAM — fast, temporary
  GCP analog: Pub/Sub message in flight

TYPE 2: SHORT-TERM (Conversation history)
  What: The list of messages in the current session
  How long: The current session only
  Like: L1/L2 cache
  GCP analog: Dataflow in-memory state

TYPE 3: LONG-TERM (Vector DB / External)
  What: Important facts stored in a vector database
  How long: Permanent (until deleted)
  Like: Hard drive
  GCP analog: BigQuery or GCS — persisted storage

TYPE 4: EPISODIC (Summarized past interactions)
  What: Compressed summaries of past sessions
  How long: Permanent
  Like: Your diary
  GCP analog: Composer/Airflow XComs between DAG runs
```

---

## LESSON 2.6 — Planning

### 📖 What it is
Planning is the agent's ability to **break a complex goal into smaller steps** and figure out the best order to execute them.

```
Goal: "Build me a Python ETL pipeline for our sales data"

Bad Agent (no planning):
  → Immediately writes random code
  → Doesn't know what the data looks like
  → Produces garbage

Good Agent (with planning):
  Plan Step 1: Understand the data → ask user for schema
  Plan Step 2: Design the pipeline → read BigQuery docs
  Plan Step 3: Write extraction code → use GCS reader
  Plan Step 4: Write transformation → apply business rules
  Plan Step 5: Write loading code → BigQuery writer
  Plan Step 6: Test it → run on sample data
  Plan Step 7: Return final code + documentation
```

The two most common planning patterns are **Chain-of-Thought** (think step by step) and **Tree-of-Thoughts** (explore multiple approaches, pick the best).

---

## LESSON 2.7 — Multi-Agent Systems

### 📖 What it is
Instead of one agent doing everything, you build **teams of specialized agents** that collaborate — just like a real company.

```
SINGLE AGENT (does everything):
  User → Agent → [searches, codes, writes, analyzes] → Output
  Problem: Jack of all trades, master of none. Gets confused on complex tasks.

MULTI-AGENT (specialized team):
  User → Orchestrator Agent → Research Agent → Writer Agent
                           → Coder Agent   → QA Agent
  
  Each agent is an EXPERT in its domain.
  The orchestrator is the "project manager."
  
  GCP Analog: Composer (Airflow) orchestrating
              specialized Dataflow jobs
```

---

# 🛠️ MINI PROJECT 2 — Research AI Agent

Build an agent that **searches the internet, reads articles, and writes a research report** — all autonomously.

### Folder Structure
```
ai-agents-course/
└── phase2/
    └── research_agent/
        ├── agent.py              ← Main agent
        ├── tools.py              ← Tool definitions
        ├── requirements.txt      ← Dependencies
        ├── .env                  ← API keys (never commit!)
        └── README.md
```

### Step 1: Install dependencies
```bash
mkdir -p ai-agents-course/phase2/research_agent
cd ai-agents-course/phase2/research_agent
python -m venv venv && source venv/bin/activate

pip install langchain langchain-community langchain-ollama \
            duckduckgo-search requests beautifulsoup4 \
            python-dotenv rich
```

### Step 2: `tools.py` — Define the agent's tools
```python
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
```

### Step 3: `agent.py` — The main agent
```python
# ============================================================
# agent.py — The Research AI Agent
# Uses LangChain + Ollama (FREE, local LLM)
# Pattern: ReAct (Reasoning + Acting)
# ============================================================

from langchain_ollama import ChatOllama           # Connects to our local Ollama LLM
from langchain.agents import create_react_agent   # Creates a ReAct-pattern agent
from langchain.agents import AgentExecutor        # Runs the agent loop
from langchain import hub                         # Downloads ReAct prompt template
from langchain.memory import ConversationBufferMemory  # Short-term memory
from langchain_core.prompts import PromptTemplate

# Import our tools from tools.py
from tools import search_web, fetch_webpage, save_report, calculate

# Rich library for beautiful terminal output
from rich.console import Console
from rich.panel import Panel
from rich.text import Text
from rich import print as rprint
import sys

# Initialize rich console for pretty printing
console = Console()


def create_research_agent():
    """
    Build and return our Research AI Agent.
    
    Architecture:
      LLM (Llama 3.2 via Ollama)
        + ReAct Prompt (Thought → Action → Observation loop)
        + Tools (search, fetch, save, calculate)
        + Memory (conversation history)
      = Research Agent
    """
    
    console.print("\n[bold cyan]🤖 Initializing Research AI Agent...[/bold cyan]")
    
    # ── Step 1: Connect to the LLM ─────────────────────────────
    # ChatOllama connects to Ollama running on your machine
    # temperature=0 means: be precise, not creative (good for research)
    llm = ChatOllama(
        model="llama3.2",       # The LLM we downloaded via `ollama pull llama3.2`
        temperature=0,          # 0 = deterministic, 1 = creative
        # timeout=120,          # Wait up to 2 min for slow responses
    )
    console.print("  ✅ LLM connected: Llama 3.2 (local)")
    
    # ── Step 2: Register the tools ─────────────────────────────
    # This is the list of tools the agent CAN choose to use
    tools = [search_web, fetch_webpage, save_report, calculate]
    console.print(f"  ✅ Tools registered: {[t.name for t in tools]}")
    
    # ── Step 3: Set up the ReAct prompt ────────────────────────
    # The ReAct prompt tells the LLM HOW to use tools in the loop:
    # "Thought: ... \nAction: tool_name\nAction Input: ...\nObservation: ..."
    # We download a pre-built template from LangChain Hub
    
    # Custom system prompt that gives the agent its identity
    system_prompt = """You are an expert Research Agent with access to web search tools.

Your job is to:
1. Research topics thoroughly using web search
2. Read relevant articles for depth  
3. Synthesize information into clear, structured reports
4. Save reports when asked

Always cite sources. Be accurate. Acknowledge when you're uncertain.
When you have enough information to answer, stop searching and compile your report.

You have access to the following tools:
{tools}

Use this format EXACTLY:
Question: the input question you must answer
Thought: think about what to do
Action: the action to take, should be one of [{tool_names}]
Action Input: the input to the action
Observation: the result of the action
... (this Thought/Action/Action Input/Observation can repeat N times)
Thought: I now know the final answer
Final Answer: the final answer to the original input question

Begin!

Question: {input}
Thought: {agent_scratchpad}"""
    
    prompt = PromptTemplate(
        template=system_prompt,
        input_variables=["input", "agent_scratchpad"],
        partial_variables={
            "tools": "\n".join([f"{t.name}: {t.description}" for t in tools]),
            "tool_names": ", ".join([t.name for t in tools])
        }
    )
    
    # ── Step 4: Create the ReAct agent ─────────────────────────
    # This combines: LLM + ReAct prompt + tools
    # The agent will: read the prompt, reason, pick a tool, observe, repeat
    agent = create_react_agent(
        llm=llm,
        tools=tools,
        prompt=prompt
    )
    console.print("  ✅ ReAct agent created")
    
    # ── Step 5: Wrap in AgentExecutor ──────────────────────────
    # AgentExecutor is the "engine" that runs the ReAct loop
    agent_executor = AgentExecutor(
        agent=agent,
        tools=tools,
        verbose=True,              # Show Thought/Action/Observation (great for learning!)
        max_iterations=10,         # Max 10 tool calls per task (prevents infinite loops)
        handle_parsing_errors=True # Don't crash on LLM format mistakes
    )
    console.print("  ✅ Agent executor ready\n")
    
    return agent_executor


def run_research_session():
    """
    Interactive research session.
    The user can give research tasks and the agent executes them.
    """
    
    # Build the agent
    agent = create_research_agent()
    
    # Display welcome message
    console.print(Panel(
        "[bold]Welcome to the Research AI Agent[/bold]\n\n"
        "I can:\n"
        "  🔍 Search the web for current information\n"
        "  📄 Read and analyze web articles\n"
        "  💾 Save research reports to files\n"
        "  🧮 Do calculations\n\n"
        "[dim]Type 'quit' to exit[/dim]",
        title="🤖 Research Agent",
        border_style="cyan"
    ))
    
    # Example research tasks to try
    examples = [
        "Research the top 3 open-source LLM frameworks in 2024 and compare them",
        "What are the latest developments in AI agents? Write a brief report and save it.",
        "Compare Google BigQuery, Snowflake, and Databricks on pricing and features",
    ]
    
    console.print("\n[bold cyan]Example tasks to try:[/bold cyan]")
    for i, ex in enumerate(examples, 1):
        console.print(f"  {i}. {ex}")
    console.print()
    
    # Main interaction loop
    while True:
        # Get user input
        user_input = console.input("\n[bold green]You:[/bold green] ").strip()
        
        if not user_input:
            continue
        if user_input.lower() in ['quit', 'exit', 'bye']:
            console.print("[bold cyan]Goodbye! Keep building agents! 🚀[/bold cyan]")
            break
        
        console.print("\n[bold cyan]🤖 Agent working...[/bold cyan]\n")
        
        try:
            # Run the agent! This triggers the ReAct loop:
            # 1. LLM reads the task
            # 2. Thinks about what tool to use
            # 3. Calls the tool
            # 4. Reads the observation
            # 5. Thinks again
            # 6. Repeats until it has the answer
            result = agent.invoke({
                "input": user_input  # The user's research task
            })
            
            # Display the final answer nicely
            console.print(Panel(
                result["output"],           # The agent's final answer
                title="📋 Research Result",
                border_style="green"
            ))
            
        except Exception as e:
            console.print(f"[red]Error: {e}[/red]")
            console.print("[yellow]Tip: Make sure Ollama is running: `ollama serve`[/yellow]")


# ── Entry point ──────────────────────────────────────────────
if __name__ == "__main__":
    run_research_session()
```

### Step 4: `requirements.txt`
```text
langchain==0.3.7
langchain-community==0.3.7
langchain-ollama==0.2.1
duckduckgo-search==6.3.5
requests==2.32.3
beautifulsoup4==4.12.3
python-dotenv==1.0.1
rich==13.9.4
```

### Step 5: How to Run
```bash
# 1. Make sure Ollama is running with Llama 3.2
ollama serve                  # Terminal 1 — keep this running
ollama pull llama3.2          # If you haven't pulled it yet

# 2. Install dependencies
cd ai-agents-course/phase2/research_agent
pip install -r requirements.txt

# 3. Run the agent!
python agent.py               # Terminal 2

# 4. Try these research tasks:
# → "Research what LangChain is and how it compares to LlamaIndex"
# → "What is RAG in AI? Explain it and save a report."
# → "Find the latest news about Google Gemini and summarize it"
```

### Step 6: Understanding the Output (Verbose Mode)
When you run with `verbose=True` you'll see the ReAct loop live:
```
> Entering new AgentExecutor chain...
  Thought: I need to search for information about LangChain
  Action: search_web
  Action Input: "LangChain framework 2024 overview"
  Observation: [search results appear here...]

  Thought: I have some results. Let me fetch the top article for more detail.
  Action: fetch_webpage
  Action Input: "https://python.langchain.com/docs/..."
  Observation: [webpage content appears here...]

  Thought: I now have enough information to write a comprehensive answer.
  Final Answer: LangChain is a framework for building LLM-powered applications...
```

This is the ReAct loop you learned about — now you can see it running in real time!

---

## 🎯 YOUR CODING TASK (Phase 2)

Extend the Research Agent with these 3 enhancements:

**Task 1 — Add a "summarize" tool:**
```python
@tool
def summarize_text(text: str) -> str:
    """
    Summarize a long piece of text into 3-5 bullet points.
    Use this when you have a long article and need key points.
    """
    # HINT: Call Ollama directly here with a summarization prompt
    # Use the ollama library from Phase 1!
    pass
```

**Task 2 — Add conversation memory:**
```python
# HINT: Replace the simple agent with a memory-enabled version
from langchain.memory import ConversationBufferWindowMemory

memory = ConversationBufferWindowMemory(
    k=5,                    # Remember last 5 exchanges
    return_messages=True
)
# How do you pass this to AgentExecutor? (hint: memory= parameter)
```

**Task 3 — Add a "GCP-powered" tool:**
```python
@tool
def query_public_bigquery(query: str) -> str:
    """
    Run a SQL query against BigQuery public datasets.
    Great for research involving public data.
    Example query: 'SELECT * FROM bigquery-public-data.samples.shakespeare LIMIT 5'
    """
    # HINT: Use google-cloud-bigquery library
    # from google.cloud import bigquery
    # client = bigquery.Client()  ← Uses your GCP credentials!
    pass
```

---

## 📝 Phase 2 Quiz

**1.** What is the fundamental difference between a chatbot and an AI agent? Give an example.

**2.** Explain the ReAct pattern in your own words. What do R and A stand for?

**3.** In the code above, what does `max_iterations=10` protect against?

**4.** What are the 4 types of memory an agent can have? Which one is like a Vector DB?

**5.** In the `@tool` decorator, why does the docstring matter so much?

**6.** Draw (in ASCII) the flow of a user asking the Research Agent: *"What is Apache Kafka?"*

**7.** What is the GCP equivalent of an AI Agent Orchestrator?

---

## 💼 Phase 2 Interview Questions

> "How does the ReAct pattern work and when would you choose it over a simple LLM call?"

> "What is the difference between short-term and long-term memory in AI agents?"

> "How would you prevent an agent from running in an infinite loop?"

> "Explain a multi-agent system you would design for a data engineering use case."

> "What are the tradeoffs of giving an agent many tools vs few tools?"

---

## 🗺️ Phase 3 Preview

```
PHASE 2 (DONE ✅):              PHASE 3 (NEXT 🔜):
  ├── Agent vs Chatbot             ├── Full LangChain deep dive
  ├── ReAct Loop                   ├── Custom tool creation
  ├── Tools                        ├── Persistent memory with ChromaDB
  ├── Memory types                 ├── Multi-step reasoning chains
  ├── Planning                     ├── Personal AI Assistant Agent:
  ├── Multi-agent intro            │     answers questions ✓
  └── Research Agent ✅            │     searches the web ✓
                                   │     writes summaries ✓
                                   └──   saves notes to disk ✓
```

In **Phase 3** you'll move from understanding agents to building a **production-quality Personal AI Assistant** with LangChain — with persistent memory, custom tools, and a Streamlit UI. This is the one you'll show in job interviews.

---

**Complete the 3-task coding challenge above, then say "ready for Phase 3" when you're done!** 💪

If you get stuck on any step — especially the BigQuery tool since that's your home turf — just paste your code and I'll debug it with you.