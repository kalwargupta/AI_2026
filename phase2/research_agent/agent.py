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
        model="llama3.1",       # The LLM we downloaded via `ollama pull llama3.2`
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
        verbose=True,
        max_iterations=15,                     # was 10
        max_execution_time=90,                 # 90 second hard timeout
        handle_parsing_errors=True,
        early_stopping_method="generate",      # graceful finish
        return_intermediate_steps=True        # useful for debugging
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
        
        #try:
        #    # Run the agent! This triggers the ReAct loop:
        #    # 1. LLM reads the task
        #    # 2. Thinks about what tool to use
        #    # 3. Calls the tool
        #    # 4. Reads the observation
        #    # 5. Thinks again
        #    # 6. Repeats until it has the answer
        #    result = agent.invoke({
        #        "input": user_input  # The user's research task
        #    })
        #    
        #    # Display the final answer nicely
        #    console.print(Panel(
        #        result["output"],           # The agent's final answer
        #        title="📋 Research Result",
        #        border_style="green"
        #    ))
        #    
        #except Exception as e:
        #    console.print(f"[red]Error: {e}[/red]")
        #    console.print("[yellow]Tip: Make sure Ollama is running: `ollama serve`[/yellow]")
        try:
            result = agent.invoke({"input": user_input})
            output = result.get("output", "").strip()
            steps  = result.get("intermediate_steps", [])
            
            # Show how many tool calls were made
            console.print(f"[dim]({len(steps)} tool calls made)[/dim]")
            
            if not output or "Agent stopped" in output:
                console.print("[yellow]⚠ Agent couldn't finish in time.[/yellow]")
                console.print("[yellow]Tip: Try a more specific question, e.g.[/yellow]")
                console.print("[yellow]  'What is LangChain in 3 sentences?'[/yellow]")
                console.print("[yellow]  instead of open-ended research tasks.[/yellow]")
            else:
                console.print(Panel(output, title="📋 Research Result", border_style="green"))

        except Exception as e:
            console.print(f"[red]Error: {e}[/red]")


# ── Entry point ──────────────────────────────────────────────
if __name__ == "__main__":
    run_research_session()