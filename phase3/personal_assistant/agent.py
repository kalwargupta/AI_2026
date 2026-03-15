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