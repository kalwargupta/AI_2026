# agent.py
# ─────────────────────────────────────────────────────────────
# Document AI Agent — assembles all components
# ─────────────────────────────────────────────────────────────

from langchain_ollama import ChatOllama
from langchain.agents import AgentExecutor, create_tool_calling_agent
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.memory import ConversationBufferWindowMemory
from langchain_community.vectorstores import FAISS

import tools as T
from tools import ALL_TOOLS, set_vectorstore
from rag import index_documents
from config import (LLM_MODEL, LLM_TEMPERATURE, DOCS_DIR,
                    MAX_ITERATIONS, MAX_EXEC_TIME)


SYSTEM_PROMPT = """You are an expert Document AI Assistant.

Your primary job is to answer questions based on indexed PDF documents.
You also have web search for general knowledge questions.

TOOL PRIORITY ORDER (follow this strictly):
1. search_documents — ALWAYS try this first for any document question
2. list_indexed_documents — use when asked what files are available
3. summarize_section — use to condense long retrieved passages
4. search_web — use only for info NOT in the documents
5. calculate — use for ALL math (never calculate in your head)
6. get_datetime — use for date/time questions

ANSWER FORMAT:
- Always cite your source: "According to [filename], page X..."
- If search_documents returns nothing relevant, say so and try search_web
- Be concise. Use bullet points for lists.
- Include page numbers in citations whenever available.

You are a trustworthy assistant — never make up facts not in the documents."""


def build_agent(vectorstore: FAISS = None) -> AgentExecutor:
    """Build and return the Document AI AgentExecutor."""

    # Inject the vectorstore into the tools module
    # (tools.py uses a module-level variable for the vectorstore)
    if vectorstore:
        set_vectorstore(vectorstore)

    llm = ChatOllama(model=LLM_MODEL, temperature=LLM_TEMPERATURE)

    prompt = ChatPromptTemplate.from_messages([
        ("system", SYSTEM_PROMPT),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}"),
        MessagesPlaceholder("agent_scratchpad"),
    ])

    memory = ConversationBufferWindowMemory(
        k=8,
        memory_key="chat_history",
        return_messages=True,
    )

    agent = create_tool_calling_agent(llm=llm, tools=ALL_TOOLS, prompt=prompt)

    return AgentExecutor(
        agent=agent,
        tools=ALL_TOOLS,
        memory=memory,
        verbose=True,
        max_iterations=MAX_ITERATIONS,
        max_execution_time=MAX_EXEC_TIME,
        handle_parsing_errors=True,
        early_stopping_method="generate",
        return_intermediate_steps=True,
    )


def chat(executor: AgentExecutor, user_input: str) -> dict:
    """Send one message, return output + steps."""
    result = executor.invoke({"input": user_input})
    return {
        "output": result.get("output", "No response."),
        "steps":  result.get("intermediate_steps", []),
    }