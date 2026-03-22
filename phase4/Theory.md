Explore all 4 tabs — especially the **Live Chunker** where you can paste any text and tune chunk size in real time. Now the full lesson:

---

# 🚀 PHASE 4 — Advanced Agents

> **Goal:** Build a Document AI Agent that reads any PDF and answers questions from it — with source citations. This is the #1 most-asked project in AI Engineer interviews.

---

## LESSON 4.1 — RAG (Retrieval-Augmented Generation)

RAG solves the biggest LLM limitation: **LLMs don't know your private documents**.

```
WITHOUT RAG:
  You: "What does our internal BigQuery style guide say about partitioning?"
  LLM: "I don't have access to your internal documents." ← useless

WITH RAG:
  You:   "What does our internal BigQuery style guide say about partitioning?"
  Agent: Searches your PDF index → finds 3 relevant chunks → gives them to LLM
  LLM:   "According to your style guide (page 12): always partition by
          DATE(_PARTITIONTIME) and cluster by the most-filtered column..."
          ← grounded in YOUR document, with source citation!
```

The key insight: **the LLM doesn't need to memorize your docs — it just needs to read the right chunk at query time.** This is exactly how your brain works when you look something up in a textbook rather than memorising every page.

---

## LESSON 4.2 — ReAct Agent (Deep Dive)

You've used ReAct already, but let's understand exactly what the LLM is generating. Here's the raw text a ReAct agent produces and how LangChain parses it:

```
# This is the ACTUAL TEXT the LLM writes (you can see it with verbose=True):

Thought: The user is asking about BigQuery partitioning cost savings.
         I should search the indexed PDF for relevant chunks.

Action: search_documents
Action Input: BigQuery partitioning cost reduction

Observation: [Chunk 3, page 12] "Partitioning tables by date reduces scanned
             data by up to 90% because BigQuery only reads partitions that
             match the WHERE clause filter..."

Thought: I found a relevant passage. I have enough to answer with a citation.

Final Answer: According to your document (page 12), partitioning tables by
              date can reduce scan costs by up to 90%...
```

LangChain parses each `Action:` line to know which tool to call, and each `Action Input:` line as the argument. That's why the format must be exact — it's literally a parser reading the LLM's text output.

---

## Full Project Structure

```
ai-agents-course/
└── phase4/
    └── document_agent/
        ├── app.py              ← Streamlit web UI
        ├── agent.py            ← Agent with RAG tool
        ├── rag.py              ← PDF indexing + retrieval
        ├── tools.py            ← All agent tools
        ├── config.py           ← Settings
        ├── requirements.txt
        ├── sample_docs/        ← Put your PDFs here
        │   └── .gitkeep
        └── vector_store/       ← FAISS index (auto-created)
```

---

## Step 1 — Install

```bash
mkdir -p ai-agents-course/phase4/document_agent/sample_docs
cd ai-agents-course/phase4/document_agent
python -m venv venv && source venv/bin/activate

pip install langchain langchain-community langchain-ollama \
            langchain-chroma faiss-cpu chromadb \
            pypdf pymupdf \
            duckduckgo-search requests beautifulsoup4 \
            streamlit python-dotenv rich

# Models
ollama pull qwen2.5            # best for tool-calling
ollama pull nomic-embed-text   # embedding model for RAG
```

---

## Step 2 — `config.py`

```python
# config.py
# Central settings — change model or paths here only

LLM_MODEL        = "qwen2.5"          # or "llama3.1", "mistral"
EMBED_MODEL      = "nomic-embed-text"  # local embedding model
LLM_TEMPERATURE  = 0

# RAG settings
CHUNK_SIZE       = 500    # tokens per chunk (~2000 characters)
CHUNK_OVERLAP    = 50     # overlap between chunks to preserve context
TOP_K_RESULTS    = 4      # how many chunks to retrieve per question

# Paths
VECTOR_STORE_DIR = "./vector_store"   # FAISS index lives here
DOCS_DIR         = "./sample_docs"    # drop your PDFs here

# Agent limits
MAX_ITERATIONS   = 12
MAX_EXEC_TIME    = 90
```

---

## Step 3 — `rag.py` — The RAG Engine

This is the heart of Phase 4. Every line is commented:

```python
# rag.py
# ─────────────────────────────────────────────────────────────
# RAG Engine: loads PDFs → chunks → embeds → stores → retrieves
#
# Think of this as your ETL pipeline where:
#   Extract  = load PDF pages
#   Transform = chunk + embed
#   Load     = store in FAISS
#   Query    = similarity_search (like a BigQuery SELECT)
# ─────────────────────────────────────────────────────────────

import os
from pathlib import Path
from typing import List, Optional

from langchain_community.document_loaders import PyPDFLoader, DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_ollama import OllamaEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.schema import Document

from config import (CHUNK_SIZE, CHUNK_OVERLAP, TOP_K_RESULTS,
                    VECTOR_STORE_DIR, EMBED_MODEL)


def get_embeddings() -> OllamaEmbeddings:
    """
    Returns the embedding model.
    This converts text → vectors (lists of floats).
    We must use the SAME model for both indexing AND querying.
    Using different models = vectors in different spaces = wrong results.
    """
    return OllamaEmbeddings(model=EMBED_MODEL)


def load_pdf(pdf_path: str) -> List[Document]:
    """
    Load a single PDF file and return a list of Document objects.
    Each Document = one page of the PDF with metadata.

    LangChain Document has two fields:
      .page_content  — the text of that page
      .metadata      — dict with source, page number, etc.
    """
    loader = PyPDFLoader(pdf_path)   # pypdf under the hood
    pages  = loader.load()           # returns List[Document]

    # Add the filename to metadata so we can cite sources later
    filename = Path(pdf_path).name
    for page in pages:
        page.metadata["source_file"] = filename
        # page.metadata["page"] is already set by PyPDFLoader (0-indexed)

    print(f"  Loaded '{filename}': {len(pages)} pages")
    return pages


def load_all_pdfs(docs_dir: str) -> List[Document]:
    """
    Load ALL PDFs from a directory.
    Like a glob + read in a Dataflow pipeline.
    """
    pdf_files = list(Path(docs_dir).glob("*.pdf"))

    if not pdf_files:
        print(f"No PDFs found in {docs_dir}/")
        print("  → Add PDF files to the sample_docs/ folder to get started")
        return []

    all_docs = []
    for pdf_path in pdf_files:
        docs = load_pdf(str(pdf_path))
        all_docs.extend(docs)

    print(f"  Total pages loaded: {len(all_docs)}")
    return all_docs


def chunk_documents(documents: List[Document]) -> List[Document]:
    """
    Split documents into smaller chunks for embedding.

    Why split?
      - LLMs have context limits (can't read 50 pages at once)
      - Smaller chunks = more precise retrieval
      - We only send the RELEVANT chunks to the LLM, not the whole doc

    RecursiveCharacterTextSplitter tries to split on:
      1. Paragraphs (\n\n)  ← preferred — keeps related text together
      2. Sentences (\n)
      3. Words (' ')
      4. Characters ('')    ← last resort
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,       # target size in characters (not tokens)
        chunk_overlap=CHUNK_OVERLAP, # overlap prevents losing context at boundaries
        length_function=len,         # use character count (simple, fast)
        separators=["\n\n", "\n", ". ", " ", ""],  # try these splits in order
    )

    chunks = splitter.split_documents(documents)

    # Add chunk index to metadata — useful for debugging
    for i, chunk in enumerate(chunks):
        chunk.metadata["chunk_id"] = i
        # Human-readable page reference (1-indexed)
        chunk.metadata["page_label"] = f"Page {chunk.metadata.get('page', 0) + 1}"

    print(f"  Split into {len(chunks)} chunks "
          f"(avg {sum(len(c.page_content) for c in chunks)//len(chunks)} chars each)")
    return chunks


def build_vector_store(chunks: List[Document]) -> FAISS:
    """
    Embed all chunks and store them in a FAISS index.

    This is the most time-consuming step — each chunk makes one
    API call to the embedding model. With 200 chunks at ~50ms each
    that's ~10 seconds. We save to disk so we only do this ONCE.

    FAISS index structure:
      - Flat index: exact search (accurate, slower for huge sets)
      - IVF index:  approximate search (faster, slight accuracy loss)
      We use Flat (default) — fine up to ~100k chunks.
    """
    print("  Embedding chunks (this may take 30-60 seconds)...")
    embeddings = get_embeddings()

    # FAISS.from_documents does 3 things:
    #   1. Calls embeddings.embed_documents() on every chunk
    #   2. Creates the FAISS index
    #   3. Stores (vector, document) pairs
    vectorstore = FAISS.from_documents(
        documents=chunks,
        embedding=embeddings,
    )

    # Persist to disk — so we don't re-embed on every run
    # Like writing a Spark DataFrame to Parquet
    os.makedirs(VECTOR_STORE_DIR, exist_ok=True)
    vectorstore.save_local(VECTOR_STORE_DIR)
    print(f"  Vector store saved to {VECTOR_STORE_DIR}/")

    return vectorstore


def load_vector_store() -> Optional[FAISS]:
    """
    Load an existing FAISS index from disk.
    Returns None if no index exists yet.
    """
    index_path = Path(VECTOR_STORE_DIR) / "index.faiss"

    if not index_path.exists():
        return None

    embeddings = get_embeddings()
    vectorstore = FAISS.load_local(
        VECTOR_STORE_DIR,
        embeddings,
        allow_dangerous_deserialization=True,  # required flag for local files
    )
    print(f"  Loaded existing vector store from {VECTOR_STORE_DIR}/")
    return vectorstore


def index_documents(docs_dir: str, force_rebuild: bool = False) -> Optional[FAISS]:
    """
    Main indexing function. Orchestrates the full ETL pipeline:
      Load PDFs → Chunk → Embed → Store

    Args:
        docs_dir:      directory containing PDF files
        force_rebuild: if True, re-index even if index exists

    Returns:
        FAISS vectorstore (or None if no PDFs found)
    """
    print("\n📚 Starting document indexing...")

    # Try loading existing index first (saves time on restarts)
    if not force_rebuild:
        vectorstore = load_vector_store()
        if vectorstore:
            return vectorstore

    # No existing index — build from scratch
    print("  Building new index...")
    documents = load_all_pdfs(docs_dir)

    if not documents:
        return None

    chunks     = chunk_documents(documents)
    vectorstore = build_vector_store(chunks)

    print("✅ Indexing complete!\n")
    return vectorstore


def retrieve_relevant_chunks(
        vectorstore: FAISS,
        query: str,
        k: int = TOP_K_RESULTS
) -> str:
    """
    Retrieve the top-k most relevant chunks for a query.
    This is called by the agent's RAG tool on every question.

    Returns formatted string with content + source citations.
    The agent passes this string directly to the LLM as context.
    """
    # similarity_search_with_score returns (Document, score) pairs
    # score = L2 distance (lower = more similar for FAISS)
    results = vectorstore.similarity_search_with_score(query, k=k)

    if not results:
        return "No relevant content found in the indexed documents."

    # Format results as context for the LLM
    formatted_chunks = []
    for i, (doc, score) in enumerate(results, 1):
        source   = doc.metadata.get("source_file", "Unknown")
        page     = doc.metadata.get("page_label", "Unknown page")
        # Convert L2 distance to a rough relevance % (lower distance = higher relevance)
        relevance = max(0, round((1 - score / 2) * 100, 1))

        formatted_chunks.append(
            f"[Source {i}: {source}, {page} | Relevance: {relevance}%]\n"
            f"{doc.page_content.strip()}"
        )

    return "\n\n---\n\n".join(formatted_chunks)
```

---

## Step 4 — `tools.py`

```python
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
```

---

## Step 5 — `agent.py`

```python
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
```

---

## Step 6 — `app.py` — Streamlit UI

```python
# app.py — run with: streamlit run app.py
# ─────────────────────────────────────────────────────────────

import streamlit as st
import os, shutil
from pathlib import Path
from agent import build_agent, chat
from rag import index_documents
from config import DOCS_DIR, VECTOR_STORE_DIR

st.set_page_config(
    page_title="Document AI Agent",
    page_icon="📄",
    layout="wide",
)

# ── Session state init ────────────────────────────────────────
for key, default in [
    ("agent",    None),
    ("messages", []),
    ("indexed_files", []),
    ("tool_log", []),
]:
    if key not in st.session_state:
        st.session_state[key] = default


def initialize_agent(force_rebuild=False):
    """Index documents and build the agent."""
    with st.spinner("📚 Indexing documents..."):
        vectorstore = index_documents(DOCS_DIR, force_rebuild=force_rebuild)
    with st.spinner("🤖 Loading AI Agent..."):
        st.session_state.agent = build_agent(vectorstore)
    # Track which files are indexed
    st.session_state.indexed_files = [
        f.name for f in Path(DOCS_DIR).glob("*.pdf")
    ]


# ── Sidebar ───────────────────────────────────────────────────
with st.sidebar:
    st.title("📄 Document AI Agent")
    st.caption("Phase 4 · RAG · FAISS · Local LLM")
    st.divider()

    # ── File uploader ─────────────────────────────────────────
    st.subheader("📁 Upload Documents")
    uploaded = st.file_uploader(
        "Upload PDF files",
        type=["pdf"],
        accept_multiple_files=True,
        help="PDFs will be indexed for question answering"
    )

    if uploaded:
        os.makedirs(DOCS_DIR, exist_ok=True)
        new_files = []
        for f in uploaded:
            dest = os.path.join(DOCS_DIR, f.name)
            with open(dest, "wb") as out:
                out.write(f.read())
            new_files.append(f.name)
        st.success(f"Saved: {', '.join(new_files)}")

    # ── Index button ──────────────────────────────────────────
    col1, col2 = st.columns(2)
    with col1:
        if st.button("🔍 Index docs", use_container_width=True):
            initialize_agent(force_rebuild=True)
            st.success("Indexed!")
            st.rerun()
    with col2:
        if st.button("🗑️ Clear index", use_container_width=True):
            if Path(VECTOR_STORE_DIR).exists():
                shutil.rmtree(VECTOR_STORE_DIR)
            st.session_state.agent = None
            st.session_state.indexed_files = []
            st.warning("Index cleared.")
            st.rerun()

    # ── Currently indexed ─────────────────────────────────────
    st.divider()
    st.subheader("📑 Indexed Files")
    pdfs = list(Path(DOCS_DIR).glob("*.pdf")) if Path(DOCS_DIR).exists() else []
    if pdfs:
        for p in pdfs:
            st.markdown(f"✅ `{p.name}`")
    else:
        st.caption("No PDFs yet — upload above")

    # ── Quick prompts ─────────────────────────────────────────
    st.divider()
    st.subheader("💡 Try asking")
    quick_prompts = [
        "What documents do you have indexed?",
        "Summarize the main topics in the document",
        "What does the document say about [topic]?",
        "List all key points from page 1",
        "Find anything related to cost or pricing",
    ]
    for p in quick_prompts:
        if st.button(p, use_container_width=True):
            st.session_state.quick_input = p

    # ── Tool log ──────────────────────────────────────────────
    st.divider()
    if st.session_state.tool_log:
        st.subheader("🔧 Tool calls")
        for entry in st.session_state.tool_log[-4:]:
            with st.expander(f"→ {entry['tool']}"):
                st.code(str(entry['input'])[:150])
                st.caption(str(entry['output'])[:300])

    st.divider()
    if st.button("🗑️ Clear chat", use_container_width=True):
        st.session_state.messages = []
        st.session_state.tool_log = []
        if st.session_state.agent:
            # Reset memory without rebuilding the whole agent
            st.session_state.agent.memory.clear()
        st.rerun()


# ── Auto-initialize if PDFs exist but agent not loaded ────────
if st.session_state.agent is None and Path(DOCS_DIR).exists():
    if list(Path(DOCS_DIR).glob("*.pdf")):
        initialize_agent()


# ── Main area ─────────────────────────────────────────────────
st.title("📄 Document AI Agent")
st.caption("Ask questions about your uploaded PDF documents")

# No documents warning
if not list(Path(DOCS_DIR).glob("*.pdf")) if Path(DOCS_DIR).exists() else True:
    st.info(
        "👆 Upload PDF files in the sidebar to get started.\n\n"
        "**Good PDFs to try:**\n"
        "- GCP documentation pages (save as PDF from Chrome)\n"
        "- Your company runbooks or style guides\n"
        "- Any technical documentation\n"
        "- Research papers"
    )

# Show chat history
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if msg.get("tools"):
            with st.expander(f"🔧 {len(msg['tools'])} tool call(s)", expanded=False):
                for t in msg["tools"]:
                    st.caption(f"**{t['tool']}** → `{str(t['input'])[:100]}`")

# Handle sidebar quick prompts
user_input = st.session_state.pop("quick_input", None)

# Chat input
typed = st.chat_input("Ask a question about your documents...")
if typed:
    user_input = typed

# Process message
if user_input:
    if st.session_state.agent is None:
        st.warning("Please upload PDFs and click 'Index docs' first!")
        st.stop()

    # Show user message
    st.session_state.messages.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.markdown(user_input)

    # Get response
    with st.chat_message("assistant"):
        with st.spinner("🔍 Searching documents..."):
            try:
                result = chat(st.session_state.agent, user_input)
                output = result["output"]
                steps  = result["steps"]

                st.markdown(output)

                # Collect tool usage
                tools_used = []
                for action, observation in steps:
                    entry = {
                        "tool":   action.tool,
                        "input":  action.tool_input,
                        "output": str(observation),
                    }
                    tools_used.append(entry)
                    st.session_state.tool_log.append(entry)

                if tools_used:
                    with st.expander(f"🔧 {len(tools_used)} tool call(s)", expanded=False):
                        for t in tools_used:
                            st.caption(f"**{t['tool']}** → `{str(t['input'])[:100]}`")

                st.session_state.messages.append({
                    "role":    "assistant",
                    "content": output,
                    "tools":   tools_used,
                })

            except Exception as e:
                err = f"⚠️ Error: {e}\n\nMake sure `ollama serve` is running."
                st.error(err)
                st.session_state.messages.append({"role": "assistant", "content": err})
```

---

## Step 7 — `requirements.txt`

```text
langchain==0.3.7
langchain-community==0.3.7
langchain-ollama==0.2.1
langchain-chroma==0.1.4
chromadb==0.5.18
faiss-cpu==1.8.0
pypdf==4.3.1
pymupdf==1.24.11
duckduckgo-search==6.3.5
requests==2.32.3
beautifulsoup4==4.12.3
streamlit==1.40.0
python-dotenv==1.0.1
rich==13.9.4
```

---

## Step 8 — Run & Test

```bash
# Terminal 1
ollama serve

# Terminal 2
cd ai-agents-course/phase4/document_agent
source venv/bin/activate
streamlit run app.py
# → http://localhost:8501
```

**Get a test PDF fast:**
```bash
# Download GCP BigQuery public docs as a test PDF:
# 1. Open https://cloud.google.com/bigquery/docs/introduction in Chrome
# 2. Press Ctrl+P → Save as PDF → save to sample_docs/bigquery_intro.pdf

# OR use Python to create a quick test PDF:
pip install fpdf2
python3 -c "
from fpdf import FPDF
pdf = FPDF()
pdf.add_page()
pdf.set_font('Helvetica', size=12)
content = [
    'BigQuery Technical Guide',
    '',
    'Chapter 1: Partitioning',
    'Partitioning divides a table into smaller segments based on a column.',
    'Date partitioning is most common in Data Engineering workflows.',
    'Partitioned tables can reduce query costs by up to 90 percent.',
    '',
    'Chapter 2: Clustering',  
    'Clustering sorts data within partitions by up to 4 columns.',
    'Use clustering on columns that appear frequently in WHERE or JOIN.',
    'Combined with partitioning, clustering maximises BigQuery efficiency.',
    '',
    'Chapter 3: Cost Optimization',
    'Use SELECT specific columns instead of SELECT star.',
    'Preview data with LIMIT before running full scans.',
    'Set a maximum bytes billed to avoid accidental large queries.',
]
for line in content:
    pdf.cell(0, 10, line, ln=True)
pdf.output('sample_docs/bigquery_guide.pdf')
print('Created sample_docs/bigquery_guide.pdf')
"
```

**Test sequence:**
```
1. Upload bigquery_guide.pdf → click "Index docs"
2. "What documents do you have indexed?"
3. "What does the document say about partitioning?"
4. "How much can partitioning reduce costs?"          ← should cite page
5. "What are the clustering recommendations?"
6. "Summarize all cost optimization tips"
7. "What is the difference between partitioning and clustering?"
8. "Search the web for the latest BigQuery pricing"   ← uses web search
```

---

## BONUS — Add Multi-PDF filtering

When you have multiple PDFs, let users filter by document:

```python
# Add to tools.py
@tool
def search_specific_document(query_and_filename: str) -> str:
    """
    Search within a SPECIFIC document only.
    Use when the user asks about a particular file.
    Input format: 'FILENAME.pdf | your search query'
    Example: 'bigquery_guide.pdf | partitioning cost savings'
    Returns: relevant passages only from that file.
    """
    if "|" not in query_and_filename:
        return "Input must be 'filename.pdf | search query'"

    filename, query = query_and_filename.split("|", 1)
    filename = filename.strip()
    query    = query.strip()

    if _vectorstore is None:
        return "No documents indexed."

    # Use FAISS filter by metadata
    results = _vectorstore.similarity_search(
        query,
        k=TOP_K_RESULTS,
        filter={"source_file": filename},  # metadata filter!
    )

    if not results:
        return f"Nothing found in '{filename}' for '{query}'"

    return "\n\n---\n\n".join(
        f"[{filename}, {doc.metadata.get('page_label','?')}]\n{doc.page_content}"
        for doc in results
    )
```

---

## 🎯 Coding Task — 3 Extensions

**Task 1 — Add a `generate_document_summary` tool** that summarizes an ENTIRE document (not just one chunk):
```python
@tool
def generate_document_summary(filename: str) -> str:
    """
    Generate a comprehensive summary of an entire indexed document.
    Use when user asks to 'summarize the whole document' or 'give me an overview of X.pdf'
    Input: the exact filename (e.g. 'bigquery_guide.pdf')
    Returns: structured summary with key topics and main points.
    """
    # Hint: retrieve ALL chunks from that file (large k value)
    # Then summarize the combined text
    # Be careful of context limits — chunk the combined text if needed
    pass
```

**Task 2 — Add source tracking** to every answer. Modify `retrieve_relevant_chunks` to also return a structured list of sources that the agent can include in its final answer as a "References" section.

**Task 3 — Your GCP superpower** — Add a tool that queries BigQuery public datasets:
```python
@tool
def query_bigquery_public(question: str) -> str:
    """
    Run a SQL query against BigQuery public datasets to answer data questions.
    Use when the user asks about public data that BigQuery hosts.
    Examples: 'How many Shakespeare works are in BigQuery?'
    Input: natural language question about public BigQuery data.
    Returns: SQL query used and the results.
    """
    # Step 1: Use LLM to generate SQL from the question
    # Step 2: Run against bigquery-public-data
    # Step 3: Return results + the SQL used
    # Hint: from google.cloud import bigquery
    pass
```

---

## 📝 Phase 4 Quiz

1. Why must you use the **same embedding model** for both indexing and querying?
2. What does `chunk_overlap=50` do and why does it matter for answers?
3. In the code, why do we call `set_vectorstore(vectorstore)` before `build_agent()`?
4. FAISS stores vectors in RAM. What happens to the index when your app restarts?
5. What is `allow_dangerous_deserialization=True` protecting against?
6. If a user asks about page 15 of a PDF but your chunk size means page 15 splits into 3 chunks — which chunk does the agent retrieve?
7. When would you choose `similarity_search_with_score` over `similarity_search`?

---

## 💼 Interview Questions

> "Explain RAG to a non-technical product manager in 2 minutes."

> "What are the failure modes of RAG and how do you debug them?"

> "How would you evaluate whether your RAG system is giving accurate answers?"

> "What's the difference between dense retrieval (embeddings) and sparse retrieval (BM25/TF-IDF)? When would you use each?"

> "How would you scale this Document AI Agent to handle 10,000 PDFs?"

---

## Phase 5 Preview

```
PHASE 4 (DONE ✅):             PHASE 5 (NEXT 🔜):
  ├── RAG pipeline               ├── CrewAI framework
  ├── FAISS + ChromaDB           ├── AutoGen conversations
  ├── PDF loading + chunking     ├── Role-based agents
  ├── Source citations           ├── Task delegation patterns
  └── Document AI Agent ✅       └── Startup Team AI System:
                                       CEO, Research, Marketing,
                                       Developer agents working
                                       together on real tasks 🔜
```

Phase 5 is where everything gets **cinematic** — you'll watch 4 AI agents argue with each other, delegate work, and produce a complete startup business plan. It's the most impressive demo you can show in an interview.

**Upload a PDF, run your agent, complete the 3 tasks, then say "ready for Phase 5"!** 🚀