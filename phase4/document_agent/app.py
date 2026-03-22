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