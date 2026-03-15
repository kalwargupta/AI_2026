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