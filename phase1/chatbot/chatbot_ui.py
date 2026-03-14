# chatbot_ui.py - A beautiful web interface for your chatbot
# Run with: streamlit run chatbot_ui.py

import streamlit as st   # Creates web UIs with pure Python
import ollama

# Page configuration
st.set_page_config(
    page_title="AI Chatbot - Phase 1",
    page_icon="🤖",
    layout="wide"
)

st.title("🤖 AI Chatbot")
st.caption("Powered by Llama 3.2 (Local & Free!)")

# Initialize session state (Streamlit's way of keeping memory between reruns)
if "messages" not in st.session_state:
    st.session_state.messages = [
        {
            "role": "system",
            "content": "You are a helpful Data Engineering assistant."
        }
    ]

# Display all previous messages (skip system prompt)
for msg in st.session_state.messages:
    if msg["role"] != "system":
        with st.chat_message(msg["role"]):  # Creates chat bubbles!
            st.write(msg["content"])

# Chat input box at the bottom
if prompt := st.chat_input("Ask me anything about Data Engineering..."):
    
    # Show user message
    with st.chat_message("user"):
        st.write(prompt)
    
    # Add to history
    st.session_state.messages.append({"role": "user", "content": prompt})
    
    # Get AI response with loading spinner
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            response = ollama.chat(
                model="llama3.2",
                messages=st.session_state.messages
            )
            ai_text = response["message"]["content"]
            st.write(ai_text)
    
    # Save response to history
    st.session_state.messages.append({"role": "assistant", "content": ai_text})