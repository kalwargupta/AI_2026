Welcome to your **AI Agents Mastery Journey**! 🚀 As a GCP Data Engineer, you already have a massive advantage — you understand data pipelines, APIs, and cloud infrastructure. We're going to build on that foundation.

Let me first give you your **Interactive Roadmap Dashboard**, then we'll dive deep into **Phase 1**.

```jsx
// AI Agents Learning Roadmap - Interactive Dashboard
```Your **Interactive Roadmap Dashboard** is live above — click topics to track your progress across all 7 phases! Now let's get into **the actual teaching**. Buckle up. 🚀

---

# 🧠 PHASE 1 — AI Foundations

> **Your GCP advantage:** You already understand APIs, data pipelines, and cloud infrastructure. We'll constantly connect new AI concepts to things you already know.

---

## LESSON 1.1 — What is AI?

### 📖 What it is
AI (Artificial Intelligence) is software that **mimics human thinking** to make decisions, recognize patterns, and solve problems.

Think of it this way:
```
Traditional Program:  IF temperature > 100 THEN alert("Too Hot")
                      (You write every rule manually)

AI Program:           You show it 1 million examples of "hot" vs "not hot"
                      It LEARNS the rules itself — including ones you didn't think of
```

### 🔌 GCP Connection You Already Know
```
BigQuery SQL Query   →   You write exact logic to get answers
BigQuery ML          →   You give it data, it learns the logic for you
                         ↑ That's AI!
```

### 🌍 Real-World Examples
| Traditional Code | AI Version |
|---|---|
| `if email contains "sale" → spam` | AI reads 1M emails, learns spam patterns itself |
| `if speed > 100 → brake` | AI sees 10M driving examples, learns when to brake |
| `if stock drops 5% → sell` | AI reads news + data, predicts market moves |

---

## LESSON 1.2 — What is Generative AI?

### 📖 What it is
Regular AI **classifies** or **predicts** (Is this spam? Yes/No).
**Generative AI** actually **creates new content** — text, images, code, audio.

```
Regular AI:    Input: Photo of cat → Output: "cat" (a label)
Generative AI: Input: "Write a poem about a cat" → Output: [entire poem]
```

### The 3 Types of Gen AI You Need to Know
```
┌─────────────────────────────────────────────────────┐
│                  GENERATIVE AI                       │
│                                                      │
│  📝 TEXT      🖼️ IMAGE       🎵 AUDIO               │
│  GPT-4        DALL-E         Suno                    │
│  Claude       Midjourney     ElevenLabs              │
│  Llama        Stable Diff.   Whisper                 │
│                                                      │
│  ← WE FOCUS HERE FOR AI AGENTS                       │
└─────────────────────────────────────────────────────┘
```

---

## LESSON 1.3 — What is an LLM? (Large Language Model)

### 📖 What it is
An LLM is a **massive AI model trained on billions of text documents** (books, websites, code, research papers). It learns to **predict the next word** — and from that simple task, it becomes surprisingly intelligent.

### 🧠 Simple Analogy
```
You:  "The sky is ___"
You (as a human): "blue" — because you've read/heard this millions of times

LLM:  "The sky is ___"  
LLM:  "blue" — because it read 500 billion words and saw this pattern billions of times
```

### 🔍 How LLMs Actually Work (Simplified)
```
TRAINING PHASE (done once, costs $millions):
  Billions of text documents
          ↓
  Model reads: "Paris is the capital of ___"
  Model guesses: "London" ❌
  Model gets corrected: "France" ✅
  Model updates its weights (numbers inside it)
          ↓
  Repeat 500 BILLION times
          ↓
  Model now "understands" language

INFERENCE PHASE (what you use every day):
  Your question → LLM → Answer
```

### Popular LLMs (Free vs Paid)
```
FREE (Self-hosted via Ollama):       PAID API:
  🦙 Llama 3 (Meta)                   GPT-4o (OpenAI)
  🔮 Mistral 7B                        Claude 3.5 (Anthropic)  
  💎 Gemma 2 (Google)                  Gemini Pro (Google)
  🌊 Phi-3 (Microsoft)
  
  ← We start here for free!
```

---

## LESSON 1.4 — Tokens & Embeddings

### 📖 Tokens — What it is
LLMs don't read words — they read **tokens**. A token is roughly **¾ of a word**.

```python
# How text becomes tokens:
"Hello, how are you?" 
→ ["Hello", ",", " how", " are", " you", "?"]
→ [15496, 11, 703, 389, 345, 30]  ← These numbers are what the LLM sees!

# Why this matters for you (Cost & Limits):
# GPT-4: $0.03 per 1000 tokens
# Llama 3: FREE (running locally)
# Context window = max tokens per conversation
```

### 🧲 Embeddings — What it is
Embeddings are **numbers that represent meaning**. Similar meanings → similar numbers → close in space.

```
WORD            EMBEDDING (simplified)
"King"       →  [0.2, 0.9, 0.1, 0.8, ...]  (512 numbers)
"Queen"      →  [0.2, 0.9, 0.1, 0.7, ...]  (very similar!)
"Apple"      →  [0.8, 0.1, 0.9, 0.2, ...]  (very different)

FAMOUS EXAMPLE:
  King - Man + Woman ≈ Queen
  (The math of meaning!)
```

### 🔌 GCP Connection
```
Your BigQuery data:   Stored as rows/columns (structured)
Embeddings:           Stored as vectors (numbers representing meaning)
                      Used in Vector Databases (like a BigQuery for AI meaning)
```

---

## LESSON 1.5 — Vector Databases

### 📖 What it is
A Vector Database stores **embeddings** and lets you search by **semantic meaning**, not exact keywords.

```
TRADITIONAL SEARCH (BigQuery, SQL):
  Query: "car"
  Finds: Documents containing exactly "car"
  Misses: Documents about "automobile", "vehicle", "Tesla"

VECTOR SEARCH:
  Query: "car"  →  Converted to embedding [0.2, 0.8, ...]
  Finds: Everything semantically similar — "automobile" ✅ "vehicle" ✅ "Tesla" ✅
  This is how AI agents "remember" and find relevant info!
```

### Architecture Diagram
```
┌──────────────────────────────────────────────────────┐
│                VECTOR DATABASE FLOW                   │
│                                                      │
│  Your Documents (PDFs, notes, SQL docs)              │
│          ↓ Embedding Model converts to vectors        │
│  [0.2, 0.9, 0.1...] [0.8, 0.2, 0.7...]              │
│          ↓ Stored in Vector DB (ChromaDB/FAISS)       │
│                                                      │
│  User Question: "How do I optimize BigQuery?"        │
│          ↓ Converted to vector                        │
│  Search Vector DB for closest vectors                │
│          ↓                                           │
│  Returns: Most relevant document chunks              │
│          ↓ LLM uses these to answer                  │
│  Perfect Answer! 🎯                                  │
└──────────────────────────────────────────────────────┘
```

### Popular Vector DBs
```
FREE (Local):          FREE (Cloud):        Paid:
  FAISS (Facebook)       ChromaDB             Pinecone
  ChromaDB               Weaviate free        Weaviate
  SQLite-VSS             Qdrant free          Qdrant
```

---

## LESSON 1.6 — Prompt Engineering

### 📖 What it is
Prompt Engineering is the art of **writing instructions to LLMs** to get the best outputs. It's like writing a perfect SQL query — garbage in, garbage out.

### The 5 Core Techniques
```python
# 1. ZERO-SHOT (Just ask)
prompt = "Translate 'Hello' to French"
# Output: "Bonjour"

# 2. FEW-SHOT (Give examples)
prompt = """
Translate English to French:
  English: Hello → French: Bonjour
  English: Thank you → French: Merci  
  English: Goodbye → French: ???
"""
# Output: "Au revoir"  ← Better with examples!

# 3. CHAIN-OF-THOUGHT (Make it think step by step)
prompt = """
Solve this step by step:
  A BigQuery job costs $5 per TB. If I scan 2.3 TB, what do I pay?
  Think step by step.
"""
# Output: "Step 1: 2.3 TB × $5 = $11.50"  ← More accurate!

# 4. ROLE PROMPTING (Give it a persona)
prompt = """
You are an expert GCP Data Engineer with 10 years experience.
Review this BigQuery SQL and suggest optimizations:
  SELECT * FROM huge_table WHERE date > '2020-01-01'
"""
# Output: Much better, specific advice!

# 5. STRUCTURED OUTPUT (Get JSON back)
prompt = """
Extract entities from this text and return as JSON:
  Text: "John works at Google in New York"
  Return: {"person": "...", "company": "...", "city": "..."}
"""
```

---

# 🛠️ HANDS-ON: Mini Project 1

## Build a Python Chatbot with Ollama (100% Free, Local)

### Step 1: Install Ollama
```bash
# Mac/Linux:
curl -fsSL https://ollama.ai/install.sh | sh

# Windows: Download from https://ollama.ai/download

# Pull a free LLM (Llama 3.2 - 2GB):
ollama pull llama3.2

# Test it works:
ollama run llama3.2 "Hello, who are you?"
```

### Step 2: Folder Structure
```
ai-agents-course/
│
├── phase1/
│   ├── chatbot/
│   │   ├── chatbot.py          ← Main chatbot file
│   │   ├── requirements.txt    ← Dependencies
│   │   └── README.md           ← How to run
```

### Step 3: Complete Code — `chatbot.py`
```python
# ============================================================
# PHASE 1 MINI PROJECT: Simple AI Chatbot
# Using: Ollama (FREE, Local LLM) + Python
# ============================================================

# Import the Ollama library to talk to local LLMs
import ollama

# Import datetime to show timestamps
from datetime import datetime


def create_system_prompt():
    """
    A system prompt tells the LLM WHO IT IS and HOW TO BEHAVE.
    Think of it like your Airflow DAG config - it sets the context.
    """
    return """
    You are a helpful AI assistant that specializes in:
    - Data Engineering (BigQuery, Airflow, Spark, GCP)
    - Python programming  
    - General knowledge questions
    
    Always be concise, accurate, and friendly.
    When discussing code, always explain what it does.
    """


def chat_with_ai(conversation_history: list, user_message: str) -> str:
    """
    Send a message to the LLM and get a response.
    
    Args:
        conversation_history: List of previous messages (this is MEMORY!)
        user_message: What the user just typed
    
    Returns:
        The AI's response as a string
    """
    
    # Add the user's new message to conversation history
    # This is how LLMs remember what was said earlier
    conversation_history.append({
        "role": "user",           # "user" = the human speaking
        "content": user_message   # The actual message text
    })
    
    # Call the Ollama API with the FULL conversation history
    # The LLM needs all previous messages to understand context
    response = ollama.chat(
        model="llama3.2",           # Which LLM to use (we downloaded this)
        messages=conversation_history  # Full conversation so far
    )
    
    # Extract just the text response from the API response object
    ai_response = response["message"]["content"]
    
    # Add the AI's response to history so next message has full context
    conversation_history.append({
        "role": "assistant",    # "assistant" = the AI speaking
        "content": ai_response  # What the AI said
    })
    
    return ai_response


def display_welcome():
    """Show a nice welcome message when the chatbot starts."""
    print("\n" + "="*60)
    print("🤖 AI CHATBOT - Phase 1 Mini Project")
    print("🧠 Powered by: Llama 3.2 (Running Locally - FREE!)")
    print("💡 Type 'quit' or 'exit' to stop")
    print("💡 Type 'history' to see conversation")
    print("💡 Type 'clear' to start fresh")
    print("="*60 + "\n")


def main():
    """
    Main function - this is where the chatbot runs.
    
    Key concept: We maintain 'conversation_history' as a list.
    Each message = {"role": "user/assistant", "content": "text"}
    The LLM reads ALL history on each request - that's how it remembers!
    """
    
    display_welcome()
    
    # Initialize conversation with system instructions
    # System message tells the LLM its role (like a Prompt Template)
    conversation_history = [
        {
            "role": "system",           # Special role for instructions
            "content": create_system_prompt()
        }
    ]
    
    print("You can start chatting now. Try asking about BigQuery or Python!\n")
    
    # Main chat loop - keeps running until user says quit
    while True:
        # Get user input
        user_input = input("You: ").strip()
        
        # Check for special commands
        if not user_input:
            continue  # Skip empty messages
            
        if user_input.lower() in ["quit", "exit", "bye"]:
            print("\n🤖 Goodbye! Keep learning AI Agents!")
            break
            
        if user_input.lower() == "history":
            # Show all messages in conversation so far
            print("\n--- CONVERSATION HISTORY ---")
            for msg in conversation_history:
                if msg["role"] != "system":  # Skip system prompt
                    print(f"{msg['role'].upper()}: {msg['content'][:100]}...")
            print("----------------------------\n")
            continue
            
        if user_input.lower() == "clear":
            # Reset conversation (loses all memory!)
            conversation_history = [conversation_history[0]]  # Keep system prompt
            print("🗑️  Conversation cleared. Starting fresh!\n")
            continue
        
        # Show typing indicator
        print(f"\n🤖 AI ({datetime.now().strftime('%H:%M:%S')}): ", end="", flush=True)
        
        try:
            # Get response from AI
            # This calls Ollama which runs Llama 3.2 on your machine
            response = chat_with_ai(conversation_history, user_input)
            print(response)
            
        except Exception as e:
            # Error handling - important in any production system!
            print(f"❌ Error: {e}")
            print("Make sure Ollama is running: `ollama serve`")
        
        print()  # Empty line for readability


# This is the Python entry point
# Only runs if you execute this file directly (not when imported)
if __name__ == "__main__":
    main()
```

### Step 4: Requirements File — `requirements.txt`
```text
ollama==0.3.3
```

### Step 5: How to Run
```bash
# 1. Create your project folder
mkdir -p ai-agents-course/phase1/chatbot
cd ai-agents-course/phase1/chatbot

# 2. Create a virtual environment (keeps dependencies clean)
python -m venv venv

# Activate it:
source venv/bin/activate     # Mac/Linux
venv\Scripts\activate        # Windows

# 3. Install dependencies
pip install ollama

# 4. Make sure Ollama is running (in a separate terminal):
ollama serve

# 5. Run the chatbot!
python chatbot.py
```

### Step 6: Try These Test Prompts
```
You: Explain what BigQuery partitioning is in simple terms
You: Write a Python function to read a CSV file and load it to BigQuery
You: What is the difference between Dataflow and Dataproc?
You: history
You: clear
You: What did we talk about before? (Should say: nothing! Memory was cleared)
```

---

## 🔥 BONUS: Streamlit Web UI Version

```python
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
```

```bash
# Install Streamlit and run:
pip install streamlit
streamlit run chatbot_ui.py
# Opens at: http://localhost:8501
```

---

## 🎯 YOUR CODING TASK

Before moving to Phase 2, complete this assignment:

**Task:** Extend the chatbot to add these 3 features:
1. **Save conversations to a file** — after each session, write the conversation to a `.txt` file with a timestamp filename
2. **Word count display** — after every AI response, show `[Response: X words, Y tokens estimated]`
3. **Persona switch** — add a command `!persona data-engineer` or `!persona python-expert` that changes the system prompt

```python
# HINT for saving conversations:
import json
from datetime import datetime

def save_conversation(history):
    filename = f"chat_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(filename, 'w') as f:
        json.dump(history, f, indent=2)
    print(f"✅ Saved to {filename}")

# HINT for token estimation:
def estimate_tokens(text):
    # Rough estimate: 1 token ≈ 4 characters
    return len(text) // 4
```

---

## 📝 Phase 1 Quiz

Test yourself before moving on:

**1.** What is the difference between Traditional AI and Generative AI?

**2.** In one sentence: what is an LLM?

**3.** Why do LLMs use tokens instead of words?

**4.** What makes a Vector Database different from a regular database like BigQuery?

**5.** What is a "system prompt" and what is it used for?

**6.** In your chatbot code, why do we pass the entire `conversation_history` on every API call?

**7.** Name 3 free LLMs you can run locally using Ollama.

---

## 💼 Phase 1 Interview Questions

These are **real questions** asked in AI Engineer interviews:

> "Explain the transformer architecture to a non-technical person"

> "What is a context window and why does it matter?"

> "How would you choose between FAISS and ChromaDB for a production RAG system?"

> "What is hallucination in LLMs and how do you mitigate it?"

> "Explain prompt engineering techniques you've used"

---

## 🗺️ What's Next — Phase 2 Preview

```
PHASE 1 (DONE ✅):          PHASE 2 (NEXT 🔜):
  ├── What is AI               ├── What is an AI Agent
  ├── LLMs                     ├── Agent Loop (Observe → Think → Act)
  ├── Tokens                   ├── Tools (Web search, calculators, APIs)
  ├── Embeddings               ├── Memory (Short & Long term)
  ├── Vector DBs               ├── Planning & Reasoning
  └── Chatbot ✅               └── Build a Research Agent 🔜
```

In **Phase 2** you'll learn the critical difference between a chatbot (which you just built) and an **AI Agent** — and why agents are 10× more powerful. You'll build an agent that autonomously searches the web, reasons about what it found, and writes you a research report.

---

**Complete your coding task above, then let me know when you're ready for Phase 2!** 🚀

If anything is unclear, just ask — we'll dig deeper into any concept before moving forward. Remember: your GCP background is a **superpower** here, not a handicap. Every AI concept has a parallel in the data engineering world you already know.