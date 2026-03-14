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
