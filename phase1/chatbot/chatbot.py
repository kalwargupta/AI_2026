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