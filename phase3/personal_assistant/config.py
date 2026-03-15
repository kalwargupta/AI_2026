# config.py
# ─────────────────────────────────────────────────────
# Central config file. Change your model here and it
# updates everywhere. Like a Terraform variables file.
# ─────────────────────────────────────────────────────

# Which local LLM to use (must be pulled via `ollama pull <name>`)
LLM_MODEL = "qwen2.5"        # Best for tool-calling. Change to "llama3.1" if needed.
LLM_TEMPERATURE = 0          # 0 = precise/factual. Increase for creative tasks.

# Memory settings
MEMORY_WINDOW_SIZE = 10      # Remember last 10 conversation turns in RAM
NOTES_DB_PATH = "./notes_db" # Where ChromaDB saves notes on disk
NOTES_COLLECTION = "assistant_notes"  # ChromaDB collection name

# Embedding model for vector search (runs locally via Ollama)
EMBEDDING_MODEL = "nomic-embed-text"  # Pull with: ollama pull nomic-embed-text

# Agent settings
MAX_ITERATIONS = 12          # Max tool calls before forced finish
MAX_EXEC_TIME  = 90          # Seconds before timeout