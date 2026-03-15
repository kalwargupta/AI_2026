# memory.py
# ─────────────────────────────────────────────────────
# Two memory systems working together:
#   1. ConversationBufferWindowMemory  → short-term (RAM)
#   2. ChromaDB                        → long-term (disk)
# ─────────────────────────────────────────────────────

from langchain.memory import ConversationBufferWindowMemory
from langchain_chroma import Chroma
from langchain_ollama import OllamaEmbeddings
from config import MEMORY_WINDOW_SIZE, NOTES_DB_PATH, NOTES_COLLECTION, EMBEDDING_MODEL


def get_short_term_memory() -> ConversationBufferWindowMemory:
    """
    Short-term memory: remembers last k conversation turns.
    Lives in RAM — cleared when the program restarts.
    
    Like a Dataflow pipeline's in-memory state window.
    """
    return ConversationBufferWindowMemory(
        k=MEMORY_WINDOW_SIZE,   # Keep last 10 turns (20 messages: 10 human + 10 AI)
        memory_key="chat_history",  # Key name used in the prompt template
        return_messages=True,       # Return Message objects (not plain strings)
    )


def get_vector_store() -> Chroma:
    """
    Long-term memory: vector database that persists to disk.
    Survives restarts. Searchable by semantic meaning.
    
    Like a BigQuery table with VECTOR_SEARCH — but local and free.
    """
    # The embedding model converts text → vectors for storage and search
    # nomic-embed-text is a free, fast embedding model that runs via Ollama
    embeddings = OllamaEmbeddings(model=EMBEDDING_MODEL)
    
    # Chroma creates the ./notes_db/ folder on first run
    # On subsequent runs it loads existing notes from disk
    vectorstore = Chroma(
        collection_name=NOTES_DB_PATH,
        embedding_function=embeddings,
        persist_directory=NOTES_DB_PATH,  # Save to disk (not just RAM)
    )
    
    return vectorstore