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