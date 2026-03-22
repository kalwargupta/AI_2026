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