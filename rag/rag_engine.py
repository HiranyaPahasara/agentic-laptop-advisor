"""
FAISS RAG engine for Smart Specs.

Loads laptop/workload documents from data/knowledge_base,
builds (or loads) a local FAISS index, and returns relevant chunks.
"""

from __future__ import annotations

from pathlib import Path

from langchain_community.document_loaders import DirectoryLoader, TextLoader
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

try:
    from langchain_text_splitters import RecursiveCharacterTextSplitter
except ImportError:  # older LangChain layouts
    from langchain.text_splitter import RecursiveCharacterTextSplitter

# Project paths
PROJECT_ROOT = Path(__file__).resolve().parent.parent
KNOWLEDGE_BASE_DIR = PROJECT_ROOT / "data" / "knowledge_base"
FAISS_INDEX_DIR = PROJECT_ROOT / "faiss_index"

# Small local embedding model (no API key needed)
EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"


def get_embeddings() -> HuggingFaceEmbeddings:
    """Create the embedding model used for indexing and search."""
    return HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL_NAME)


def load_documents():
    """Load all .txt documents from the knowledge base folder."""
    if not KNOWLEDGE_BASE_DIR.exists():
        raise FileNotFoundError(f"Knowledge base not found: {KNOWLEDGE_BASE_DIR}")

    loader = DirectoryLoader(
        str(KNOWLEDGE_BASE_DIR),
        glob="*.txt",
        loader_cls=TextLoader,
        loader_kwargs={"encoding": "utf-8"},
        show_progress=False,
    )
    documents = loader.load()
    if not documents:
        raise ValueError("No .txt documents found in knowledge base.")
    return documents


def split_documents(documents):
    """Split long documents into smaller chunks for better retrieval."""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=80,
        length_function=len,
    )
    return splitter.split_documents(documents)


def build_faiss_index(save: bool = True) -> FAISS:
    """Build a new FAISS index from the knowledge base and optionally save it."""
    documents = load_documents()
    chunks = split_documents(documents)
    embeddings = get_embeddings()
    vectorstore = FAISS.from_documents(chunks, embeddings)

    if save:
        FAISS_INDEX_DIR.mkdir(parents=True, exist_ok=True)
        vectorstore.save_local(str(FAISS_INDEX_DIR))

    return vectorstore


def load_faiss_index() -> FAISS:
    """Load an existing FAISS index from disk."""
    if not FAISS_INDEX_DIR.exists():
        raise FileNotFoundError(
            f"FAISS index not found at {FAISS_INDEX_DIR}. Build it first."
        )

    embeddings = get_embeddings()
    return FAISS.load_local(
        str(FAISS_INDEX_DIR),
        embeddings,
        allow_dangerous_deserialization=True,
    )


def get_vectorstore(rebuild: bool = False) -> FAISS:
    """
    Return a ready FAISS vectorstore.
    Rebuilds the index if rebuild=True or if no saved index exists.
    """
    if rebuild or not FAISS_INDEX_DIR.exists():
        return build_faiss_index(save=True)
    return load_faiss_index()


def retrieve_context(query: str, k: int = 4, rebuild: bool = False) -> str:
    """
    Retrieve top-k relevant chunks for a user query and join them as context text.
    This context is what Agent 2 will use instead of guessing specs.
    """
    vectorstore = get_vectorstore(rebuild=rebuild)
    docs = vectorstore.similarity_search(query, k=k)

    if not docs:
        return "No relevant laptop documents were found."

    parts = []
    for i, doc in enumerate(docs, start=1):
        source = Path(doc.metadata.get("source", "unknown")).name
        parts.append(f"[{i}] Source: {source}\n{doc.page_content}")

    return "\n\n".join(parts)


if __name__ == "__main__":
    # Quick manual test: builds index (first run) and prints retrieved chunks
    sample_query = "student laptop 16GB RAM under budget for coding"
    print("Building/loading FAISS index and retrieving context...\n")
    context = retrieve_context(sample_query, k=3, rebuild=True)
    print(context)
