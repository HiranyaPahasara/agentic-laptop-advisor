"""
FAISS RAG engine for Smart Specs.

Loads laptop/workload documents from data/knowledge_base,
builds (or loads) a local FAISS index, and returns relevant chunks.
"""

from __future__ import annotations

import re
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

def _extract_price(text: str) -> float | None:
    """Extract exact listed LKR price. Ignores old wide ranges like 230000-320000."""
    for line in text.splitlines():
        stripped = line.strip()
        if not stripped.lower().startswith("price"):
            continue
        # Ignore range-style prices
        if re.search(r"\d[\d,]*\s*-\s*\d", stripped):
            return None
        match = re.search(r"([\d,]+(?:\.\d+)?)", stripped)
        if not match:
            return None
        try:
            return float(match.group(1).replace(",", ""))
        except ValueError:
            return None
    return None


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


def get_laptops_in_budget(
    budget_min: float | None,
    budget_max: float | None,
    query: str = "",
    limit: int = 8,
) -> list[dict]:
    """
    Return laptop docs whose exact listed price is inside the budget range.
    Prefers docs that also match the workload query keywords.
    """
    if not KNOWLEDGE_BASE_DIR.exists():
        return []

    bmin = float(budget_min) if budget_min is not None else 0.0
    bmax = float(budget_max) if budget_max is not None else float("inf")
    if bmin > bmax:
        bmin, bmax = bmax, bmin

    query_terms = [t.lower() for t in re.findall(r"[a-zA-Z0-9]+", query) if len(t) > 2]
    scored: list[tuple[int, float, Path, str]] = []

    for path in KNOWLEDGE_BASE_DIR.glob("*.txt"):
        # Skip guide/shortlist docs; use real laptop listings only
        name = path.name.lower()
        if name.startswith("budget_") or "requirements" in name or "warnings" in name:
            continue

        text = path.read_text(encoding="utf-8", errors="ignore")
        price = _extract_price(text)
        if price is None:
            continue
        if price < bmin or price > bmax:
            continue

        lower = text.lower()
        score = sum(1 for term in query_terms if term in lower)
        scored.append((score, price, path, text))

    # Higher keyword match first, then closer to middle of budget
    mid = (bmin + bmax) / 2 if bmax != float("inf") else bmin
    scored.sort(key=lambda item: (-item[0], abs(item[1] - mid)))

    results = []
    for score, price, path, text in scored[:limit]:
        results.append(
            {
                "source": path.name,
                "price": price,
                "content": text.strip(),
                "score": score,
            }
        )
    return results


def retrieve_context(
    query: str,
    k: int = 4,
    rebuild: bool = False,
    budget_min: float | None = None,
    budget_max: float | None = None,
) -> str:
    """
    Retrieve relevant chunks for a user query.
    If budget range is given, prioritize exact-priced laptops inside that range.
    """
    parts: list[str] = []

    # 1) Exact budget-filtered laptop listings
    if budget_min is not None or budget_max is not None:
        in_budget = get_laptops_in_budget(
            budget_min=budget_min,
            budget_max=budget_max,
            query=query,
            limit=max(k, 6),
        )
        if in_budget:
            parts.append(
                "EXACT PRICE MATCHES INSIDE BUDGET RANGE "
                f"({budget_min} to {budget_max} LKR):\n"
                "Use ONLY these laptops for the recommendation table."
            )
            for i, item in enumerate(in_budget, start=1):
                parts.append(
                    f"[{i}] Source: {item['source']} | Exact Price: LKR {item['price']:,.0f}\n"
                    f"{item['content']}"
                )
        else:
            parts.append(
                "No exact-priced laptops were found inside the requested budget range. "
                "Say that clearly and do not invent in-range models."
            )

    # 2) Extra semantic RAG context (guides/warnings)
    vectorstore = get_vectorstore(rebuild=rebuild)
    docs = vectorstore.similarity_search(query, k=k)
    if docs:
        parts.append("ADDITIONAL CONTEXT (guides/warnings only):")
        for i, doc in enumerate(docs, start=1):
            source = Path(doc.metadata.get("source", "unknown")).name
            parts.append(f"[RAG-{i}] Source: {source}\n{doc.page_content}")

    if not parts:
        return "No relevant laptop documents were found."
    return "\n\n".join(parts)


if __name__ == "__main__":
    sample_query = "student laptop 16GB RAM under budget for coding"
    print("Building/loading FAISS index and retrieving context...\n")
    context = retrieve_context(
        sample_query,
        k=3,
        rebuild=True,
        budget_min=230000,
        budget_max=250000,
    )
    print(context)
