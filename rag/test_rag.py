"""
Simple test for the FAISS RAG engine.
Run: python -m rag.test_rag
"""

from rag.rag_engine import retrieve_context


def main() -> None:
    queries = [
        "best laptop for coding students with 16GB RAM",
        "budget gaming laptop with RTX GPU",
        "warnings about soldered RAM and battery life",
    ]

    print("Running RAG retrieval tests...\n")
    for i, query in enumerate(queries, start=1):
        print("=" * 60)
        print(f"Test {i}: {query}")
        print("=" * 60)
        context = retrieve_context(query, k=3, rebuild=(i == 1))
        print(context)
        print()

    print("RAG test finished. If chunks look related to each query, retrieval works.")


if __name__ == "__main__":
    main()
