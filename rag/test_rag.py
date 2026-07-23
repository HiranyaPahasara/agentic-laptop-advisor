"""
Simple test for the FAISS RAG engine.
Run: python -m rag.test_rag
"""

from rag.rag_engine import get_laptops_in_budget, retrieve_context


def main() -> None:
    queries = [
        "coding student 16GB RAM budget 230000 to 250000",
        "budget gaming laptop with RTX GPU",
        "warnings about soldered RAM and battery life",
        "video editing requirements 16GB RAM",
        "laptop under 200000 LKR Windows 11",
    ]

    print("Running 5-query RAG retrieval evaluation...\n")
    for i, query in enumerate(queries, start=1):
        print("=" * 60)
        print(f"Test {i}: {query}")
        print("=" * 60)
        context = retrieve_context(
            query,
            k=3,
            rebuild=(i == 1),
            budget_min=180000 if "200000" in query or "250000" in query else None,
            budget_max=250000 if "250000" in query else (200000 if "200000" in query else None),
        )
        print(context[:1200])
        print("...\n")

    print("Budget filter sample (230000-250000):")
    for item in get_laptops_in_budget(230000, 250000, "coding 16GB", 5):
        print(f"- LKR {int(item['price'])} | {item['source']}")

    print("\nRAG 5-query evaluation finished.")


if __name__ == "__main__":
    main()
