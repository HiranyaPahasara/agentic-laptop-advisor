"""
Smart Specs multi-agent pipeline.

Flow:
1) Agent 1 Intent Router  -> structured JSON
2) Agent 2 RAG Advisor    -> Markdown draft (uses FAISS retrieval)
3) Agent 3 Feasibility Critic -> audited final Markdown report
"""

from __future__ import annotations

from typing import Any

from agents.feasibility_critic import critique_recommendations
from agents.intent_router import parse_user_intent
from agents.pdf_generator import save_pdf_for_download
from agents.rag_advisor import draft_recommendations


def run_recommendation_pipeline(user_text: str, make_pdf: bool = True) -> dict[str, Any]:
    """
    Run the full agent recommendation flow.

    Returns a dict with:
    - intent: Agent 1 JSON
    - draft: Agent 2 Markdown
    - final_report: Agent 3 audited Markdown
    - pdf: optional dict with path/bytes/filename from Agent 4
    """
    if not user_text or not str(user_text).strip():
        raise ValueError("Please provide a laptop request (budget + workload).")

    # Agent 1
    intent = parse_user_intent(user_text)

    # Agent 2
    draft = draft_recommendations(intent)

    # Agent 3
    final_report = critique_recommendations(intent, draft)

    result: dict[str, Any] = {
        "intent": intent,
        "draft": draft,
        "final_report": final_report,
        "pdf": None,
    }

    # Agent 4
    if make_pdf:
        result["pdf"] = save_pdf_for_download(final_report)

    return result


if __name__ == "__main__":
    sample = (
        "I am a university student learning Python coding. "
        "Budget is under 250000 LKR. I need 16GB RAM and decent battery."
    )
    print("Running Smart Specs pipeline...\n")
    result = run_recommendation_pipeline(sample, make_pdf=True)

    print("=== Agent 1 Intent ===")
    print(result["intent"])
    print("\n=== Agent 2 Draft (preview) ===")
    print(result["draft"][:500], "...\n")
    print("=== Agent 3 Final Report ===")
    print(result["final_report"])
    if result.get("pdf"):
        print("\n=== Agent 4 PDF ===")
        print(result["pdf"]["path"])
