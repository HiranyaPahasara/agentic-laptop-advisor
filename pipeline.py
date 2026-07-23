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
from agents.rag_advisor import draft_recommendations


def run_recommendation_pipeline(user_text: str) -> dict[str, Any]:
    """
    Run the full 3-agent recommendation flow.

    Returns a dict with:
    - intent: Agent 1 JSON
    - draft: Agent 2 Markdown
    - final_report: Agent 3 audited Markdown
    """
    if not user_text or not str(user_text).strip():
        raise ValueError("Please provide a laptop request (budget + workload).")

    # Agent 1
    intent = parse_user_intent(user_text)

    # Agent 2
    draft = draft_recommendations(intent)

    # Agent 3
    final_report = critique_recommendations(intent, draft)

    return {
        "intent": intent,
        "draft": draft,
        "final_report": final_report,
    }


if __name__ == "__main__":
    sample = (
        "I am a university student learning Python coding. "
        "Budget is under 250000 LKR. I need 16GB RAM and decent battery."
    )
    print("Running Smart Specs pipeline...\n")
    result = run_recommendation_pipeline(sample)

    print("=== Agent 1 Intent ===")
    print(result["intent"])
    print("\n=== Agent 2 Draft (preview) ===")
    print(result["draft"][:500], "...\n")
    print("=== Agent 3 Final Report ===")
    print(result["final_report"])
