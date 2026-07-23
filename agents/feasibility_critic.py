"""
Agent 3 — Feasibility Critic (reflection / audit)

Reviews Agent 2's Markdown draft against the user's intent.
Checks budget fit and adds buyer warnings (RAM, battery, upgrade limits).
"""

from __future__ import annotations

import json
import os
from typing import Any

from dotenv import load_dotenv
from groq import Groq

load_dotenv()

# Different role from Agent 1/2: critic / auditor
GROQ_MODEL = os.getenv("AGENT3_GROQ_MODEL", "llama-3.1-8b-instant")

SYSTEM_PROMPT = """
You are Agent 3 (Feasibility Critic) for a laptop recommendation system.
Audit the Advisor draft. Do not invent new laptop models.

Your job:
1) Check each recommended laptop against the budget range
   (budget_min to budget_max when both exist; otherwise budget_max)
2) Flag risks: soldered/non-upgradeable RAM, weak battery, 8GB RAM bottlenecks,
   heavy gaming laptops for study-only needs, storage too small, etc.
3) Keep good recommendations, but add clear buyer warnings
4) If a laptop is clearly outside the range, mark it OUT OF BUDGET
5) Choose exactly ONE Best Solution from options inside the budget range

Return Markdown only with these sections:
## Final Recommendations
(keep/adjust the comparison table)

## Best Solution
- Name the single best laptop from the list
- Give 2-4 short reasons (budget fit, workload fit, specs, fewer risks)
- Do NOT choose an OUT OF BUDGET laptop as Best Solution

## Feasibility Audit
- budget range check bullets

## Buyer Warnings
- important warnings the user should read before buying

## Critic Verdict
One short paragraph: approve / approve with warnings / revise
""".strip()


def _get_groq_client() -> Groq:
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key or api_key == "your_groq_key_here":
        raise ValueError("GROQ_API_KEY is missing in .env")
    return Groq(api_key=api_key)


def critique_recommendations(intent: dict[str, Any], draft_markdown: str) -> str:
    """
    Agent 3 entry point.
    Input: Agent 1 intent + Agent 2 Markdown draft
    Output: audited final Markdown report
    """
    if not isinstance(intent, dict) or not intent:
        raise ValueError("Intent must be a non-empty dictionary from Agent 1.")
    if not draft_markdown or not draft_markdown.strip():
        raise ValueError("Advisor draft is empty.")

    client = _get_groq_client()
    user_prompt = f"""
User intent JSON:
{json.dumps(intent, indent=2)}

Agent 2 draft to audit:
{draft_markdown.strip()}

Produce the audited Markdown report now.
""".strip()

    response = client.chat.completions.create(
        model=GROQ_MODEL,
        temperature=0.2,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
        ],
    )

    report = response.choices[0].message.content or ""
    if not report.strip():
        raise ValueError("Agent 3 returned an empty audit report.")
    return report.strip()


if __name__ == "__main__":
    sample_intent = {
        "budget_min": None,
        "budget_max": 250000,
        "currency": "LKR",
        "workload": "coding",
        "priorities": ["battery"],
        "must_have": ["16GB RAM"],
        "nice_to_have": [],
        "constraints": [],
    }
    sample_draft = """
## Laptop Recommendations
| Model | Key Specs | Est. Price (LKR) | Best For | Notes |
| --- | --- | --- | --- | --- |
| ASUS Vivobook 15 | i5, 16GB, 512GB | 180000-260000 | Coding/study | Check RAM upgrade |
| Lenovo IdeaPad Slim 3 | i3/i5, 8GB, 512GB | 150000-220000 | Budget study | 8GB may be tight |
"""
    print("Agent 3 sample audit:\n")
    print(critique_recommendations(sample_intent, sample_draft))
