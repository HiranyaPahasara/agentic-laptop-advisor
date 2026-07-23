"""
Agent 2 — RAG Advisor

Takes structured intent from Agent 1, retrieves real laptop docs via FAISS RAG,
and drafts 2-3 recommendations as a Markdown comparison table.

Providers:
- Default: Groq (free) with a DIFFERENT model than Agent 1
- Optional: OpenRouter when you have credits

Set in .env:
  AGENT2_PROVIDER=groq          # default (no payment needed)
  AGENT2_PROVIDER=openrouter    # later, when you add OpenRouter credits
"""

from __future__ import annotations

import json
import os
from typing import Any

from dotenv import load_dotenv
from groq import Groq
from openai import OpenAI

from rag.rag_engine import retrieve_context

load_dotenv()

# Agent 1 uses Groq llama-3.1-8b-instant
# Agent 2 default uses a different Groq model (still free)
GROQ_MODEL = os.getenv("AGENT2_GROQ_MODEL", "llama-3.3-70b-versatile")

OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"
OPENROUTER_MODEL = os.getenv(
    "OPENROUTER_MODEL",
    "meta-llama/llama-3.3-70b-instruct",
)

# groq | openrouter
AGENT2_PROVIDER = os.getenv("AGENT2_PROVIDER", "groq").strip().lower()

SYSTEM_PROMPT = """
You are Agent 2 (RAG Advisor) for a laptop recommendation system.
Use ONLY the provided retrieved context for specs and price guidance.
Do not invent exact prices or fake model specs.

Return Markdown only with:
1) A short intro (2-3 sentences)
2) A comparison table with columns:
   Model | Key Specs | Est. Price (LKR) | Best For | Notes
3) Exactly 2 or 3 laptop recommendations from the context
4) A section called "## Best Solution" that clearly names ONE winner
   from the table and explains why it is the best fit (budget + workload)
5) A short "Why these fit" bullet list

Rules for Best Solution:
- Must be one of the 2-3 recommended models
- Prefer in-budget options
- Prefer better RAM/storage match for the workload
- Do not pick an over-budget laptop as Best Solution

If budget is given, prefer options near/under budget_max.
If context is weak, say what is uncertain instead of guessing.
""".strip()


def _chat_with_groq(messages: list[dict[str, str]]) -> str:
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key or api_key == "your_groq_key_here":
        raise ValueError("GROQ_API_KEY is missing in .env")

    client = Groq(api_key=api_key)
    response = client.chat.completions.create(
        model=GROQ_MODEL,
        temperature=0.3,
        messages=messages,
    )
    return response.choices[0].message.content or ""


def _chat_with_openrouter(messages: list[dict[str, str]]) -> str:
    api_key = os.getenv("OPENROUTER_API_KEY")
    if not api_key or api_key == "your_openrouter_key_here":
        raise ValueError("OPENROUTER_API_KEY is missing in .env")

    client = OpenAI(api_key=api_key, base_url=OPENROUTER_BASE_URL)
    response = client.chat.completions.create(
        model=OPENROUTER_MODEL,
        temperature=0.3,
        messages=messages,
    )
    return response.choices[0].message.content or ""


def _generate_draft(messages: list[dict[str, str]]) -> str:
    """Call the configured provider (Groq free by default)."""
    if AGENT2_PROVIDER == "openrouter":
        return _chat_with_openrouter(messages)
    return _chat_with_groq(messages)


def _intent_to_query(intent: dict[str, Any]) -> str:
    """Build a retrieval query from Agent 1 structured intent."""
    parts: list[str] = []

    workload = intent.get("workload")
    if workload:
        parts.append(str(workload))

    budget_max = intent.get("budget_max")
    currency = intent.get("currency") or "LKR"
    if budget_max is not None:
        parts.append(f"budget under {budget_max} {currency}")

    for key in ("must_have", "priorities", "nice_to_have", "constraints"):
        values = intent.get(key) or []
        if isinstance(values, list) and values:
            parts.append(", ".join(str(v) for v in values))

    parts.append("laptop recommendation specs price limitations")
    return " | ".join(parts)


def draft_recommendations(intent: dict[str, Any], k: int = 5) -> str:
    """
    Agent 2 entry point.
    Input: intent dict from Agent 1
    Output: Markdown recommendation draft for Agent 3
    """
    if not isinstance(intent, dict) or not intent:
        raise ValueError("Intent must be a non-empty dictionary from Agent 1.")

    query = _intent_to_query(intent)
    context = retrieve_context(query, k=k)

    user_prompt = f"""
User intent JSON:
{json.dumps(intent, indent=2)}

Retrieved context from local knowledge base:
{context}

Write the Markdown recommendation draft now.
Remember: include exactly 2-3 laptops in a table, then a clear "## Best Solution" section naming ONE winner.
""".strip()

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_prompt},
    ]
    draft = _generate_draft(messages)
    if not draft.strip():
        raise ValueError("Agent 2 returned an empty recommendation draft.")
    return draft.strip()


if __name__ == "__main__":
    sample_intent = {
        "budget_min": None,
        "budget_max": 250000,
        "currency": "LKR",
        "workload": "coding",
        "priorities": ["battery", "portability"],
        "must_have": ["16GB RAM"],
        "nice_to_have": ["backlit keyboard"],
        "constraints": [],
    }
    print(f"Agent 2 provider: {AGENT2_PROVIDER}")
    print("Agent 2 sample intent:")
    print(json.dumps(sample_intent, indent=2))
    print("\nAgent 2 Markdown draft:\n")
    print(draft_recommendations(sample_intent))
