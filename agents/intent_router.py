"""
Agent 1 — Intent Router (Groq)

Takes a user's natural-language laptop request and converts it into
structured JSON (budget + workload + preferences) for Agent 2.
"""

from __future__ import annotations

import json
import os
import re
from typing import Any

from dotenv import load_dotenv
from groq import Groq

load_dotenv()

# Fast Groq model for structured parsing
GROQ_MODEL = "llama-3.1-8b-instant"

SYSTEM_PROMPT = """
You are Agent 1 (Intent Router) for a laptop recommendation system.
Extract the user's needs into STRICT JSON only. No markdown. No extra text.

Return exactly this schema:
{
  "budget_min": number or null,
  "budget_max": number or null,
  "currency": "LKR",
  "workload": "string",
  "priorities": ["string"],
  "must_have": ["string"],
  "nice_to_have": ["string"],
  "constraints": ["string"]
}

Rules:
- If budget is a single number, set budget_max to that number and budget_min to null.
- If budget is a range, fill both budget_min and budget_max.
- Convert budget words like "200k" to 200000 when currency is LKR.
- workload should be a short label, e.g. "coding", "student", "video_editing", "gaming", "office".
- priorities/must_have/nice_to_have/constraints should be short phrases.
- If something is unknown, use null or [].
""".strip()


def _get_groq_client() -> Groq:
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key or api_key == "your_groq_key_here":
        raise ValueError(
            "GROQ_API_KEY is missing. Put your real key in the .env file."
        )
    return Groq(api_key=api_key)


def _extract_json(text: str) -> dict[str, Any]:
    """Parse JSON from model output, even if wrapped in markdown fences."""
    text = text.strip()

    # Direct JSON first
    try:
        data = json.loads(text)
        if isinstance(data, dict):
            return data
    except json.JSONDecodeError:
        pass

    # Fenced code block: ```json ... ```
    fenced = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL)
    if fenced:
        return json.loads(fenced.group(1))

    # First {...} block
    brace = re.search(r"\{.*\}", text, re.DOTALL)
    if brace:
        return json.loads(brace.group(0))

    raise ValueError(f"Agent 1 did not return valid JSON: {text}")


def _validate_intent(data: dict[str, Any]) -> dict[str, Any]:
    """Ensure required keys exist with safe defaults."""
    validated = {
        "budget_min": data.get("budget_min"),
        "budget_max": data.get("budget_max"),
        "currency": data.get("currency") or "LKR",
        "workload": data.get("workload") or "general",
        "priorities": data.get("priorities") or [],
        "must_have": data.get("must_have") or [],
        "nice_to_have": data.get("nice_to_have") or [],
        "constraints": data.get("constraints") or [],
    }

    for key in ("priorities", "must_have", "nice_to_have", "constraints"):
        if not isinstance(validated[key], list):
            validated[key] = [str(validated[key])]

    for key in ("budget_min", "budget_max"):
        value = validated[key]
        if value is not None:
            try:
                validated[key] = float(value)
            except (TypeError, ValueError):
                validated[key] = None

    if not isinstance(validated["workload"], str):
        validated["workload"] = str(validated["workload"])

    return validated


def parse_user_intent(user_text: str) -> dict[str, Any]:
    """
    Agent 1 entry point.
    Input: free-text user request
    Output: structured intent JSON (Python dict)
    """
    if not user_text or not user_text.strip():
        raise ValueError("User text is empty.")

    client = _get_groq_client()
    response = client.chat.completions.create(
        model=GROQ_MODEL,
        temperature=0.1,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {
                "role": "user",
                "content": f"Parse this laptop request into JSON:\n{user_text.strip()}",
            },
        ],
    )

    raw = response.choices[0].message.content or ""
    parsed = _extract_json(raw)
    return _validate_intent(parsed)


if __name__ == "__main__":
    sample = (
        "I need a laptop for university coding under 250000 LKR. "
        "Prefer 16GB RAM and good battery."
    )
    print("Agent 1 sample input:")
    print(sample)
    print("\nAgent 1 structured output:")
    intent = parse_user_intent(sample)
    print(json.dumps(intent, indent=2))
