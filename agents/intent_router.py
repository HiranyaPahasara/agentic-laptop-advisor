"""
Agent 1 — Intent Router (Groq)

Takes a user's natural-language laptop request and converts it into
structured JSON (budget + workload + preferences) for Agent 2.
"""

from __future__ import annotations

import json
import re
from typing import Any

from agents.secrets_util import get_secret
from groq import Groq

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
    api_key = get_secret("GROQ_API_KEY")
    if not api_key or api_key == "your_groq_key_here":
        raise ValueError(
            "GROQ_API_KEY is missing. Add it to .env locally or Streamlit Secrets on Cloud."
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


def _clean_string_list(value: Any) -> list[str]:
    """Normalize list fields into a clean list of non-empty strings."""
    if value is None:
        return []
    if isinstance(value, str):
        value = [value]
    if not isinstance(value, list):
        value = [value]

    cleaned: list[str] = []
    for item in value:
        text = str(item).strip()
        if text:
            cleaned.append(text)
    return cleaned


def _guess_budget_max(user_text: str) -> float | None:
    """Best-effort budget parse from raw text if the model fails."""
    text = user_text.lower().replace(",", "")

    match_k = re.search(r"(\d+(?:\.\d+)?)\s*k\b", text)
    if match_k:
        return float(match_k.group(1)) * 1000

    match_num = re.search(r"(?:under|below|budget(?:\s+of)?|max(?:imum)?)\s*(\d{5,7})", text)
    if match_num:
        return float(match_num.group(1))

    match_plain = re.search(r"\b(\d{5,7})\b", text)
    if match_plain:
        return float(match_plain.group(1))

    return None


def _guess_workload(user_text: str) -> str:
    """Best-effort workload label from raw text."""
    text = user_text.lower()
    mapping = [
        ("video", "video_editing"),
        ("edit", "video_editing"),
        ("gaming", "gaming"),
        ("game", "gaming"),
        ("cod", "coding"),
        ("program", "coding"),
        ("student", "student"),
        ("office", "office"),
        ("zoom", "student"),
    ]
    for keyword, label in mapping:
        if keyword in text:
            return label
    return "general"


def _fallback_intent(user_text: str) -> dict[str, Any]:
    """Safe structured intent when Groq output is invalid."""
    return {
        "budget_min": None,
        "budget_max": _guess_budget_max(user_text),
        "currency": "LKR",
        "workload": _guess_workload(user_text),
        "priorities": [],
        "must_have": [],
        "nice_to_have": [],
        "constraints": ["parsed_with_fallback"],
    }


def _validate_intent(data: dict[str, Any]) -> dict[str, Any]:
    """Ensure required keys exist with safe defaults and sane budget values."""
    validated = {
        "budget_min": data.get("budget_min"),
        "budget_max": data.get("budget_max"),
        "currency": data.get("currency") or "LKR",
        "workload": data.get("workload") or "general",
        "priorities": _clean_string_list(data.get("priorities")),
        "must_have": _clean_string_list(data.get("must_have")),
        "nice_to_have": _clean_string_list(data.get("nice_to_have")),
        "constraints": _clean_string_list(data.get("constraints")),
    }

    for key in ("budget_min", "budget_max"):
        value = validated[key]
        if value is not None:
            try:
                number = float(value)
                validated[key] = number if number > 0 else None
            except (TypeError, ValueError):
                validated[key] = None

    # Fix swapped ranges: min should not be greater than max
    bmin, bmax = validated["budget_min"], validated["budget_max"]
    if bmin is not None and bmax is not None and bmin > bmax:
        validated["budget_min"], validated["budget_max"] = bmax, bmin

    if not isinstance(validated["workload"], str) or not validated["workload"].strip():
        validated["workload"] = "general"
    else:
        validated["workload"] = validated["workload"].strip().lower().replace(" ", "_")

    if not isinstance(validated["currency"], str) or not validated["currency"].strip():
        validated["currency"] = "LKR"

    return validated


def parse_user_intent(user_text: str) -> dict[str, Any]:
    """
    Agent 1 entry point.
    Input: free-text user request
    Output: structured intent JSON (Python dict)
    Retries once, then falls back to heuristic parsing if JSON is invalid.
    """
    if not user_text or not user_text.strip():
        raise ValueError("User text is empty.")

    client = _get_groq_client()
    last_error: Exception | None = None

    for attempt in range(2):
        try:
            response = client.chat.completions.create(
                model=GROQ_MODEL,
                temperature=0.1,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {
                        "role": "user",
                        "content": (
                            "Parse this laptop request into JSON only.\n"
                            f"Attempt: {attempt + 1}\n"
                            f"Request: {user_text.strip()}"
                        ),
                    },
                ],
            )
            raw = response.choices[0].message.content or ""
            parsed = _extract_json(raw)
            return _validate_intent(parsed)
        except Exception as exc:  # noqa: BLE001 - keep Agent 1 resilient
            last_error = exc

    # Model failed twice -> still return usable structured data
    fallback = _validate_intent(_fallback_intent(user_text))
    fallback["constraints"] = _clean_string_list(
        fallback.get("constraints", []) + [f"fallback_reason: {last_error}"]
    )
    return fallback


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
