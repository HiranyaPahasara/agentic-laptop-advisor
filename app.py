"""
Smart Specs — Streamlit UI

Collects budget + workload, runs the multi-agent pipeline,
shows Markdown recommendations, and offers PDF download.
"""

from __future__ import annotations

import re

import streamlit as st

from agents.feasibility_critic import critique_recommendations
from agents.intent_router import parse_user_intent
from agents.pdf_generator import save_pdf_for_download
from agents.rag_advisor import draft_recommendations

st.set_page_config(
    page_title="Smart Specs | Laptop Advisor",
    page_icon="💻",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# Visual direction: cool slate + teal (not purple / not cream-terracotta)
st.markdown(
    """
<style>
@import url('https://fonts.googleapis.com/css2?family=Sora:wght@500;600;700&family=Source+Sans+3:wght@400;500;600&display=swap');

:root {
  --bg1: #0b1220;
  --bg2: #102033;
  --panel: rgba(255, 255, 255, 0.06);
  --line: rgba(148, 163, 184, 0.28);
  --text: #e8eef7;
  --muted: #9fb0c6;
  --accent: #14b8a6;
  --accent-2: #38bdf8;
}

html, body, [class*="css"] {
  font-family: "Source Sans 3", sans-serif;
}

.stApp {
  background:
    radial-gradient(1200px 500px at 10% -10%, rgba(20, 184, 166, 0.18), transparent 55%),
    radial-gradient(900px 500px at 100% 0%, rgba(56, 189, 248, 0.14), transparent 50%),
    linear-gradient(160deg, var(--bg1), var(--bg2));
  color: var(--text);
}

/* Hide default Streamlit chrome clutter */
#MainMenu {visibility: hidden;}
footer {visibility: hidden;}
header {visibility: hidden;}

.block-container {
  max-width: 920px;
  padding-top: 1.6rem;
  padding-bottom: 3rem;
}

.hero-panel {
  position: relative;
  overflow: hidden;
  background:
    linear-gradient(135deg, rgba(15, 23, 42, 0.55), rgba(8, 47, 73, 0.28)),
    radial-gradient(600px 220px at 15% 20%, rgba(45, 212, 191, 0.16), transparent 60%),
    radial-gradient(520px 240px at 90% 10%, rgba(56, 189, 248, 0.14), transparent 55%);
  border: 1px solid rgba(148, 163, 184, 0.22);
  border-radius: 24px;
  padding: 2.1rem 1.8rem 1.9rem;
  backdrop-filter: blur(10px);
  box-shadow: 0 18px 50px rgba(2, 8, 23, 0.35);
  animation: heroIn 0.7s ease-out both;
}

.hero-panel::before {
  content: "";
  position: absolute;
  inset: 0;
  background-image:
    linear-gradient(rgba(148, 163, 184, 0.06) 1px, transparent 1px),
    linear-gradient(90deg, rgba(148, 163, 184, 0.06) 1px, transparent 1px);
  background-size: 28px 28px;
  mask-image: linear-gradient(180deg, rgba(0,0,0,0.55), transparent 85%);
  pointer-events: none;
}

.hero-kicker {
  position: relative;
  display: inline-block;
  margin: 0 0 0.85rem;
  padding: 0.28rem 0.7rem;
  border-radius: 999px;
  border: 1px solid rgba(45, 212, 191, 0.35);
  background: rgba(15, 118, 110, 0.18);
  color: #99f6e4;
  font-size: 0.78rem;
  font-weight: 600;
  letter-spacing: 0.08em;
  text-transform: uppercase;
  animation: fadeUp 0.7s ease-out 0.1s both;
}

.brand {
  position: relative;
  font-family: "Sora", sans-serif;
  font-size: clamp(2.8rem, 6vw, 4rem);
  font-weight: 700;
  letter-spacing: -0.04em;
  line-height: 1.02;
  margin: 0;
  background: linear-gradient(100deg, #f8fafc 10%, #5eead4 48%, #38bdf8 88%);
  -webkit-background-clip: text;
  -webkit-text-fill-color: transparent;
  animation: fadeUp 0.75s ease-out 0.18s both;
}

.brand-underline {
  position: relative;
  width: 86px;
  height: 4px;
  margin-top: 0.85rem;
  border-radius: 999px;
  background: linear-gradient(90deg, #14b8a6, #38bdf8);
  animation: growLine 0.8s ease-out 0.35s both;
}

.tagline {
  position: relative;
  margin: 1rem 0 0;
  color: #c6d4e6;
  font-size: clamp(1.02rem, 2.2vw, 1.18rem);
  line-height: 1.55;
  max-width: 34rem;
  animation: fadeUp 0.75s ease-out 0.28s both;
}

@keyframes heroIn {
  from { opacity: 0; transform: translateY(10px); }
  to { opacity: 1; transform: translateY(0); }
}

@keyframes fadeUp {
  from { opacity: 0; transform: translateY(8px); }
  to { opacity: 1; transform: translateY(0); }
}

@keyframes growLine {
  from { width: 0; opacity: 0; }
  to { width: 86px; opacity: 1; }
}

.form-panel, .result-panel {
  background: var(--panel);
  border: 1px solid var(--line);
  border-radius: 18px;
  padding: 1.35rem 1.4rem;
  backdrop-filter: blur(8px);
}

.best-box {
  margin: 0.4rem 0 1.1rem;
  padding: 1.05rem 1.15rem;
  border-radius: 16px;
  border: 1px solid rgba(45, 212, 191, 0.45);
  background:
    linear-gradient(135deg, rgba(20, 184, 166, 0.18), rgba(56, 189, 248, 0.10));
  box-shadow: 0 10px 30px rgba(2, 8, 23, 0.25);
}

.best-box .best-label {
  margin: 0 0 0.35rem;
  font-family: "Sora", sans-serif;
  font-size: 0.78rem;
  letter-spacing: 0.08em;
  text-transform: uppercase;
  color: #99f6e4;
  font-weight: 700;
}

.best-box .best-score {
  margin: 0 0 0.55rem;
  font-family: "Sora", sans-serif;
  font-size: 1.15rem;
  color: #5eead4;
}

.best-box .best-body {
  margin: 0;
  color: #e8eef7;
  line-height: 1.5;
  font-size: 1.02rem;
}

section[data-testid="stForm"] {
  background: transparent;
  border: 0;
}

.stTextInput input, .stNumberInput input, .stTextArea textarea {
  border-radius: 12px !important;
}

div[data-testid="stFormSubmitButton"] button {
  background: linear-gradient(90deg, #0d9488, #0284c7) !important;
  color: white !important;
  border: 0 !important;
  border-radius: 12px !important;
  font-weight: 600 !important;
  padding: 0.55rem 1rem !important;
}

h2, h3 {
  font-family: "Sora", sans-serif !important;
  letter-spacing: -0.02em;
}

[data-testid="stMarkdownContainer"] p, [data-testid="stMarkdownContainer"] li {
  color: #dbe7f5;
}
</style>
""",
    unsafe_allow_html=True,
)

st.markdown(
    """
<div class="hero-panel">
  <p class="hero-kicker">Laptop Recommendation System</p>
  <p class="brand">Smart Specs</p>
  <div class="brand-underline"></div>
  <p class="tagline">
    Real local specs. Budget-aware advice. Clear buyer warnings —
    so you choose a laptop with confidence.
  </p>
</div>
""",
    unsafe_allow_html=True,
)

st.write("")
st.subheader("Find your laptop")
st.caption(
    "Set your price range and workload. Smart Specs will recommend 2–3 laptops in that budget and name the Best Solution."
)

with st.form("laptop_request_form"):
    col_min, col_max = st.columns(2)
    with col_min:
        budget_min = st.number_input(
            "Minimum budget (LKR)",
            min_value=50000,
            max_value=2000000,
            value=180000,
            step=10000,
            help="Lowest price you want to consider.",
        )
    with col_max:
        budget_max = st.number_input(
            "Maximum budget (LKR)",
            min_value=50000,
            max_value=2000000,
            value=250000,
            step=10000,
            help="Highest price you can pay.",
        )

    workload = st.text_area(
        "Workload / use case",
        value="",
        placeholder="Example: University coding student. Need 16GB RAM and good battery.",
        height=120,
        help="What will you use the laptop for?",
    )
    submitted = st.form_submit_button("Get Recommendations", use_container_width=True)

def is_valid_workload(text: str) -> bool:
    """Reject greetings / empty chatty input that is not a real laptop use case."""
    cleaned = " ".join((text or "").lower().split())
    if len(cleaned) < 12:
        return False

    greetings = {
        "hi",
        "hello",
        "hey",
        "hi how are you",
        "hello how are you",
        "how are you",
        "good morning",
        "good night",
        "thanks",
        "thank you",
    }
    if cleaned in greetings:
        return False

    # Must mention at least one laptop/use-case signal
    signals = [
        "laptop",
        "student",
        "coding",
        "program",
        "office",
        "game",
        "gaming",
        "edit",
        "video",
        "design",
        "ram",
        "battery",
        "study",
        "university",
        "work",
        "zoom",
        "browsing",
    ]
    return any(signal in cleaned for signal in signals)


if submitted:
    if budget_min > budget_max:
        st.error("Minimum budget cannot be greater than maximum budget.")
        st.stop()

    if not is_valid_workload(workload):
        st.warning(
            "Please describe a real laptop use case (example: university coding, 16GB RAM, good battery). "
            "Greetings like “Hi how are you” are not valid workload input."
        )
        st.stop()

    user_text = (
        f"Budget range {int(budget_min)} to {int(budget_max)} LKR. "
        f"{workload.strip()}"
    )

    try:
        with st.status("Running Smart Specs agents...", expanded=True) as status:
            st.write("Agent 1 — parsing intent...")
            intent = parse_user_intent(user_text)
            # Keep the exact UI price range (more reliable than model parsing)
            intent["budget_min"] = float(budget_min)
            intent["budget_max"] = float(budget_max)
            intent["currency"] = "LKR"

            st.write("Agent 2 — retrieving specs and drafting recommendations...")
            draft = draft_recommendations(intent)

            st.write("Agent 3 — auditing budget fit and buyer risks...")
            final_report = critique_recommendations(intent, draft)

            st.write("Agent 4 — generating PDF report...")
            pdf_info = save_pdf_for_download(final_report)

            status.update(label="Recommendations ready", state="complete")

        st.session_state["intent"] = intent
        st.session_state["draft"] = draft
        st.session_state["final_report"] = final_report
        st.session_state["pdf_info"] = pdf_info

    except Exception as exc:  # noqa: BLE001 - show friendly UI errors
        st.error(f"Pipeline failed: {exc}")
        st.stop()

def extract_best_solution(report_markdown: str) -> str:
    """Pull the Best Solution section for a highlighted UI card."""
    text = report_markdown or ""
    match = re.search(
        r"##\s*Best Solution\s*(.*?)(?=\n##\s|\Z)",
        text,
        flags=re.IGNORECASE | re.DOTALL,
    )
    if not match:
        return ""
    body = match.group(1).strip()
    # Keep it readable in the highlight card
    body = re.sub(r"\n{3,}", "\n\n", body)
    return body


def extract_match_percent(best_solution_text: str) -> int | None:
    """Extract Match score percentage from Best Solution text if present."""
    match = re.search(
        r"Match\s*(?:score|%)?\s*[:=]?\s*(\d{1,3})\s*%",
        best_solution_text or "",
        flags=re.IGNORECASE,
    )
    if not match:
        # fallback: any standalone xx%
        match = re.search(r"\b(\d{1,3})\s*%", best_solution_text or "")
    if not match:
        return None
    value = int(match.group(1))
    return max(0, min(100, value))


def report_without_best_solution(report_markdown: str) -> str:
    """Remove Best Solution section so it is not shown twice in the UI."""
    text = report_markdown or ""
    cleaned = re.sub(
        r"##\s*Best Solution\s*.*?(?=\n##\s|\Z)",
        "",
        text,
        flags=re.IGNORECASE | re.DOTALL,
    )
    return re.sub(r"\n{3,}", "\n\n", cleaned).strip()


if "final_report" in st.session_state:
    st.write("")
    st.subheader("Results")

    best_solution = extract_best_solution(st.session_state["final_report"])
    if best_solution:
        match_pct = extract_match_percent(best_solution)
        safe = (
            best_solution.replace("&", "&amp;")
            .replace("<", "&lt;")
            .replace(">", "&gt;")
            .replace("\n", "<br>")
        )
        score_html = (
            f'<p class="best-score">Match score: <strong>{match_pct}%</strong></p>'
            if match_pct is not None
            else ""
        )
        st.markdown(
            f"""
<div class="best-box">
  <p class="best-label">Best Solution</p>
  {score_html}
  <div class="best-body">{safe}</div>
</div>
""",
            unsafe_allow_html=True,
        )
        st.caption("Every laptop has tradeoffs — this is the strongest overall fit in your budget range.")

    st.markdown("### Recommendations")
    st.markdown(report_without_best_solution(st.session_state["final_report"]))

    pdf_info = st.session_state.get("pdf_info")
    if pdf_info:
        st.download_button(
            label="Download PDF report",
            data=pdf_info["bytes"],
            file_name=pdf_info["filename"],
            mime="application/pdf",
            use_container_width=True,
        )
else:
    st.caption("Enter your budget range and workload, then click Get Recommendations.")
