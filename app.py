"""
Smart Specs — Streamlit UI

Collects budget + workload, runs the multi-agent pipeline,
and shows the final Markdown recommendations.
"""

from __future__ import annotations

import streamlit as st

from agents.feasibility_critic import critique_recommendations
from agents.intent_router import parse_user_intent
from agents.rag_advisor import draft_recommendations

st.set_page_config(
    page_title="Smart Specs | Laptop Advisor",
    page_icon="💻",
    layout="centered",
)

st.title("Smart Specs")
st.caption("Agentic AI Laptop Recommendation System")

st.markdown(
    """
Smart Specs uses a multi-agent pipeline with local RAG:

1. **Intent Router** — understands your budget and workload  
2. **RAG Advisor** — retrieves real laptop knowledge  
3. **Feasibility Critic** — checks budget fit and buyer risks  
4. **PDF Generator** — creates a downloadable report  
"""
)

st.subheader("Your requirements")

with st.form("laptop_request_form"):
    budget = st.number_input(
        "Maximum budget (LKR)",
        min_value=50000,
        max_value=2000000,
        value=250000,
        step=10000,
        help="Enter the highest price you can pay.",
    )
    workload = st.text_area(
        "Workload / use case",
        value="University coding student. Need 16GB RAM and good battery.",
        height=120,
        help="Describe what you will use the laptop for.",
    )
    extra_notes = st.text_input(
        "Extra preferences (optional)",
        placeholder="e.g. lightweight, backlit keyboard, quiet fans",
    )
    submitted = st.form_submit_button("Get Recommendations", type="primary")

if submitted:
    parts = [f"Budget under {int(budget)} LKR.", workload.strip()]
    if extra_notes.strip():
        parts.append(extra_notes.strip())
    user_text = " ".join(parts)

    try:
        with st.status("Running Smart Specs agents...", expanded=True) as status:
            st.write("Agent 1 — parsing intent with Groq...")
            intent = parse_user_intent(user_text)

            st.write("Agent 2 — retrieving specs and drafting recommendations...")
            draft = draft_recommendations(intent)

            st.write("Agent 3 — auditing budget fit and buyer risks...")
            final_report = critique_recommendations(intent, draft)

            status.update(label="Recommendations ready", state="complete")

        st.session_state["intent"] = intent
        st.session_state["draft"] = draft
        st.session_state["final_report"] = final_report

    except Exception as exc:  # noqa: BLE001 - show friendly UI errors
        st.error(f"Pipeline failed: {exc}")
        st.stop()

if "final_report" in st.session_state:
    st.subheader("Structured intent (Agent 1)")
    st.json(st.session_state["intent"])

    with st.expander("Advisor draft (Agent 2)", expanded=False):
        st.markdown(st.session_state["draft"])

    st.subheader("Final recommendations (Agent 3)")
    st.markdown(st.session_state["final_report"])
else:
    st.info("Enter your budget and workload, then click **Get Recommendations**.")
