"""
Smart Specs — Streamlit UI

Collects budget + workload, then runs the multi-agent pipeline.
"""

from __future__ import annotations

import streamlit as st

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

    st.session_state["user_text"] = user_text
    st.success("Request captured. Pipeline wiring comes in the next UI update.")
    st.write("**Parsed request preview:**")
    st.code(user_text)
else:
    st.info("Enter your budget and workload, then click **Get Recommendations**.")
