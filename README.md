# agentic-laptop-advisor

An AI assistant that recommends laptops based on your budget and workload using a 3-agent pipeline and local spec data.

# Smart Specs — Agentic AI Laptop Advisor

An AI assistant that recommends laptops based on your budget and workload using a multi-agent pipeline and local RAG (real laptop specs), instead of a generic chatbot.

## Problem

Students and buyers often get wrong laptop advice: fake specs, outdated prices, or recommendations that ignore budget and workload.

## Solution

Smart Specs uses 4 agents:

1. **Agent 1 — Intent Router (Groq):** Parses budget and workload into structured JSON.
2. **Agent 2 — RAG Advisor (OpenRouter):** Retrieves real laptop/spec documents from a local FAISS index and drafts 2–3 recommendations.
3. **Agent 3 — Feasibility Critic:** Audits the draft for budget fit and hardware risks (e.g. non-expandable RAM, battery warnings).
4. **Agent 4 — PDF Generator:** Converts the final report into a downloadable PDF.

## Final Output

A Streamlit app shows a Markdown comparison table (2–3 laptops) with specs, price ranges, advantages, and buyer warnings, plus a PDF download.

## Tech Stack

- Python, Streamlit
- Groq + OpenRouter (2 different model providers)
- LangChain + FAISS + sentence-transformers (RAG)
- ReportLab (PDF)

## Project Structure

- `agents/` — Agent 1–4 logic
- `rag/` — FAISS RAG engine
- `data/knowledge_base/` — domain documents (20+)
- `app.py` — Streamlit UI
- `reports/` — generated PDFs

