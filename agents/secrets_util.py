"""Load API keys from .env locally or Streamlit Secrets on Cloud."""

from __future__ import annotations

import os

from dotenv import load_dotenv

load_dotenv()


def get_secret(name: str, default: str | None = None) -> str | None:
    """Prefer environment variables, then Streamlit secrets."""
    value = os.getenv(name)
    if value:
        return value

    try:
        import streamlit as st

        if hasattr(st, "secrets") and name in st.secrets:
            return str(st.secrets[name])
    except Exception:
        pass

    return default
