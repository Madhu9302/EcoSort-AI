"""
Session state helpers for the Streamlit application.
"""

from __future__ import annotations
import streamlit as st
from datetime import datetime


def init_session_state() -> None:
    """Initialise all required session-state keys on first run."""
    defaults = {
        "classification_history": [],   # list of past classification results
        "chat_history": [],             # list of {role, content} dicts for Eco Assistant
        "model_loaded": False,          # whether the ML model is ready
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


def record_classification(waste_class: str, confidence: float, image_name: str = "") -> None:
    """Append a classification result to the session history."""
    st.session_state.classification_history.append({
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "image": image_name,
        "predicted_class": waste_class,
        "confidence": confidence,
    })
