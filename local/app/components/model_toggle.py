import streamlit as st

# Canonical engine identifiers used across the app (intent.py, main.py,
# handlers). The UI shows friendly labels; comparisons use these constants
# so model choices can change without touching routing logic.
ENGINE_CLOUD = "cloud"
ENGINE_LOCAL = "local"

_LABELS = {
    ENGINE_CLOUD: "☁️ Cloud",
    ENGINE_LOCAL: "💻 Local",
}


def render_model_toggle() -> str:
    """
    Renders the Cloud/Local execution-tier toggle.
    Returns ENGINE_CLOUD or ENGINE_LOCAL.
    """
    st.markdown("### 🧠 Inference Engine Selection")

    if "active_engine" not in st.session_state:
        st.session_state["active_engine"] = ENGINE_CLOUD

    selected_label = st.radio(
        "Route Generation Through:",
        options=[_LABELS[ENGINE_CLOUD], _LABELS[ENGINE_LOCAL]],
        index=0 if st.session_state["active_engine"] == ENGINE_CLOUD else 1,
        horizontal=True,
        help="Cloud: full multi-source reasoning via the backend API. "
             "Local: fully on-device inference — nothing leaves this machine.",
    )
    selected_engine = ENGINE_CLOUD if selected_label == _LABELS[ENGINE_CLOUD] else ENGINE_LOCAL

    st.session_state["active_engine"] = selected_engine

    if selected_engine == ENGINE_LOCAL:
        st.warning(
            "⚠️ **Local tier active:** best for definitional and single-document "
            "queries. `COMPARE` and `EWS` modes require the Cloud tier."
        )
    else:
        st.caption(
            "☁️ **Cloud tier active:** full `EXTRACT`, `BENCHMARK`, and `HYBRID` "
            "multi-source reasoning."
        )

    return selected_engine
