import streamlit as st

def render_model_toggle():
    """
    Renders the A/B testing toggle between the local SLM and cloud LLM.
    Acts as the centerpiece for demonstrating terminology precision and 
    architectural constraints during the technical interview.
    """
    st.markdown("### 🧠 Inference Engine Selection")
    
    # Initialize default state
    if "active_model" not in st.session_state:
        st.session_state["active_model"] = "Gemini Pro (Cloud)"

    # Render horizontal toggle for clean UI
    selected_model = st.radio(
        "Route Generation Through:",
        options=["Gemini Pro (Cloud)", "Phi-3 (Local Edge)"],
        index=0 if st.session_state["active_model"] == "Gemini Pro (Cloud)" else 1,
        horizontal=True,
        help="Compare terminology precision between the localized SLM and the heavy cloud LLM."
    )
    
    st.session_state["active_model"] = selected_model

    # Enforce the strict constraints outlined in the project plan
    if selected_model == "Phi-3 (Local Edge)":
        st.warning(
            "⚠️ **Local Engine Active:** Phi-3 is restricted to `GENERAL` intent queries "
            "(definitional or contextual) only. It lacks the parameter count to reliably synthesize "
            "`BENCHMARK` or `HYBRID` context chunks."
        )
    else:
        st.caption(
            "☁️ **Cloud Engine Active:** Gemini 1.5 Pro is enabled. Fully capable of "
            "handling `EXTRACT`, `BENCHMARK`, and `HYBRID` multi-source reasoning paths."
        )
        
    return selected_model