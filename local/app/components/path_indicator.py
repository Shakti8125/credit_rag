import streamlit as st

def render_path_indicator(execution_path: str = None):
    """
    Renders a visual routing map showing the lifecycle of the active query
    across the local client edge and the cloud orchestration layer.
    """
    # Fallback to session state if path parameter is not explicitly provided
    if not execution_path:
        execution_path = st.session_state.get("last_execution_path", "Idle")

    st.markdown("### 🗺️ RAG Routing Path Tracker")

    if execution_path == "Idle":
        st.info("System standby. Submit a query to track execution pathing.")
        return

    # Visual container for path routing breakdown
    with st.container():
        if execution_path.lower() in ["local", "local_tier", "edge"]:
            st.markdown("#### **Current Route:** 🟢 Local Processing Path")
            st.caption("Query resolved entirely at the local workstation tier to ensure absolute data isolation.")
            
            # Step breakdown for local route
            st.status("1. Input Received", state="complete")
            st.status("2. Local Semantic Embedding Match (FAISS/Chroma)", state="complete")
            st.status("3. Edge Model Generation (Local LLM Core)", state="complete")
            
        elif execution_path.lower() in ["cloud", "cloud_hybrid", "orchestrated"]:
            st.markdown("#### **Current Route:** ⚡ Cloud Hybrid Orchestration Path")
            st.caption("Query scrubbed locally and dispatched to the FastAPI container layer for deep semantic analysis.")
            
            # Step breakdown for cloud route
            st.status("1. Local Document Ingestion & PII Scrubbing", state="complete")
            st.status("2. Cloud Routing Node Interception (`/query` Router)", state="complete")
            st.status("3. Orchestrator Service Analysis & Parameter Extraction", state="complete")
            st.status("4. High-Context Generation (Gemini 1.5 Pro Engine)", state="complete")
            
        else:
            # Fallback pathing representation for ambiguous or multi-hop evaluation loops
            st.markdown(f"#### **Current Route:** 🌀 Custom Pipeline Variant ({execution_path})")
            st.warning("Dynamic routing logic executed an alternative multi-path or verification loop.")
            
    st.markdown("---")