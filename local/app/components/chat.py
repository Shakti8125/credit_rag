import streamlit as st

def render_chat_interface():
    """
    Renders the principal chat history pane and processes user query inputs
    for the Credit Risk RAG analytical engine.
    """
    st.markdown("### 💬 Credit Risk Policy & Analysis Workspace")
    
    # Initialize message log array in state memory if missing
    if "messages" not in st.session_state:
        st.session_state["messages"] = [
            {
                "role": "assistant",
                "content": "Welcome to the Credit Risk RAG system. Upload an internal credit memo or policy framework in the sidebar to begin context-grounded analysis."
            }
        ]

    # Container setup to keep the conversational log visually distinct
    chat_container = st.container()

    with chat_container:
        for message in st.session_state["messages"]:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])
                
                # Check if metadata elements like localized citations are appended
                if "citations" in message and message["role"] == "assistant":
                    with st.expander("🔍 View Grounded References & Source Chunks"):
                        for idx, citation in enumerate(message["citations"], 1):
                            st.caption(f"**Reference [{idx}]:** Source Page {citation.get('page', 'N/A')}")
                            st.info(citation.get("text", ""))

    # Interactive input anchor positioned at the base of the layout
    if prompt := st.chat_input("Ask a question regarding credit limits, counterparty metrics, or regulatory caps..."):
        
        # Immediately render the analyst query to keep interface feedback real-time
        st.session_state["messages"].append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)
            
        return prompt

    return None