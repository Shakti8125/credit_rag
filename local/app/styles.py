"""Global CSS for the CreditRAG Streamlit theme."""

import streamlit as st


def apply_theme() -> None:
    st.markdown("""
<style>

@import url(
'https://fonts.googleapis.com/css2?family=Inter:wght@400;600;700'
);


.header-title {

font-family:'Inter',sans-serif;
font-weight:700;

}


.header-sub {

font-family:'Inter',sans-serif;

letter-spacing:.08em;

font-size:.75rem;

}


</style>
""", unsafe_allow_html=True)
