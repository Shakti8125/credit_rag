import streamlit as st
from typing import Dict


def render_masking_log(masking_log: Dict[str, str]):

    with st.expander(
        "🛡️ Masking Audit Log",
        expanded=False
    ):

        st.caption(
            "Only masked entities are displayed. "
            "Financial values are never logged."
        )

        if not masking_log:
            st.info("No entities masked.")
            return


        st.markdown("### Mask Dictionary")

        for masked, original in masking_log.items():

            col1, col2 = st.columns(2)

            with col1:
                st.code(masked)

            with col2:
                st.write(original)