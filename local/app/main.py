import sys
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from shared.env import load_env
from shared.logging_config import setup_logging, get_logger

load_env()
setup_logging()

import time
import streamlit as st
import requests

logger = get_logger(__name__)

from local.app.styles import apply_theme
from local.app.session import init_session_state
from local.app.document_pipeline import process_uploaded_document, recompute_analytics
from local.app.intent import classify_intent
from local.app.ui_helpers import intent_chip
from local.app.handlers.compare import handle_compare
from local.app.handlers.ews import handle_ews
from local.app.handlers.local_edge import handle_local_edge
from local.app.handlers.cloud import handle_cloud

from components.upload_panel        import render_upload_panel
from components.model_toggle        import render_model_toggle
from components.chat                import render_chat_interface
from components.masking_log         import render_masking_log
from components.financial_profile   import render_financial_profile
from components.policy_breach_panel import render_policy_breach_panel
from components.audit_trail         import render_audit_trail

# ---------------------------------------------------------------------------
# PAGE CONFIG — must be first Streamlit call
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="CreditRAG · Risk Intelligence",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded",
)

apply_theme()
init_session_state()

# ---------------------------------------------------------------------------
# SIDEBAR
# ---------------------------------------------------------------------------
with st.sidebar:
    # Brand mark
    st.markdown("""
    <div style="padding: 12px 0 20px 0;">
        <div style="font-size:1.2rem;font-weight:700;color:#22d3ee;letter-spacing:-0.02em;">
            🛡️ CreditRAG
        </div>
        <div style="font-size:0.65rem;color:#3a6a8a;text-transform:uppercase;letter-spacing:0.12em;margin-top:2px;">
            Risk Intelligence Platform
        </div>
    </div>
    """, unsafe_allow_html=True)

    primary_file, secondary_file = render_upload_panel()
    selected_model               = render_model_toggle()

    # Privacy status
    st.markdown('<div class="panel-section">Security Status</div>', unsafe_allow_html=True)
    st.markdown("""
    <div style="background:#071a0e;border:1px solid #1b5e20;border-radius:8px;padding:10px 14px;">
        <div style="font-size:0.72rem;font-weight:600;color:#66bb6a;text-transform:uppercase;letter-spacing:0.08em;">
            🔒 Local Anonymisation Active
        </div>
        <div style="font-size:0.7rem;color:#388e3c;margin-top:4px;line-height:1.5;">
            All PII masked on-device before cloud dispatch.
        </div>
    </div>
    """, unsafe_allow_html=True)

    mode = st.session_state.get("analysis_mode", "Standard")

    # Document Type changed after upload → re-run analytics (extraction,
    # policy check, EWS) on the stored raw text with the new type. Fast:
    # no re-extraction, re-masking, or re-indexing.
    _current_doc_type = st.session_state.get("document_type")
    if (
        st.session_state.get("doc_raw_text")
        and st.session_state.get("doc_type_used") not in (None, _current_doc_type)
    ):
        with st.spinner("Re-running analysis for new document type…"):
            if recompute_analytics(_current_doc_type):
                st.success("✅ Analysis updated for new document type")
                st.rerun()

    # Process primary document — guard compares the RAW filename;
    # last_uploaded_file holds the MASKED name and would never match,
    # reprocessing the document on every rerun.
    if primary_file and st.session_state.get("last_uploaded_raw_name") != primary_file.name:
        with st.spinner(f"Processing {primary_file.name}…"):
            ok = process_uploaded_document(primary_file, slot="primary")
        if ok:
            idx    = st.session_state.get("doc_faiss_index")
            breach = st.session_state.get("breach_report")
            if idx and idx.is_built:
                st.success(f"✅ {idx.chunk_count} chunks indexed")
            if breach:
                if breach.breach_count:
                    st.error(f"🔴 {breach.breach_count} policy breach(es)")
                elif breach.warning_count:
                    st.warning(f"🟡 {breach.warning_count} warning(s)")
                else:
                    st.success("🟢 All policy checks passed")
            st.rerun()

    # Process secondary document (Compare mode)
    if (
        mode == "Compare Two Documents"
        and secondary_file
        and st.session_state.get("doc_b_raw_name") != secondary_file.name
    ):
        with st.spinner(f"Processing {secondary_file.name}…"):
            ok = process_uploaded_document(secondary_file, slot="secondary")
        if ok:
            idx = st.session_state.get("doc_b_faiss_index")
            if idx and idx.is_built:
                st.success(f"✅ Doc B: {idx.chunk_count} chunks")
            st.rerun()


# ---------------------------------------------------------------------------
# MAIN LAYOUT
# ---------------------------------------------------------------------------

# Header banner
doc_name  = st.session_state.get("last_uploaded_file") or "No document loaded"
mode_disp = st.session_state.get("analysis_mode", "Standard")
st.markdown(f"""
<div class="header-banner">
    <div>
        <div class="header-title">Risk Intelligence Terminal</div>
        <div class="header-sub">Credit Risk RAG · Regulatory Intelligence · Policy Compliance</div>
    </div>
    <div style="text-align:right;">
        <div class="header-badge">📄 {doc_name}</div>
        <div style="font-size:0.65rem;color:#3a6a8a;margin-top:4px;text-transform:uppercase;letter-spacing:0.08em;">
            Mode: {mode_disp}
        </div>
    </div>
</div>
""", unsafe_allow_html=True)

col1, col2 = st.columns([2, 1], gap="medium")

# ── RIGHT PANEL ───────────────────────────────────────────────────────────
with col2:

    # Execution status
    path = st.session_state["last_execution_path"]
    if path == "Idle":
        pill_cls, pill_lbl = "pill-idle", "● IDLE"
    elif "Local" in path:
        pill_cls, pill_lbl = "pill-active", "● LOCAL EDGE"
    else:
        pill_cls, pill_lbl = "pill-active", "● CLOUD ACTIVE"

    col_path, col_lat = st.columns([2, 1])
    with col_path:
        st.markdown(
            f'<div class="panel-section">Execution Path</div>'
            f'<span class="status-pill {pill_cls}">{pill_lbl}</span>'
            f'<div style="font-size:0.72rem;color:#4a7a9b;margin-top:6px;">{path}</div>',
            unsafe_allow_html=True,
        )
    with col_lat:
        if st.session_state["last_execution_time"]:
            st.metric("⏱ Latency", f"{st.session_state['last_execution_time']}s")

    st.markdown("---")

    if st.session_state["doc_text"]:
        st.markdown('<div class="panel-section">Policy & Risk Signals</div>', unsafe_allow_html=True)
        render_policy_breach_panel(st.session_state.get("breach_report"))
        render_financial_profile(st.session_state.get("financial_profile"))

        st.markdown('<div class="panel-section">Privacy Audit</div>', unsafe_allow_html=True)
        render_masking_log(
            st.session_state["mask_dictionary"]
        )

    st.markdown('<div class="panel-section">Session Audit Trail</div>', unsafe_allow_html=True)
    # FIX: use `or "session"` to guard None value — .get() default only applies
    # when the key is absent, not when it holds None
    safe_doc_name = st.session_state.get("last_uploaded_file") or "session"
    render_audit_trail(
        st.session_state.get("audit_trail", []),
        doc_filename=safe_doc_name,
    )

# ── LEFT PANEL (main chat) ────────────────────────────────────────────────
with col1:

    # Mode-specific info banners
    if mode_disp == "Compare Two Documents":
        doc_a = st.session_state.get("last_uploaded_file") or "—"
        doc_b = st.session_state.get("doc_b_label") or "not loaded"
        st.markdown(f"""
        <div style="background:#001a1a;border:1px solid #006064;border-radius:10px;padding:12px 16px;margin-bottom:12px;">
            <div style="font-size:0.72rem;font-weight:700;color:#4dd0e1;text-transform:uppercase;letter-spacing:0.08em;">
                🔄 Comparison Mode Active
            </div>
            <div style="font-size:0.8rem;color:#80deea;margin-top:6px;">
                <b>A:</b> {doc_a} &nbsp;·&nbsp; <b>B:</b> {doc_b}
            </div>
        </div>
        """, unsafe_allow_html=True)

    elif mode_disp == "Early Warning Scan":
        st.markdown("""
        <div style="background:#1a0505;border:1px solid #7f0000;border-radius:10px;padding:12px 16px;margin-bottom:12px;">
            <div style="font-size:0.72rem;font-weight:700;color:#ef9a9a;text-transform:uppercase;letter-spacing:0.08em;">
                ⚡ Early Warning Scan Mode
            </div>
            <div style="font-size:0.8rem;color:#ef9a9a;margin-top:4px;opacity:0.8;">
                Queries trigger deep Gemini EWS analysis. Upload a document first.
            </div>
        </div>
        """, unsafe_allow_html=True)

    # Last intent indicator
    last_intent = st.session_state.get("last_intent", "")
    if last_intent:
        st.markdown(
            f'<div style="margin-bottom:8px;font-size:0.75rem;color:#4a7a9b;">Last intent: '
            f'{intent_chip(last_intent)}</div>',
            unsafe_allow_html=True,
        )

    prompt = render_chat_interface()

    if prompt:
        document_attached = bool(st.session_state["doc_text"])
        intent = classify_intent(prompt, document_attached, selected_model)

        # Block the local tier for Tier 2 intents
        if intent in ("COMPARE", "EWS") and selected_model == "local":
            st.warning(
                "⚠️ **Compare** and **EWS** modes are cloud-only. "
                "Switch to **Cloud** in the sidebar."
            )
            st.stop()

        try:
            with st.chat_message("assistant"):
                with st.spinner("Analysing…"):
                    start_time = time.time()

                    if intent == "COMPARE":
                        answer_text = handle_compare(prompt)
                    elif intent == "EWS":
                        answer_text = handle_ews(prompt)
                    elif selected_model == "local":
                        answer_text = handle_local_edge(prompt, intent)
                    else:
                        answer_text = handle_cloud(prompt, intent, document_attached)

                    # ── TIMING + AUDIT TRAIL ──────────────────────────
                    elapsed = round(time.time() - start_time, 2)
                    st.session_state["last_execution_time"] = elapsed
                    st.caption(f"⏱️ **{elapsed}s** · Intent: {intent}")

                    try:
                        from local.analysis.audit_logger import build_entry
                        breach = st.session_state.get("breach_report")
                        ews    = st.session_state.get("ews_report")
                        breach_sum = " | ".join(filter(None, [
                            breach.summary() if breach else None,
                            ews.summary()    if ews    else None,
                        ])) or None

                        st.session_state["audit_trail"].append(build_entry(
                            query          = prompt,
                            intent         = intent,
                            answer         = answer_text,
                            citations      = st.session_state["last_citations"],
                            execution_path = st.session_state["last_execution_path"],
                            elapsed_sec    = elapsed,
                            masked_count   = len(st.session_state.get("mask_dictionary", {})),
                            doc_filename   = st.session_state.get("last_uploaded_file"),
                            breach_summary = breach_sum,
                        ))
                    except Exception as ae:
                        logger.warning("Audit trail: %s", ae)

                    st.session_state["messages"].append({
                        "role":      "assistant",
                        "content":   answer_text,
                        "citations": st.session_state["last_citations"],
                        "time":      elapsed,
                    })
                    st.rerun()

        except requests.exceptions.ConnectionError:
            st.error(
                "🚨 Connection refused — ensure FastAPI is running: "
                "`uvicorn app.main:app --reload --port 8000`"
            )
