"""
local/analysis/audit_logger.py

Per-session audit trail for every query-response cycle.

Captures:
  - Timestamp
  - User query
  - Classified intent
  - Retrieved chunks (source, section, score)
  - LLM response (unmasked)
  - Masked entities count
  - Execution path and latency
  - Policy breach summary (if a document was attached)

Stored in st.session_state["audit_trail"] as a list of AuditEntry dicts.
Exportable as a formatted PDF via export_audit_pdf().
"""

import logging
from datetime import datetime
from typing import List, Dict, Any, Optional

logger = logging.getLogger(__name__)


def _now() -> str:
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def build_entry(
    query:          str,
    intent:         str,
    answer:         str,
    citations:      List[Dict[str, Any]],
    execution_path: str,
    elapsed_sec:    float,
    masked_count:   int,
    doc_filename:   Optional[str] = None,
    breach_summary: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Builds a single audit entry dict. Call this after every successful
    query-response cycle and append to st.session_state["audit_trail"].
    """
    entry = {
        "timestamp":      _now(),
        "query":          query,
        "intent":         intent,
        "answer":         answer,
        "execution_path": execution_path,
        "elapsed_sec":    elapsed_sec,
        "masked_count":   masked_count,
        "doc_filename":   doc_filename or "—",
        "breach_summary": breach_summary or "—",
        "citations": [
            {
                "source":  c.get("source",  c.get("section", "Unknown")),
                "section": c.get("section", "—"),
                "page":    c.get("page",    "—"),
                "score":   round(float(c.get("score", c.get("rerank_score", 0)) or 0), 3),
                "preview": (c.get("text", "") or "")[:200],
            }
            for c in (citations or [])
        ],
    }
    logger.info(
        "Audit entry: intent=%s path=%s elapsed=%.2fs citations=%d",
        intent, execution_path, elapsed_sec, len(entry["citations"]),
    )
    return entry


def export_audit_pdf(audit_trail: List[Dict[str, Any]], doc_filename: str = "document") -> bytes:
    """
    Renders the audit trail as a PDF and returns the raw bytes.
    Uses reportlab (already available via docling dependencies).

    Falls back to a plain UTF-8 text export if reportlab is not installed.
    """
    try:
        from reportlab.lib.pagesizes import A4
        from reportlab.lib.styles   import getSampleStyleSheet, ParagraphStyle
        from reportlab.lib.units    import cm
        from reportlab.lib           import colors
        from reportlab.platypus      import (
            SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, HRFlowable
        )
        import io as _io

        buf    = _io.BytesIO()
        doc    = SimpleDocTemplate(buf, pagesize=A4,
                                   leftMargin=2*cm, rightMargin=2*cm,
                                   topMargin=2*cm,  bottomMargin=2*cm)
        styles = getSampleStyleSheet()

        title_style = ParagraphStyle(
            "Title2", parent=styles["Title"],
            fontSize=16, spaceAfter=6
        )
        h2_style = ParagraphStyle(
            "H2", parent=styles["Heading2"],
            fontSize=11, textColor=colors.HexColor("#1a3c5e"), spaceAfter=4
        )
        body_style = ParagraphStyle(
            "Body2", parent=styles["Normal"],
            fontSize=9, leading=13, spaceAfter=3
        )
        caption_style = ParagraphStyle(
            "Caption", parent=styles["Normal"],
            fontSize=8, textColor=colors.grey, spaceAfter=6
        )

        story = [
            Paragraph("Credit Risk RAG — Session Audit Trail", title_style),
            Paragraph(
                f"Document: <b>{doc_filename}</b> &nbsp;|&nbsp; "
                f"Generated: {_now()} &nbsp;|&nbsp; "
                f"Entries: {len(audit_trail)}",
                caption_style
            ),
            HRFlowable(width="100%", thickness=1, color=colors.HexColor("#1a3c5e")),
            Spacer(1, 0.3*cm),
        ]

        for i, entry in enumerate(audit_trail, 1):
            story.append(Paragraph(f"Entry {i} — {entry['timestamp']}", h2_style))

            meta_data = [
                ["Intent",         entry.get("intent", "—")],
                ["Execution Path", entry.get("execution_path", "—")],
                ["Latency",        f"{entry.get('elapsed_sec', 0):.2f}s"],
                ["Masked Entities",str(entry.get("masked_count", 0))],
                ["Document",       entry.get("doc_filename", "—")],
                ["Policy Status",  entry.get("breach_summary", "—")],
            ]
            meta_table = Table(meta_data, colWidths=[4*cm, 13*cm])
            meta_table.setStyle(TableStyle([
                ("FONTSIZE",       (0, 0), (-1, -1), 8),
                ("FONTNAME",       (0, 0), (0, -1),  "Helvetica-Bold"),
                ("TEXTCOLOR",      (0, 0), (0, -1),  colors.HexColor("#1a3c5e")),
                ("ROWBACKGROUNDS", (0, 0), (-1, -1), [colors.HexColor("#f5f8fc"), colors.white]),
                ("GRID",           (0, 0), (-1, -1), 0.25, colors.lightgrey),
                ("TOPPADDING",     (0, 0), (-1, -1), 3),
                ("BOTTOMPADDING",  (0, 0), (-1, -1), 3),
            ]))
            story.extend([meta_table, Spacer(1, 0.2*cm)])

            story.append(Paragraph("<b>Query:</b>", body_style))
            story.append(Paragraph(entry.get("query", ""), body_style))
            story.append(Spacer(1, 0.15*cm))

            story.append(Paragraph("<b>Answer:</b>", body_style))
            answer_text = (entry.get("answer", "") or "").replace("<", "&lt;").replace(">", "&gt;")
            story.append(Paragraph(answer_text[:1500] + ("…" if len(answer_text) > 1500 else ""), body_style))
            story.append(Spacer(1, 0.15*cm))

            cits = entry.get("citations", [])
            if cits:
                story.append(Paragraph(f"<b>Retrieved Chunks ({len(cits)}):</b>", body_style))
                cit_data = [["#", "Source / Section", "Page", "Score", "Preview"]]
                for j, c in enumerate(cits, 1):
                    preview = (c.get("preview", "") or "")[:80].replace("<","&lt;")
                    cit_data.append([
                        str(j),
                        f"{c.get('source','—')} / {c.get('section','—')}"[:40],
                        str(c.get("page", "—")),
                        str(c.get("score", "—")),
                        preview,
                    ])
                cit_table = Table(cit_data, colWidths=[0.5*cm, 5*cm, 1.5*cm, 1.5*cm, 8.5*cm])
                cit_table.setStyle(TableStyle([
                    ("FONTSIZE",      (0, 0), (-1, -1), 7),
                    ("FONTNAME",      (0, 0), (-1, 0),  "Helvetica-Bold"),
                    ("BACKGROUND",    (0, 0), (-1, 0),  colors.HexColor("#1a3c5e")),
                    ("TEXTCOLOR",     (0, 0), (-1, 0),  colors.white),
                    ("ROWBACKGROUNDS",(0, 1), (-1, -1), [colors.HexColor("#f5f8fc"), colors.white]),
                    ("GRID",          (0, 0), (-1, -1), 0.25, colors.lightgrey),
                    ("TOPPADDING",    (0, 0), (-1, -1), 2),
                    ("BOTTOMPADDING", (0, 0), (-1, -1), 2),
                    ("WORDWRAP",      (4, 1), (4, -1),  True),
                ]))
                story.extend([cit_table, Spacer(1, 0.15*cm)])

            story.extend([
                HRFlowable(width="100%", thickness=0.5, color=colors.lightgrey),
                Spacer(1, 0.2*cm),
            ])

        doc.build(story)
        return buf.getvalue()

    except ImportError:
        logger.warning("reportlab not installed — exporting audit trail as plain text.")
        return _export_plain_text(audit_trail)


def _export_plain_text(audit_trail: List[Dict[str, Any]]) -> bytes:
    lines = ["Credit Risk RAG — Session Audit Trail", "=" * 60, ""]
    for i, e in enumerate(audit_trail, 1):
        lines += [
            f"Entry {i}  [{e.get('timestamp','—')}]",
            f"  Intent        : {e.get('intent','—')}",
            f"  Path          : {e.get('execution_path','—')}",
            f"  Latency       : {e.get('elapsed_sec',0):.2f}s",
            f"  Masked        : {e.get('masked_count',0)} entities",
            f"  Policy Status : {e.get('breach_summary','—')}",
            f"  Query  : {e.get('query','')}",
            f"  Answer : {(e.get('answer','') or '')[:500]}",
            "",
        ]
        for j, c in enumerate(e.get("citations", []), 1):
            lines.append(
                f"    Chunk {j}: {c.get('source','—')} | {c.get('section','—')} "
                f"| p.{c.get('page','—')} | score {c.get('score','—')}"
            )
        lines += ["-" * 60, ""]
    return "\n".join(lines).encode("utf-8")
