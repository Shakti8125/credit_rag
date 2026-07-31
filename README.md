# CreditRAG — Privacy-First Credit Risk Intelligence Platform

CreditRAG is a two-tier Retrieval-Augmented Generation (RAG) system for credit-risk analysts. An analyst uploads a confidential credit document (credit memo, financial statement, policy dossier) into a **local Streamlit app**, where all PII and bank names are masked **on-device** before anything leaves the machine. Masked queries are then answered either by a **local Phi-3 SLM** (fully offline) or by a **cloud FastAPI backend** that grounds Gemini on a Pinecone index of UAE/Basel regulatory corpora (CBUAE standards, Basel III, IFRS 9, market/operational/climate risk manuals).

**The core privacy contract:** the placeholder→original mapping (`[ORG_1]` → real bank name) never leaves the analyst's machine. The cloud only ever sees masked text and returns masked answers; unmasking happens locally.

---

## High-Level Architecture

```
┌────────────────────────── ANALYST MACHINE (local/) ──────────────────────────┐
│                                                                              │
│  Streamlit UI (local/app/main.py)                                            │
│    │                                                                         │
│    ├─ Upload → PrivacyPipeline (local/privacy/)                              │
│    │     Phase 1  Docling PDF → Markdown            (extractor.py)           │
│    │     Phase 4* Financial extraction / policy     (local/analysis/)        │
│    │              breaches / EWS on RAW text                                 │
│    │     Phase 2  PII masking: regex + spaCy NER +  (masker.py)              │
│    │              GCC bank dictionary                                        │
│    │     Phase 3  Egress firewall validation        (validator.py)           │
│    │                                                                         │
│    ├─ Masked text → MarkdownChunker → ephemeral FAISS index (local/rag/)     │
│    │                                                                         │
│    └─ Query → intent classifier (intent.py / slm/router.py)                  │
│          ├─ LOCAL path: Phi-3 GGUF via llama.cpp    (local/slm/)             │
│          └─ CLOUD path: masked payload over HTTPS ──────────────┐            │
│                                                                 │            │
│  Unmasking of answers + citations happens HERE, locally         │            │
└─────────────────────────────────────────────────────────────────┼────────────┘
                                                                  │ masked only
┌──────────────────────────── CLOUD TIER (cloud/) ────────────────▼────────────┐
│  FastAPI backend (cloud/backend/app/)                                        │
│    /query    RAG inference (EXTRACT / HYBRID / BENCHMARK / GENERAL)          │
│    /compare  Tier-2 multi-document comparison                                │
│    /ews      Tier-2 early-warning-signal deep scan                           │
│    /health   liveness probe                                                  │
│                                                                              │
│  Retrieval: Pinecone (regulatory corpus, ns "cbuae-manuals")                 │
│             + BM25 / FAISS hybrid + cross-encoder rerank                     │
│  Generation: Google Gemini (GEMINI_MODEL, default gemini-2.5-flash)          │
│  Secrets: AWS SSM Parameter Store (env-var fallback for local dev)           │
└──────────────────────────────────────────────────────────────────────────────┘
```

\* Phase 4 runs *before* masking so numeric metric values are intact; EWS excerpts are re-masked before they can ever be sent to the cloud.

---

## Repository Layout

```
credit-risk-rag_v_0.1/
├── local/          # Analyst-machine tier: Streamlit UI + privacy + local RAG + SLM
├── cloud/          # Cloud tier: FastAPI backend + corpus ingestion + infra stubs
├── shared/         # Code imported by BOTH tiers (no tier reaches into the other)
├── eval/           # Privacy, retrieval, and answer-quality evaluation suites
├── .env            # PINECONE_API_KEY, GEMINI_API_KEY, PINECONE_INDEX_NAME
└── .github/workflows/deploy.yml   # CI/CD stub (empty)
```

### `shared/` — cross-tier primitives

| File | Purpose |
|---|---|
| [paths.py](shared/paths.py) | Central path constants (project root, SLM model path, `.env` location). |
| [env.py](shared/env.py) | Single `.env` resolution point (`load_env` / `get_env`) used by both tiers. |
| [logging_config.py](shared/logging_config.py) | One logging setup for Streamlit and FastAPI tiers. |
| [masking.py](shared/masking.py) | `EntityRegistry` — thread-safe entity→placeholder registry with audit log — and `unmask_text()`. Framework-agnostic (no spaCy), so the cloud can unmask-format without heavy deps. |
| [chunking.py](shared/chunking.py) | `TextChunk` dataclass (common chunk shape) and `chunk_by_words()` word-window splitter used by cloud ingestion. |
| [hybrid.py](shared/hybrid.py) | Dependency-free `BM25Okapi` and `rrf_fuse()` (Reciprocal Rank Fusion) — the lexical half of hybrid retrieval in both tiers. |

### `local/` — the analyst-machine tier

**`local/app/` — Streamlit frontend**

| File | Purpose |
|---|---|
| [main.py](local/app/main.py) | Entry point. Page config, sidebar (upload, model toggle, security status), chat loop, intent dispatch to handlers. |
| [config.py](local/app/config.py) | API endpoint URLs (`CLOUD_API_BASE`, default `http://127.0.0.1:8000`), SLM model path, retrieval tuning constants (top-k, rerank pool, chunk cap). |
| [session.py](local/app/session.py) | Session-state defaults, incl. per-document mask dictionaries. |
| [document_pipeline.py](local/app/document_pipeline.py) | Upload → `PrivacyPipeline` → FAISS index wiring; `recompute_analytics()` re-runs Phase 4 when the doc type changes. |
| [intent.py](local/app/intent.py) | Fast keyword-based intent classification: EXTRACT / HYBRID / BENCHMARK / GENERAL / COMPARE / EWS. |
| [retrieval.py](local/app/retrieval.py) | Hybrid FAISS + BM25 retrieval over the uploaded doc, plus Pinecone regulatory retrieval, with cross-encoder reranking. |
| [masking_utils.py](local/app/masking_utils.py) | Snapshot-dict-based mask/unmask helpers: `mask_outbound_query()`, `unmask_response()`, `unmask_with_merged()` (for two-document compare). |
| [resources.py](local/app/resources.py) | `st.cache_resource` singletons: privacy pipeline, embedding model, Phi-3 engine, reranker, Pinecone retriever. |
| [ui_helpers.py](local/app/ui_helpers.py) / [styles.py](local/app/styles.py) | Streaming response rendering, intent chips, dark theme CSS. |

**`local/app/handlers/`** — one handler per execution path, all called from `main.py`:

| File | Path |
|---|---|
| [local_edge.py](local/app/handlers/local_edge.py) | EXTRACT/HYBRID/GENERAL answered fully offline by Phi-3 with local retrieval. |
| [cloud.py](local/app/handlers/cloud.py) | EXTRACT/HYBRID/BENCHMARK/GENERAL sent (masked) to cloud `/query`; unmasks answer + citations locally. |
| [compare.py](local/app/handlers/compare.py) | COMPARE → cloud `/compare` with both masked documents; unmasks with merged A+B dictionaries. |
| [ews.py](local/app/handlers/ews.py) | EWS → cloud `/ews` deep scan with masked doc + local EWS report. |

**`local/app/components/`** — UI panels: [upload_panel.py](local/app/components/upload_panel.py), [model_toggle.py](local/app/components/model_toggle.py) (local vs cloud), [chat.py](local/app/components/chat.py), [citations.py](local/app/components/citations.py), [masking_log.py](local/app/components/masking_log.py) (what was masked and why), [financial_profile.py](local/app/components/financial_profile.py), [policy_breach_panel.py](local/app/components/policy_breach_panel.py), [ews_panel.py](local/app/components/ews_panel.py), [compare_panel.py](local/app/components/compare_panel.py), [audit_trail.py](local/app/components/audit_trail.py), [path_indicator.py](local/app/components/path_indicator.py).

**`local/privacy/` — the masking pipeline**

| File | Purpose |
|---|---|
| [pipeline.py](local/privacy/pipeline.py) | `PrivacyPipeline` orchestrator — the four-phase flow described above. |
| [extractor.py](local/privacy/extractor.py) | IBM Docling PDF→Markdown extraction (tables, headers preserved). |
| [masker.py](local/privacy/masker.py) | `DocumentMasker`: bank dictionary + regex (emails/phones/SSNs/EINs) + spaCy NER (persons, orgs, locations). Preserves financial numbers, ratios, and domain terms (PD, LGD, DSCR…). |
| [bank_dictionary.py](local/privacy/bank_dictionary.py) | GCC/UAE bank-name list — highest-confidence masking targets; shared by masker and validator. |
| [validator.py](local/privacy/validator.py) | Egress firewall: regex sweep of the masked text for residual PII before any cloud transmission. |

**`local/analysis/` — on-device analytics (run on RAW text, pre-mask)**

| File | Purpose |
|---|---|
| [financial_extractor.py](local/analysis/financial_extractor.py) | Regex-based structured extraction → `FinancialProfile` (ratios, facility terms, P&L metrics) keyed by document type. |
| [policy_checker.py](local/analysis/policy_checker.py) | Compares the profile against CBUAE Circular 33/2023, Basel III, and generic underwriting thresholds → `BreachReport` (BREACH / WARNING / PASS). |
| [ews_detector.py](local/analysis/ews_detector.py) | Early-warning-signal scan (financial, qualitative, structural categories) with auditable verbatim excerpts → `EWSReport`. Excerpts are masked by the pipeline before any cloud use. |
| [audit_logger.py](local/analysis/audit_logger.py) | Per-session audit trail of every query cycle; PDF export via reportlab. |

**`local/rag/` — local retrieval**

| File | Purpose |
|---|---|
| [chunker.py](local/rag/chunker.py) | `MarkdownChunker`: header-aware splitting (2400 chars / 400 overlap) tuned for regulatory prose. |
| [large_chunker.py](local/rag/large_chunker.py) | 3600/600 variant for Tier-2 cloud tasks (compare, EWS). |
| [local_index.py](local/rag/local_index.py) | Ephemeral in-memory FAISS index over masked doc chunks (all-MiniLM-L6-v2 embeddings) + BM25, fused with RRF. Nothing written to disk. |
| [pinecone_index.py](local/rag/pinecone_index.py) | Pinecone regulatory-corpus retriever for the local Phi-3 path (GENERAL/HYBRID). |
| [reranker.py](local/rag/reranker.py) | `cross-encoder/ms-marco-MiniLM-L-6-v2` reranker shared by both paths. |

**`local/slm/` — offline model**

| File | Purpose |
|---|---|
| [inference.py](local/slm/inference.py) | llama-cpp-python wrapper for the quantized GGUF binary. |
| [router.py](local/slm/router.py) | SLM-based intent router (EXTRACT/BENCHMARK/HYBRID/GENERAL) with rule-based and fallback guards. |
| [prompt_templates.py](local/slm/prompt_templates.py) | Phi-3 chat-format prompts for routing and answering. |
| `models/phi3-basel-q4km.gguf` | Phi-3 Mini, Q4_K_M quantized (~2.2 GB). |

### `cloud/` — the cloud tier

**`cloud/backend/app/routes/`**

| Route | Purpose |
|---|---|
| [query.py](cloud/backend/app/routes/query.py) | `POST /query` — main RAG endpoint. Pinecone retrieval for BENCHMARK/GENERAL/HYBRID; uploaded-doc chunk selection (BM25 fallback) for EXTRACT/HYBRID; honors `doc_is_prechunked` so client-retrieved chunks are used as-is; prompts Gemini; returns **masked** answer + citations. |
| [compare.py](cloud/backend/app/routes/compare.py) | `POST /compare` — Tier-2 side-by-side analysis of two masked documents. |
| [ews.py](cloud/backend/app/routes/ews.py) | `POST /ews` — Tier-2 LLM deep-dive over the local EWS report + doc chunks. |
| [health.py](cloud/backend/app/routes/health.py) | `GET /health` — static liveness probe for load balancers / warmup rules. |

**`cloud/backend/app/services/`**

| File | Purpose |
|---|---|
| [engines.py](cloud/backend/app/services/engines.py) | Process-wide singletons for the generation and retrieval services (built once, shared by all routes). |
| [generation.py](cloud/backend/app/services/generation.py) | Gemini wrapper via `google-genai`; model from `GEMINI_MODEL` (default `gemini-2.5-flash`). |
| [retrieval.py](cloud/backend/app/services/retrieval.py) | `PineconeRetrievalService`: Pinecone inference embedding (`multilingual-e5-large`) recall pool → BM25 re-score → cross-encoder rerank → `Citation` list. |
| [doc_search.py](cloud/backend/app/services/doc_search.py) | Ephemeral uploaded-doc search for `/compare` and `/ews`: chunk + FAISS + BM25 + RRF + rerank, with a lazy singleton embedder. |
| [prompt_builder.py](cloud/backend/app/services/prompt_builder.py) | All prompt templates (grounding, audit, hybrid, preformatted) with grounding/citation/placeholder-preservation contracts. |
| [doc_injector.py](cloud/backend/app/services/doc_injector.py) | Token gatekeeper (tiktoken) that truncates full-document payloads to fit the model context. |
| [secrets.py](cloud/backend/app/services/secrets.py) | Secret resolution: local env var first, then AWS SSM Parameter Store (`ap-south-1`), with a warm-start cache. |
| [dynamo.py](cloud/backend/app/services/dynamo.py) | Optional DynamoDB telemetry logger (intent, path, latency only — never payload text). |

**Corpus ingestion (run once, offline)**

| File | Purpose |
|---|---|
| [1_extract_and_chunk.py](cloud/backend/1_extract_and_chunk.py) | Docling-extracts every PDF in `base_documents/` (CBUAE, Basel, IFRS 9, market/operational/climate risk), chunks by words (200/40) → `chunks_staging.json`. |
| [2_embed_and_upload.py](cloud/backend/2_embed_and_upload.py) | Embeds staged chunks with Pinecone inference (`multilingual-e5-large`) and upserts into namespace `cbuae-manuals`. |

**Other cloud files:** [Dockerfile](cloud/backend/Dockerfile) (builds from project root so `shared/` is in context; pre-downloads embedding + reranker models at build time; exposes both uvicorn and a Mangum Lambda `handler`), [cloud/infra/template.yaml](cloud/infra/template.yaml) and [samconfig.toml](cloud/infra/samconfig.toml) (SAM stubs — currently empty; see [DEPLOYMENT_AWS.md](DEPLOYMENT_AWS.md)).

### `eval/` — evaluation harness

Industry-grade evaluation: configurable pass/fail gates ([thresholds.json](eval/thresholds.json)), machine-readable reports + run history, baseline regression tracking ([baseline.json](eval/baseline.json)), and a one-command orchestrator with a scorecard. See [eval/README.md](eval/README.md). The suites:

- [privacy_eval.py](eval/privacy_eval.py) — bank-name/PII leak rate (gate 0 %), financial-value preservation (gate 100 %), egress-firewall catch rate, mask→unmask round-trip fidelity.
- [adversarial_eval.py](eval/adversarial_eval.py) — masking robustness under attack (case variants, filenames, obfuscated/homoglyph/zero-width PII); a leak counts only when masker **and** egress firewall both miss. In-spec forms gate at 0; obfuscations are the tracked hardening metric.
- [retrieval_eval.py](eval/retrieval_eval.py) — Hit@1 / Hit@3 / MRR / nDCG@5 for BM25 vs dense vs hybrid (RRF) on a labeled mini-corpus.
- [ragas_eval.py](eval/ragas_eval.py) — live-backend answer quality on [golden_set.json](eval/golden_set.json) (reference answers included): grounding/citation/leak/latency heuristics, optional LLM-as-judge rubric scoring via [judge.py](eval/judge.py) (`--judge`), optional RAGAS.
- [run_all.py](eval/run_all.py) — runs everything, writes `eval/reports/scorecard.{md,json}`, and updates the baseline with `--update-baseline`.

Offline suites (privacy, adversarial, retrieval) run in CI on every PR via [.github/workflows/eval.yml](.github/workflows/eval.yml).

---

## How the Pieces Interact

**1. Document upload** (all on-device)

`upload_panel` → `document_pipeline.process_uploaded_document()` → `PrivacyPipeline.process_document()`:
Docling extraction → financial extraction + policy check + EWS on raw text → masking (registry assigns `[ORG_n]`/`[PERSON_n]` placeholders) → egress validation. The masked Markdown is chunked (`MarkdownChunker`) and indexed into an ephemeral FAISS + BM25 index in session state. The mask dictionary snapshot is stored in session state only.

**2. Query — local path** (offline)

`classify_intent()` → `handle_local_edge()`: hybrid retrieval over the doc index (EXTRACT) and/or Pinecone (GENERAL/HYBRID), cross-encoder rerank, Phi-3 generates, answer is unmasked and rendered with citations. No network egress except the optional Pinecone lookup.

**3. Query — cloud path**

`handle_cloud()` masks the typed query itself (`mask_outbound_query`), retrieves + reranks doc chunks locally, and POSTs `{query, intent, doc_text, doc_is_prechunked, doc_type}` to `/query`. **The mask dictionary is deliberately not sent.** The backend retrieves regulatory context from Pinecone, builds the intent-specific prompt, calls Gemini, and returns a masked answer + citations. The frontend unmasks both locally and records the exchange in the audit trail.

**4. Compare / EWS (Tier 2, cloud-only)**

`handle_compare()` sends both masked documents to `/compare` (unmasking uses the merged A+B dictionaries); `handle_ews()` sends the masked doc plus the locally computed (and re-masked) EWS report to `/ews` for LLM synthesis. Both routes use `doc_search.py` for ephemeral hybrid retrieval with the large chunker sizes.

---

## Running Locally

**Prerequisites:** Python 3.11, a `.env` at the project root:

```
PINECONE_API_KEY=...
GEMINI_API_KEY=...
PINECONE_INDEX_NAME=creditrag
```

**One-time corpus ingestion** (populates Pinecone from `base_documents/`):

```bash
cd cloud/backend
python 1_extract_and_chunk.py
python 2_embed_and_upload.py
```

**Cloud backend:**

```bash
pip install -r cloud/backend/requirements.txt
cd cloud/backend
uvicorn app.main:app --port 8000
# or: docker build -f cloud/backend/Dockerfile -t creditrag-backend .   (from project root)
```

**Local tier:**

```bash
pip install -r local/requirements.txt
python -m spacy download en_core_web_lg
streamlit run local/app/main.py
```

The frontend targets `http://127.0.0.1:8000` by default; set `CLOUD_API_BASE` to point at a deployed backend.

**Evals:**

```bash
python eval/run_all.py                                                # offline: privacy + adversarial + retrieval
python eval/run_all.py --api http://127.0.0.1:8000 --judge --sleep 13 # + live e2e with LLM judge (backend running)
python eval/run_all.py --update-baseline                              # bless current metrics for regression tracking
```

---

## Deployment

The cloud tier is the only deployable component — the local tier must stay on the analyst's machine by design.

It runs as a **Lambda container behind a Function URL** in `ap-south-1`: 2048 MB / 300 s, image built and pushed to ECR by [.github/workflows/deploy.yml](.github/workflows/deploy.yml) on every push to `main` (GitHub OIDC, no long-lived AWS keys), gated on the privacy and adversarial masking evals. Secrets resolve from SSM Parameter Store; telemetry goes to DynamoDB. Cold start ~25 s, warm `/query` 8–15 s, ~$0.30/month.

Requests to `/query`, `/compare` and `/ews` require an `X-API-Key` header matching the `CREDITRAG_API_KEY` secret ([auth.py](cloud/backend/app/auth.py)); the check self-disables when the secret is unset, so local development is unchanged. `/health` stays open for probes.

Point the frontend at a deployment by setting `CLOUD_API_BASE` and `CLOUD_API_KEY` in `.env`.

See **[DEPLOYMENT_AWS.md](DEPLOYMENT_AWS.md)** for the full runbook — including four traps that will otherwise cost you an afternoon: the base image must be `python:3.12` (the 3.11 runtime is glibc 2.26 and cannot install modern `manylinux_2_28` wheels), pip must be upgraded before installing, a public Function URL needs **two** IAM statements since October 2025, and `.dockerignore` is mandatory because the build context is otherwise 3.8 GB.
