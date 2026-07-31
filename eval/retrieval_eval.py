"""
eval/retrieval_eval.py — Retrieval quality: BM25 vs dense vs hybrid (RRF).

Uses a small labeled corpus of regulatory-flavoured passages with queries
whose correct source passage is known. Reports Hit@1, Hit@3, MRR, and
nDCG@5 for each retrieval mode so the hybrid fusion can be justified (or
tuned) with numbers instead of vibes.

The BM25 and hybrid-fusion paths are dependency-free (shared/hybrid.py).
Dense scoring runs only if sentence-transformers is installed; otherwise
the dense/hybrid rows are skipped and BM25 is still evaluated (hybrid
gates auto-skip when the metrics are absent).

Results gate against eval/thresholds.json (suite "retrieval") and write
eval/reports/retrieval_latest.json.

Run from the project root:
    python eval/retrieval_eval.py
"""

import math
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from shared.hybrid import BM25Okapi, rrf_fuse
from eval.common import build_gates, print_gates, warn_regressions, write_report, exit_code

# ---------------------------------------------------------------------------
# Labeled mini-corpus: passages mimic the CBUAE/Basel/IFRS9 knowledge base
# ---------------------------------------------------------------------------

CORPUS = [
    # 0
    "Basel III Pillar 1 requires banks to maintain a minimum total capital "
    "adequacy ratio of 10.5% of risk-weighted assets including the capital "
    "conservation buffer, with Common Equity Tier 1 of at least 7%.",
    # 1
    "Under IFRS 9, financial assets move to Stage 2 when a significant "
    "increase in credit risk (SICR) has occurred since initial recognition, "
    "requiring lifetime expected credit losses to be recognised.",
    # 2
    "The CBUAE Model Management Guidelines require banks to validate rating "
    "models annually, covering discriminatory power, calibration accuracy, "
    "and population stability of the modelling data.",
    # 3
    "Debt service coverage ratio (DSCR) measures cash flow available to "
    "service debt obligations; credit policy typically requires a minimum "
    "DSCR of 1.25x for corporate term facilities.",
    # 4
    "Loan-to-value limits for commercial real estate exposures are capped "
    "at 75% under CBUAE mortgage regulations, with lower caps applying to "
    "subsequent investment properties.",
    # 5
    "Expected credit loss under IFRS 9 is computed as PD x LGD x EAD, "
    "discounted at the effective interest rate, with forward-looking "
    "macroeconomic scenarios probability-weighted.",
    # 6
    "The population stability index (PSI) compares score distributions "
    "between development and recent application samples; a PSI above 0.25 "
    "indicates significant population shift requiring model review.",
    # 7
    "Stage 3 exposures are credit-impaired assets where one or more events "
    "with detrimental impact on estimated future cash flows have occurred, "
    "such as significant financial difficulty or default of the obligor.",
    # 8
    "The Gini coefficient and area under the ROC curve (AUC) quantify a "
    "rating model's discriminatory power; validation teams commonly expect "
    "Gini above 40 percent for retail scorecards.",
    # 9
    "Early warning indicators for credit deterioration include covenant "
    "breaches, auditor going-concern qualifications, delayed financial "
    "reporting, and rising utilisation of working capital lines.",
]

# (query, index of the correct passage)
GOLD = [
    ("What is the minimum capital adequacy ratio under Basel III?",            0),
    ("When does an asset move to Stage 2 under IFRS 9?",                       1),
    ("How often must rating models be validated per CBUAE MMG?",               2),
    ("What DSCR is required for corporate term loans?",                        3),
    ("What is the LTV cap for commercial real estate?",                        4),
    ("How is expected credit loss calculated?",                                5),
    ("What PSI value indicates population shift?",                             6),
    ("What defines a Stage 3 credit-impaired asset?",                          7),
    ("What Gini is expected for retail scorecards?",                           8),
    ("What are early warning signs of credit deterioration?",                  9),
    # paraphrases / keyword-light variants
    ("CET1 minimum including conservation buffer",                             0),
    ("significant increase in credit risk trigger",                            1),
    ("model discriminatory power annual review requirement",                   2),
    ("cash flow to debt obligations coverage threshold",                       3),
    ("PD LGD EAD formula",                                                     5),
    ("score distribution drift metric threshold",                              6),
]


def _rank_of(target: int, ranking: list) -> int:
    """1-based rank of target in ranking, or 0 if absent."""
    try:
        return ranking.index(target) + 1
    except ValueError:
        return 0


def _metrics(prefix: str, rankings: dict) -> dict:
    """Hit@1, Hit@3, MRR, nDCG@5 with a single relevant passage per query."""
    hit1 = hit3 = mrr = ndcg5 = 0.0
    n = len(GOLD)
    for (query, gold_idx) in GOLD:
        rank = _rank_of(gold_idx, rankings[query])
        if rank == 1:
            hit1 += 1
        if 1 <= rank <= 3:
            hit3 += 1
        if rank:
            mrr += 1.0 / rank
        if 1 <= rank <= 5:
            ndcg5 += 1.0 / math.log2(rank + 1)   # ideal DCG = 1 for one relevant doc
    out = {
        f"{prefix}_hit@1":  hit1 / n,
        f"{prefix}_hit@3":  hit3 / n,
        f"{prefix}_mrr":    mrr / n,
        f"{prefix}_ndcg@5": ndcg5 / n,
    }
    print(f"{prefix:<8} Hit@1={out[f'{prefix}_hit@1']:.2f}  Hit@3={out[f'{prefix}_hit@3']:.2f}  "
          f"MRR={out[f'{prefix}_mrr']:.3f}  nDCG@5={out[f'{prefix}_ndcg@5']:.3f}")
    return out


def run() -> int:
    # ── BM25 ───────────────────────────────────────────────────────────
    bm25 = BM25Okapi(CORPUS)
    bm25_rankings = {
        q: [i for i, _ in bm25.top_n(q, len(CORPUS))]
        for q, _ in GOLD
    }

    # ── Dense (optional) ───────────────────────────────────────────────
    dense_rankings = None
    try:
        from sentence_transformers import SentenceTransformer
        import numpy as np
        model = SentenceTransformer("all-MiniLM-L6-v2")
        doc_emb = model.encode(CORPUS, normalize_embeddings=True, convert_to_numpy=True)
        dense_rankings = {}
        for q, _ in GOLD:
            q_emb = model.encode([q], normalize_embeddings=True, convert_to_numpy=True)[0]
            sims  = doc_emb @ q_emb
            dense_rankings[q] = list(np.argsort(-sims))
    except ImportError:
        print("(sentence-transformers not installed — dense/hybrid skipped)\n")

    print("================ RETRIEVAL EVAL ================")
    print(f"{len(CORPUS)} passages, {len(GOLD)} labeled queries\n")

    metrics = _metrics("bm25", bm25_rankings)

    per_query_rows = [
        {"query": q, "gold": gi, "bm25_rank": _rank_of(gi, bm25_rankings[q])}
        for q, gi in GOLD
    ]

    if dense_rankings:
        metrics.update(_metrics("dense", dense_rankings))
        hybrid_rankings = {
            q: [i for i, _ in rrf_fuse([dense_rankings[q], bm25_rankings[q]])]
            for q, _ in GOLD
        }
        metrics.update(_metrics("hybrid", hybrid_rankings))
        for row in per_query_rows:
            row["dense_rank"]  = _rank_of(row["gold"], dense_rankings[row["query"]])
            row["hybrid_rank"] = _rank_of(row["gold"], hybrid_rankings[row["query"]])

    gates = build_gates("retrieval", metrics)
    print()
    print_gates(gates)
    warn_regressions("retrieval", metrics)
    write_report("retrieval", metrics, gates, rows=per_query_rows)

    code = exit_code(gates)
    print("\nRESULT:", "PASS" if code == 0 else "FAIL")
    return code


if __name__ == "__main__":
    sys.exit(run())
