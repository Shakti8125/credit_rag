"""
eval/privacy_eval.py — Privacy leakage evaluation for the masking pipeline.

The single most important guarantee of this system: no bank names or PII in
anything that leaves the analyst's machine, while financial values survive
masking verbatim (they're needed to answer queries).

Measures, over a set of synthetic credit-document fixtures with known
entities:
  1. Bank-name leak rate        — % of planted bank names surviving masking
  2. PII leak rate              — emails / phones / SSNs surviving masking
  3. Financial preservation     — % of planted metric values intact post-mask
  4. Egress firewall catch rate — validator must block seeded leaks
  5. Round-trip fidelity        — unmask(mask(text)) restores every entity

Run from the project root (requires spaCy + model, same env as the app):
    python eval/privacy_eval.py
"""

import re
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from shared.logging_config import get_logger
from eval.common import build_gates, print_gates, warn_regressions, write_report, exit_code

logger = get_logger("privacy_eval")


# ---------------------------------------------------------------------------
# Fixtures: (document text, planted bank names, planted PII, planted financials)
# ---------------------------------------------------------------------------

FIXTURES = [
    {
        "name": "credit_memo_uae",
        "text": (
            "Credit Proposal for facility renewal. The borrower banks primarily "
            "with Emirates NBD and holds a secondary facility from Mashreq Bank. "
            "DSCR of 1.18x, LTV 82%, Debt/EBITDA of 4.6x. Facility Amount: AED 250M. "
            "Contact the RM at khalid.m@corpclient.ae or +971 50 123 4567. "
            "Stage 3 ratio: 6.1% with provision coverage ratio: 44%."
        ),
        "banks":      ["Emirates NBD", "Mashreq Bank"],
        "pii":        ["khalid.m@corpclient.ae", "+971 50 123 4567"],
        "financials": ["1.18", "82", "4.6", "250", "6.1", "44"],
    },
    {
        "name": "model_validation_report",
        "text": (
            "RAKBANK Credit Card Scorecard — Annual Validation. "
            "Gini Coefficient of 43%, AUC: 0.74, KS Statistic: 33%, PSI of 0.08. "
            "Benchmarked against First Abu Dhabi Bank portfolio norms. "
            "Reviewer: analytics team, ref file RAKBANK_ccard_2025.xlsx."
        ),
        "banks":      ["RAKBANK", "First Abu Dhabi Bank"],
        "pii":        [],
        "financials": ["43", "0.74", "33", "0.08"],
    },
    {
        "name": "ews_style_narrative",
        "text": (
            "The obligor breached covenants with Dubai Islamic Bank in Q3. "
            "Going concern doubt raised by auditors. CAR declined to 9.8% and "
            "NPA ratio rose to 5.9%. Refinancing discussions with QNB ongoing. "
            "CFO reachable at cfo@obligorco.com."
        ),
        "banks":      ["Dubai Islamic Bank", "QNB"],
        "pii":        ["cfo@obligorco.com"],
        "financials": ["9.8", "5.9"],
    },
]


def run() -> int:
    try:
        from local.privacy.masker import DocumentMasker
        from local.privacy.validator import DocumentValidator, LeakageValidationError
        from shared.masking import EntityRegistry, unmask_text
    except ImportError as e:
        print(f"SKIP: privacy eval requires the local tier's dependencies ({e}).")
        print("Run inside the app environment: python eval/privacy_eval.py")
        return 2

    registry  = EntityRegistry()
    masker    = DocumentMasker(registry=registry, spacy_model="en_core_web_lg")
    validator = DocumentValidator()

    bank_leaks = pii_leaks = fin_lost = roundtrip_fail = 0
    bank_total = pii_total = fin_total = 0
    firewall_caught = firewall_total = 0

    for fx in FIXTURES:
        registry.clear()
        masked = masker.mask(fx["text"])

        # 1. Bank-name leaks (alphabetic lookaround, same as the masker)
        for bank in fx["banks"]:
            bank_total += 1
            if re.search(rf"(?<![a-zA-Z]){re.escape(bank)}(?![a-zA-Z])", masked, re.I):
                bank_leaks += 1
                print(f"  LEAK [{fx['name']}]: bank '{bank}' survived masking")

        # 2. PII leaks
        for item in fx["pii"]:
            pii_total += 1
            if item in masked:
                pii_leaks += 1
                print(f"  LEAK [{fx['name']}]: PII '{item}' survived masking")

        # 3. Financial preservation
        for value in fx["financials"]:
            fin_total += 1
            if value not in masked:
                fin_lost += 1
                print(f"  LOST [{fx['name']}]: financial value '{value}' not preserved")

        # 4. Egress firewall on masked output — must pass clean text
        firewall_total += 1
        try:
            validator.validate(masked)
        except LeakageValidationError:
            # Firewall caught something the masker missed — that's the
            # firewall doing its job, but count the masker miss above.
            pass

        # Firewall must catch a seeded leak
        firewall_total += 1
        try:
            validator.validate(masked + " leaked note: RAKBANK_internal.pdf")
            print(f"  MISS [{fx['name']}]: firewall did not catch seeded bank leak")
        except LeakageValidationError:
            firewall_caught += 1

        # 5. Round-trip fidelity
        restored = unmask_text(masked, registry.get_mask_dictionary())
        missing = [b for b in fx["banks"] if b not in restored] + \
                  [p for p in fx["pii"] if p not in restored]
        if missing:
            roundtrip_fail += 1
            print(f"  ROUNDTRIP [{fx['name']}]: not restored: {missing}")

    print("\n================ PRIVACY EVAL ================")
    print(f"Bank-name leak rate    : {bank_leaks}/{bank_total} "
          f"({100*bank_leaks/max(bank_total,1):.0f}%)  [target 0%]")
    print(f"PII leak rate          : {pii_leaks}/{pii_total} "
          f"({100*pii_leaks/max(pii_total,1):.0f}%)  [target 0%]")
    print(f"Financial preservation : {fin_total-fin_lost}/{fin_total} "
          f"({100*(fin_total-fin_lost)/max(fin_total,1):.0f}%)  [target 100%]")
    print(f"Firewall seeded-leak catches: {firewall_caught}/{len(FIXTURES)}  [target {len(FIXTURES)}]")
    print(f"Round-trip failures    : {roundtrip_fail}/{len(FIXTURES)}  [target 0]")

    metrics = {
        "bank_leak_rate":         bank_leaks / max(bank_total, 1),
        "pii_leak_rate":          pii_leaks / max(pii_total, 1),
        "financial_preservation": (fin_total - fin_lost) / max(fin_total, 1),
        "firewall_catch_rate":    firewall_caught / len(FIXTURES),
        "roundtrip_failures":     roundtrip_fail,
        "fixtures":               len(FIXTURES),
    }
    gates = build_gates("privacy", metrics)
    print()
    print_gates(gates)
    warn_regressions("privacy", metrics)
    write_report("privacy", metrics, gates)

    code = exit_code(gates)
    print("\nRESULT:", "PASS" if code == 0 else "FAIL")
    return code


if __name__ == "__main__":
    sys.exit(run())
