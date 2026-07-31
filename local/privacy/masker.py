import re
import logging
from typing import List, Tuple, Set
import spacy
from spacy.tokens import Doc

from shared.masking import EntityRegistry
from local.privacy.bank_dictionary import BANK_DICTIONARY


logger = logging.getLogger(__name__)


class DocumentMasker:
    """
    Privacy masking layer for financial documents.

    Masks:
        - Named banks (dictionary-based)
        - Emails, phones, SSNs, EINs (regex)
        - Real person names, genuine org names, GPE/LOC entities (spaCy NER)

    Preserves:
        - Financial numbers, ratios, percentages, years
        - Credit risk domain terms (PD, LGD, DSCR, NPA, GINI, WoE, etc.)
        - Column names, abbreviations, model variable names
        - Currency codes (AED, USD, INR…)
        - Short tokens (≤2 chars) — too ambiguous to mask safely
        - Legal entity suffixes used as common nouns ("Public Joint Stock Company")

    Designed for Credit-RAG pipeline.
    """

    def __init__(
        self,
        registry: EntityRegistry,
        spacy_model: str = "en_core_web_trf"
    ) -> None:

        self.registry = registry

        logger.info("Loading spaCy model: %s", spacy_model)

        try:
            self.nlp = spacy.load(spacy_model)
        except OSError:
            logger.warning("Transformer unavailable, falling back to en_core_web_sm")
            self.nlp = spacy.load("en_core_web_sm")

        # ======================================================
        # Financial expressions — NEVER MASK
        # ======================================================

        self.financial_patterns = [

            # Currency values
            re.compile(
                r"""
                \b
                (?:AED|USD|EUR|GBP|INR|SAR|QAR|KWD|BHD|OMR)?
                \s?
                [$€£₹]?
                \s?
                \d{1,3}
                (?:,\d{3})*
                (?:\.\d+)?
                \s?
                (?:k|m|b|million|billion|cr|lakh)?
                \b
                """,
                re.I | re.X
            ),

            # Credit ratios with value
            re.compile(
                r"""
                \b
                (?:DSCR|LTV|ROE|ROA|
                TOL/ATNW|
                Debt/EBITDA|
                Leverage|ICR|CAR|CRAR|
                NPA|GNPA|NNPA|
                CET1|Tier\s*1)
                \s*
                (?:of|is|:|=)?
                \s*
                \d+(?:\.\d+)?
                \s*
                (?:%|x)?
                \b
                """,
                re.I | re.X
            ),

            # Bare percentage
            re.compile(r"\b\d+(?:\.\d+)?%\b"),

            # Multiples
            re.compile(r"\b\d+(?:\.\d+)?x\b", re.I),

            # Financial years
            re.compile(r"\b(?:19|20)\d{2}\b"),

            # FY notation
            re.compile(r"\bFY\d{2,4}\b", re.I),
        ]

        # ======================================================
        # NER labels to consider for masking
        # ======================================================

        self.target_ner_labels: Set[str] = {
            "PERSON",
            "ORG",
            "GPE",
            "LOC",
            "FAC",
        }

        # ======================================================
        # ORG confidence filter — only mask orgs that contain
        # at least one of these keywords (signals a real entity)
        # ======================================================

        self.high_confidence_org_keywords = {
            "bank",
            "limited",
            "ltd",
            "llc",
            "plc",
            "corp",
            "corporation",
            "inc",
            "company",
            "consulting",
            "solutions",
            "private limited",
            "group",
            "holding",
            "holdings",
            "finance",
            "capital",
            "investment",
            "securities",
            "insurance",
        }

        # ======================================================
        # DOMAIN BLOCKLIST — tokens spaCy frequently
        # misclassifies as PERSON / GPE / LOC / ORG.
        #
        # Covers:
        #   - Credit risk metrics & model terms
        #   - Statistical / ML variable names
        #   - Column headers from banking data files
        #   - Currency / country codes
        #   - Common legal suffixes used as nouns
        # ======================================================

        self.domain_blocklist: Set[str] = {

            # ── Credit risk metrics ───────────────────────────────
            "pd", "lgd", "ead", "el", "ul", "rwa",
            "dscr", "icr", "ltv", "ltc", "npa", "gnpa", "nnpa",
            "car", "crar", "cet1", "tier1",
            "roe", "roa", "roce", "ebitda", "ebit", "pat", "pbt",
            "tol", "atnw", "tnw", "opex", "capex",

            # ── Statistical / ML / scorecard terms ───────────────
            "gini", "ks", "auc", "roc", "iv", "woe",
            "chi", "chi-square", "wald", "wald chi-square",
            "p-value", "pvalue", "r-squared", "rsquared",
            "logit", "probit", "regression", "coefficient",
            "odds", "log-odds", "beta", "alpha",
            "precision", "recall", "f1", "accuracy",
            "train", "test", "validation",
            "oversampling", "undersampling", "smote",
            "bootstrap", "bagging", "binning",
            "fine binning", "coarse binning",
            "information value", "weight of evidence",
            "scorecard", "score", "cutoff",
            "decile", "quartile", "percentile", "quintile",
            "snapshot", "vintage",

            # ── Column / variable names common in banking data ────
            "cif_no", "crnno", "finware_acno",
            "maritalstatus", "allocatedamt",
            "mismonth", "mis_month",
            "gt85p_le100p", "gt100p",
            "fourth snapshot", "first snapshot",
            "max", "min", "mean", "median", "mode", "std",
            "null", "nan", "n/a", "na",
            "stat", "status",

            # ── Currency & country codes ──────────────────────────
            "aed", "usd", "eur", "gbp", "inr", "sar",
            "qar", "kwd", "bhd", "omr",
            "uae", "gcc", "mena",

            # ── Number magnitude abbreviations ────────────────────
            "mn", "bn", "cr", "lakh", "lac",

            # ── Legal entity suffixes used as common nouns ────────
            "public joint stock company",
            "private joint stock company",
            "joint stock company",
            "limited liability company",
            "sole proprietorship",
            "partnership",

            # ── Generic credit / banking phrases ─────────────────
            "fir the bank",          # hallucination
            "the bank",
            "bank",
            "borrower",
            "guarantor",
            "obligor",
            "counterparty",
            "lender",
        }

        # ======================================================
        # Regex PII patterns
        # ======================================================

        self.pii_patterns: List[Tuple[str, re.Pattern]] = [

            (
                "EMAIL",
                re.compile(
                    r"[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}",
                    re.X
                )
            ),

            (
                "PHONE",
                re.compile(
                    r"""
                    (?:
                        \+\d{1,3}[\s-]?
                    )?
                    (?:
                        \(?\d{2,4}\)?[\s-]?
                    )
                    \d{3,4}[\s-]?\d{3,4}
                    """,
                    re.X
                )
            ),

            (
                "SSN",
                re.compile(r"\b\d{3}-\d{2}-\d{4}\b")
            ),

            (
                "EIN",
                re.compile(r"\b\d{2}-\d{7}\b")
            ),
        ]

        # ======================================================
        # Bank dictionary (dictionary-based, highest confidence)
        # ======================================================

        self.bank_patterns = BANK_DICTIONARY

    # ==========================================================
    # Guard helpers
    # ==========================================================

    def _is_in_domain_blocklist(self, text: str) -> bool:
        """
        Returns True if the entity text (case-insensitive, stripped)
        matches any entry in the domain blocklist.
        Checks both exact match and prefix match for compound terms.
        """
        normalised = text.strip().lower()

        # Exact match
        if normalised in self.domain_blocklist:
            return True

        # Any blocklist phrase contained within the entity text
        for term in self.domain_blocklist:
            if term in normalised:
                return True

        return False

    def _is_too_short(self, text: str) -> bool:
        """
        Tokens with 2 or fewer non-whitespace characters are too
        ambiguous to mask safely (e.g. 'PD', 'KS', 'IV', 'MN').
        """
        return len(text.strip()) <= 2

    def _is_all_caps_abbreviation(self, text: str) -> bool:
        """
        ALL-CAPS tokens of 3–6 characters are almost always domain
        acronyms (NPA, GINI, WoE, DSCR) — never real person names.
        spaCy frequently misclassifies these as PERSON or GPE.
        """
        stripped = text.strip()
        return (
            3 <= len(stripped) <= 6
            and stripped.isupper()
            and stripped.isalpha()
        )

    def _is_real_org(self, text: str) -> bool:
        """
        Only mask ORG entities that contain a known org-type keyword.
        Prevents masking generic phrases like 'fir the bank'.
        """
        value = text.lower()
        return any(kw in value for kw in self.high_confidence_org_keywords)

    def _is_financial_number(self, text: str) -> bool:
        return any(
            p.fullmatch(text.strip())
            for p in self.financial_patterns
        )

    def _get_protected_spans(self, text: str) -> List[Tuple[int, int]]:
        spans = []
        for pattern in self.financial_patterns:
            for match in pattern.finditer(text):
                spans.append(match.span())
        return spans

    def _is_overlapping(self, entity_span, protected_spans) -> bool:
        start, end = entity_span
        for p_start, p_end in protected_spans:
            if not (end <= p_start or start >= p_end):
                return True
        return False

    def _should_skip_entity(self, ent_text: str, ent_label: str) -> bool:
        """
        Single consolidated gate: returns True if the entity should
        NOT be masked. Called once per spaCy entity.
        """
        # 1. Domain blocklist (catches column names, acronyms, etc.)
        if self._is_in_domain_blocklist(ent_text):
            logger.debug("Blocklist skip: '%s' (%s)", ent_text, ent_label)
            return True

        # 2. Pure financial number
        if self._is_financial_number(ent_text):
            return True

        # 3. Too short to be a meaningful PII entity
        if self._is_too_short(ent_text):
            return True

        # 4. ALL-CAPS abbreviation (≤6 chars) — almost always a domain acronym
        if self._is_all_caps_abbreviation(ent_text):
            logger.debug("ALL-CAPS skip: '%s' (%s)", ent_text, ent_label)
            return True

        # 5. ORG requires additional keyword confirmation
        if ent_label == "ORG" and not self._is_real_org(ent_text):
            return True

        # 6. PERSON / GPE / LOC — skip if the text looks like a variable
        #    name (contains underscores, digits mixed with letters, or is
        #    snake_case — typical of column headers in banking datasets)
        if ent_label in ("PERSON", "GPE", "LOC"):
            stripped = ent_text.strip()
            if "_" in stripped:          # snake_case column name
                return True
            if re.search(r"\d", stripped) and re.search(r"[a-zA-Z]", stripped):
                # Alphanumeric token — likely a code, not a person/place
                return True

        return False

    # ==========================================================
    # Regex masking
    # ==========================================================

    def _apply_regex_masks(self, text: str) -> str:
        replacements = []
        for label, pattern in self.pii_patterns:
            for match in pattern.finditer(text):
                value = match.group().strip()
                if not value:
                    continue
                token = self.registry.register_entity(value, label)
                replacements.append((match.start(), match.end(), token))

        replacements.sort(key=lambda x: x[0], reverse=True)
        buffer = list(text)
        for start, end, replacement in replacements:
            buffer[start:end] = list(replacement)
        return "".join(buffer)

    def _mask_banks(self, text: str) -> str:
        """
        Dictionary-based bank masking.

        Uses lookahead/lookbehind instead of \\b word boundaries so bank names
        are caught in all surface forms:
          - Normal prose      : "RAKBANK issued a facility"
          - Filename embedded : "RAK Bank Credit Card Model Development.docx"
          - Underscore joined : "RAKBANK_report.pdf"
          - Slash separated   : "data/RAKBANK/2024"

        \\b fails in these cases because \\b treats underscore, dot, and slash
        as non-word characters, so the boundary is absent between the bank
        name and those characters.

        The lookahead/lookbehind `(?<![a-zA-Z])` / `(?![a-zA-Z])` anchors on
        alphabetic characters only, matching across punctuation and filename
        delimiters while still preventing substring matches inside longer words
        (e.g. "FAB" must not match inside "FABRIC").
        """
        for bank in sorted(self.bank_patterns, key=len, reverse=True):
            pattern = re.compile(
                rf"(?<![a-zA-Z]){re.escape(bank)}(?![a-zA-Z])",
                re.I,
            )
            for match in pattern.finditer(text):
                token = self.registry.register_entity(match.group(), "BANK")
                text  = text.replace(match.group(), token)
        return text

    def mask_filename(self, filename: str) -> str:
        """
        Masks bank names embedded in a filename string.

        Filenames are not prose — they have no sentence structure — so we
        apply only the bank dictionary pass.  Regex PII and NER are not
        needed (filenames don't contain emails, phones, or complex entities).

        Returns the masked filename.  If no bank name is found, the original
        filename is returned unchanged.

        Example:
            "RAK Bank Credit Card Model Development.docx"
            → "[BANK_1] Credit Card Model Development.docx"
        """
        return self._mask_banks(filename)

    # ==========================================================
    # Main entry point
    # ==========================================================

    def mask(self, text: str) -> str:

        if not text.strip():
            return text

        # Step 1: Dictionary bank masking
        text = self._mask_banks(text)

        # Step 2: Regex PII
        text = self._apply_regex_masks(text)

        # Step 3: Compute financial protection spans
        # (must run AFTER bank/regex replacements — offsets are now stable)
        protected_spans = self._get_protected_spans(text)

        # Step 4: NER-based masking
        doc: Doc = self.nlp(text)
        replacements = []

        for ent in doc.ents:

            if ent.label_ not in self.target_ner_labels:
                continue

            if self._is_overlapping(
                (ent.start_char, ent.end_char),
                protected_spans,
            ):
                continue

            if self._should_skip_entity(ent.text, ent.label_):
                continue

            token = self.registry.register_entity(ent.text, ent.label_)
            replacements.append((ent.start_char, ent.end_char, token))

        replacements.sort(key=lambda x: x[0], reverse=True)
        buffer = list(text)
        for start, end, replacement in replacements:
            buffer[start:end] = list(replacement)

        return "".join(buffer)