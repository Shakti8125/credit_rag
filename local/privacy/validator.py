import re
import logging
from typing import Dict, Any, List

# Configure module-level logger
logger = logging.getLogger(__name__)

class LeakageValidationError(Exception):
    """Exception raised when raw, unmasked sensitive data patterns are detected in the egress payload."""
    pass

class DocumentValidator:
    """
    Pre-flight validation firewall that inspects processed text for residual 
    unmasked PII or financial identifiers before cloud tier transmission.
    """
    
    def __init__(self) -> None:
        """
        Initializes the validator with robust, pre-compiled regex patterns 
        targeting common high-risk data leakage vulnerabilities.
        """
        self.leakage_patterns: Dict[str, re.Pattern] = {
            # Standard RFC 5322 compliant email pattern match
            "email": re.compile(
                r"[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}", 
                re.IGNORECASE
            ),
            # US Social Security Number (SSN) tracking: XXX-XX-XXXX
            "ssn": re.compile(r"\b\d{3}-\d{2}-\d{4}\b"),
            
            # US Employer Identification Number (EIN) / Corporate Tax ID: XX-XXXXXXX
            "tax_id_ein": re.compile(r"\b\d{2}-\d{7}\b"),
            
            # International/Standard E.164 phone formats and common localized variations
            "phone_number": re.compile(
                r"\b(?:\+?\d{1,3}[-.\s]?)?\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}\b"
            ),
            
            # Catches dangling unmasked templating expressions or structural leakage artifacts
            "malformed_placeholder": re.compile(r"\{\{.*?\}\}|\{\%.*?\%\}")
        }

    def validate(self, masked_text: str) -> Dict[str, Any]:
        """
        Scans a text string against all configured leakage risk signatures.

        Args:
            masked_text: The anonymized Markdown output string from the DocumentMasker.

        Returns:
            A validation summary dictionary if the payload passes all security checks.

        Raises:
            LeakageValidationError: If any unmasked PII pattern matches are found.
        """
        logger.debug("Initiating pre-flight privacy firewall scan on egress text payload...")
        
        if not masked_text.strip():
            logger.warning("Empty payload submitted for validation. Passing check by default.")
            return {"status": "passed", "findings": {}}

        leak_report: Dict[str, List[str]] = {}
        total_violations = 0

        # Scan the payload against each signature pattern
        for tracking_label, pattern in self.leakage_patterns.items():
            matches = pattern.findall(masked_text)
            if matches:
                # Deduplicate findings to keep logs concise
                unique_matches = list(set(matches))
                leak_report[tracking_label] = unique_matches
                total_violations += len(unique_matches)
                
                logger.error(
                    f"Privacy Violation: Detected {len(unique_matches)} unique unmasked "
                    f"'{tracking_label}' sequences attempting to leave local boundary."
                )

        # If any patterns matched, halt execution immediately to safeguard data
        if total_violations > 0:
            error_summary = ", ".join([f"{k} ({len(v)})" for k, v in leak_report.items()])
            critical_msg = (
                f"Egress Blocked! Data leakage patterns detected in masked text. "
                f"Violations found: [{error_summary}]. Transmission aborted."
            )
            logger.critical(critical_msg)
            raise LeakageValidationError(critical_msg)

        logger.info("Privacy validation successful. Zero high-risk data leakage patterns detected.")
        
        return {
            "status": "passed",
            "checks_evaluated": list(self.leakage_patterns.keys())
        }