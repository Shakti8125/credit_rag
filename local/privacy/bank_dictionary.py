"""
GCC/UAE bank-name dictionary — the highest-confidence masking targets.

Standalone module (no heavy imports) so both the masker (which needs spaCy)
and the egress validator (which must stay lightweight) can share one list.
"""

BANK_DICTIONARY = [
    "Emirates NBD",
    "First Abu Dhabi Bank",
    "FAB",
    "Abu Dhabi Commercial Bank",
    "ADCB",
    "Dubai Islamic Bank",
    "DIB",
    "Mashreq Bank",
    "Mashreq",
    "RAKBANK",
    "Rak Bank",
    "Commercial Bank of Dubai",
    "CBD Bank",
    "National Bank of Abu Dhabi",
    "NBAD",
    "HSBC UAE",
    "Standard Chartered UAE",
    "Citibank UAE",
    "Qatar National Bank",
    "QNB",
    "Doha Bank",
    "Kuwait Finance House",
    "KFH",
    "National Bank of Kuwait",
    "NBK",
    "Al Rajhi Bank",
    "Saudi National Bank",
    "SNB",
    "Bank Muscat",
    "National Bank of Ras Al Khaimah",
]
