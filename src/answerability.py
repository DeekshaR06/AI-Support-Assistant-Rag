"""
Answerability checks: decide whether the knowledge base can safely answer.
These are simple, readable rules, not machine learning. They are crude by design
and should be evaluated, not trusted. Retrieval similarity alone is NOT used as
proof of an answer.
"""
import re

MESSAGES = {
    "empty_query": "Please type a question so I can search the knowledge base.",
    "customer_specific": (
        "I can't see account, order or payment records, so I can't answer questions "
        "about a specific order, refund or charge. Please contact our support team."
    ),
    "no_evidence": (
        "I couldn't find anything about that in the knowledge base. "
        "Please contact our support team for help."
    ),
}

# Patterns suggesting the customer needs THEIR OWN record (status, tracking, etc.)
CUSTOMER_SPECIFIC_PATTERNS = [
    r"#\s*\d{3,}",                                             # order numbers like #55123
    r"\bmy specific\b",
    r"\b(status|tracking)\b.*\bmy\b|\bmy\b.*\b(status|tracking)\b",
    r"\bwhere\s+is\s+my\s+(order|package|parcel|refund|delivery|shipment)\b",
    r"\bwhen\b.*\bmy\b.*\b(arrive|arrives|reach|credited|processed|refunded)\b",
    r"\bwhy\b.*\bmy\b.*\b(declined|rejected|blocked|failed)\b",
]
_COMPILED = [re.compile(p, re.IGNORECASE) for p in CUSTOMER_SPECIFIC_PATTERNS]


def check_answerability(query: str, retrieved: list) -> dict:
    """Return {"abstain": bool, "reason": str|None, "message": str|None}."""
    if not query or not query.strip():
        reason = "empty_query"
    elif any(p.search(query) for p in _COMPILED):
        reason = "customer_specific"
    elif not retrieved or retrieved[0]["similarity"] <= 0.0:
        reason = "no_evidence"
    else:
        return {"abstain": False, "reason": None, "message": None}
    return {"abstain": True, "reason": reason, "message": MESSAGES[reason]}