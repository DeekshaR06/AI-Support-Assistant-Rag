"""
Answerability checks: decide whether the knowledge base can safely answer.
Simple, readable rules, not machine learning. They are crude by design and
must be evaluated, not trusted. Retrieval similarity alone is NOT proof of an answer.
"""
import re
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS

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
    "low_coverage": (
        "The knowledge base doesn't seem to cover the details of your question. "
        "Please contact our support team for help."
    ),
}

CUSTOMER_SPECIFIC_PATTERNS = [
    r"#\s*\d{3,}",
    r"\bmy specific\b",
    r"\b(status|tracking)\b.*\bmy\b|\bmy\b.*\b(status|tracking)\b",
    r"\bwhere\s+is\s+my\s+(order|package|parcel|refund|delivery|shipment)\b",
    r"\bwhen\b.*\bmy\b.*\b(arrive|arrives|reach|credited|processed|refunded)\b",
    r"\bwhy\b.*\bmy\b.*\b(declined|rejected|blocked|failed)\b",
]
_COMPILED = [re.compile(p, re.IGNORECASE) for p in CUSTOMER_SPECIFIC_PATTERNS]

# Term-coverage rule. None = rule switched off. Set after tuning on DEV questions only.
COVERAGE_THRESHOLD = 0.3

_EXTRA_STOP = {"want", "need", "know", "help", "please", "like", "get", "got",
               "just", "able", "possible", "tell", "thing", "things", "use"}


def _stem(word: str) -> str:
    """Very crude stemmer. Applied to both query and article, so it only needs to be consistent."""
    for suffix in ("ing", "ed", "es", "s"):
        if word.endswith(suffix) and not word.endswith("ss") and len(word) - len(suffix) >= 3:
            word = word[: -len(suffix)]
            break
    if word.endswith("e") and len(word) > 3:
        word = word[:-1]
    if len(word) > 3 and word[-1] == word[-2]:
        word = word[:-1]
    return word


def content_terms(text: str) -> set:
    words = re.findall(r"[a-z]+", text.lower())
    return {_stem(w) for w in words
            if len(w) >= 3 and w not in ENGLISH_STOP_WORDS and w not in _EXTRA_STOP}


def term_coverage(query: str, retrieved: list):
    """Fraction of the query's meaningful words found in the retrieved articles (None if undefined)."""
    q_terms = content_terms(query)
    if not q_terms or not retrieved:
        return None
    doc_terms = set()
    for r in retrieved:
        doc_terms |= content_terms(r["question"] + " " + r["answer"])
    return len(q_terms & doc_terms) / len(q_terms)


def check_answerability(query: str, retrieved: list) -> dict:
    """Return {"abstain": bool, "reason": str|None, "message": str|None}."""
    if not query or not query.strip():
        reason = "empty_query"
    elif any(p.search(query) for p in _COMPILED):
        reason = "customer_specific"
    elif not retrieved or retrieved[0]["similarity"] <= 0.0:
        reason = "no_evidence"
    elif COVERAGE_THRESHOLD is not None:
        cov = term_coverage(query, retrieved)
        if cov is not None and cov < COVERAGE_THRESHOLD:
            reason = "low_coverage"
        else:
            return {"abstain": False, "reason": None, "message": None}
    else:
        return {"abstain": False, "reason": None, "message": None}
    return {"abstain": True, "reason": reason, "message": MESSAGES[reason]}