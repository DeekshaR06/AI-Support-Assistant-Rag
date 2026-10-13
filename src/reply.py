"""Reply composition, offline template mode. Builds the reply from retrieved
articles only. Whether to answer at all is decided earlier, by answerability.py."""

NO_ARTICLES_MESSAGE = (
    "I couldn't find anything specific about that in our knowledge base. "
    "Please contact our support team, who can help further."
)
TONE_PREFIX = {
    "negative": "I'm really sorry for the trouble this has caused. ",
    "positive": "Glad to hear it! ",
}
SECOND_ARTICLE_MIN_SIMILARITY = 0.5


def compose_reply(retrieved: list, sentiment_label: str = "neutral") -> dict:
    """Return {"answer": str, "backend": str, "sources": [article ids actually used]}."""
    if not retrieved:
        return {"answer": NO_ARTICLES_MESSAGE, "backend": "template", "sources": []}

    top = retrieved[0]
    parts = [TONE_PREFIX.get(sentiment_label, "") + top["answer"]]
    sources = [top["id"]]

    if len(retrieved) > 1 and retrieved[1]["similarity"] > SECOND_ARTICLE_MIN_SIMILARITY:
        parts.append("You may also find this useful: " + retrieved[1]["answer"])
        sources.append(retrieved[1]["id"])

    return {"answer": " ".join(parts), "backend": "template", "sources": sources}
