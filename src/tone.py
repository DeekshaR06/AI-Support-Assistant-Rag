"""Tone scoring with VADER. Describes the tone of a message only: it does not
establish intent, urgency, or the correctness of an answer."""
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

_analyzer = SentimentIntensityAnalyzer()
POSITIVE_AT = 0.05
NEGATIVE_AT = -0.05


def score_tone(text: str) -> dict:
    text = text if isinstance(text, str) else ""
    compound = _analyzer.polarity_scores(text)["compound"]
    if compound >= POSITIVE_AT:
        label = "positive"
    elif compound <= NEGATIVE_AT:
        label = "negative"
    else:
        label = "neutral"
    return {"label": label, "compound": round(compound, 3)}