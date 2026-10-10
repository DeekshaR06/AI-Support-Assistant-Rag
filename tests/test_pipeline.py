"""
Sanity tests for the pipeline. Run with:  pytest tests/ -v
(or: python3 -m pytest tests/ -v   from the project root)

These aren't exhaustive unit tests of every function — they're the tests that
actually matter for a demo: does the trained model exist and load, does retrieval
return something for an obviously-matching query, does the sentiment direction make
sense, does the whole pipeline run without crashing on edge cases like empty input.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import pytest
from src.sentiment import analyze_sentiment
from src.retriever import load_index, retrieve
from src.intent_classifier import load_classifier, predict_intent
from src.pipeline import SupportAssistant


@pytest.fixture(scope="module")
def assistant():
    return SupportAssistant()


def test_sentiment_direction():
    assert analyze_sentiment("This is terrible, I'm furious!")["label"] == "negative"
    assert analyze_sentiment("Thank you so much, this was great!")["label"] == "positive"
    assert analyze_sentiment("What is your refund policy?")["label"] == "neutral"


def test_intent_classifier_loads_and_predicts():
    clf = load_classifier()
    result = predict_intent(clf, "My order arrived broken, I want a refund")
    assert result["intent"] in {
        "billing", "technical_support", "account_login",
        "cancellation_refund", "shipping_delivery", "product_info",
    }
    assert 0.0 <= result["confidence"] <= 1.0


def test_retriever_returns_relevant_top_match():
    resources = load_index()
    results = retrieve("I can't log in and the reset email never arrives", resources, top_k=3)
    assert len(results) > 0
    # top match should be the login/reset FAQ, not something unrelated like shipping
    assert "log" in results[0]["question"].lower() or "reset" in results[0]["question"].lower()


def test_retriever_similarity_is_ordered():
    resources = load_index()
    results = retrieve("refund for duplicate charge", resources, top_k=5)
    similarities = [r["similarity"] for r in results]
    assert similarities == sorted(similarities, reverse=True)


def test_full_pipeline_smoke(assistant):
    result = assistant.handle("The app keeps crashing on startup, please help")
    assert result["intent"]["intent"] is not None
    assert result["sentiment"]["label"] in {"positive", "neutral", "negative"}
    assert isinstance(result["answer"], str) and len(result["answer"]) > 0

def test_empty_query_abstains(assistant):
    result = assistant.handle("")
    assert result["abstained"] is True
    assert result["abstain_reason"] == "empty_query"


def test_no_evidence_query_abstains(assistant):
    result = assistant.handle("What is the weather like on Mars today?")
    assert result["abstained"] is True
    assert result["abstain_reason"] == "no_evidence"


def test_customer_specific_query_abstains(assistant):
    result = assistant.handle("Where is my order #55123 right now?")
    assert result["abstained"] is True
    assert result["abstain_reason"] == "customer_specific"


def test_answerable_query_does_not_abstain(assistant):
    result = assistant.handle("How do I get a refund for a duplicate charge?")
    assert result["abstained"] is False


def test_abstention_does_not_claim_escalation(assistant):
    result = assistant.handle("")
    assert "escalat" not in result["answer"].lower()
