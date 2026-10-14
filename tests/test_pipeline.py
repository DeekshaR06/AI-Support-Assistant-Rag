import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import pytest
from src.intent_classifier import load_classifier, predict_intent
from src.pipeline import SupportAssistant


@pytest.fixture(scope="module")
def assistant():
    return SupportAssistant()


def test_intent_classifier_loads_and_predicts():
    result = predict_intent(load_classifier(), "My order arrived broken, I want a refund")
    assert result["intent"] in {
        "billing", "technical_support", "account_login",
        "cancellation_refund", "shipping_delivery", "product_info",
    }
    assert 0.0 <= result["confidence"] <= 1.0


def test_login_question_retrieves_login_faq(assistant):
    result = assistant.handle("I can't log in and the reset email never arrives")
    assert "faq_009" in [r["id"] for r in result["retrieved"]]


def test_answerable_question_returns_answer_with_sources(assistant):
    result = assistant.handle("How long does shipping take?")
    assert result["abstained"] is False
    assert "faq_015" in result["sources"]
    assert isinstance(result["answer"], str) and result["answer"]


def test_result_has_expected_fields(assistant):
    result = assistant.handle("How do I cancel my subscription?")
    for key in ("query", "intent", "sentiment", "retrieved", "answer",
                "generation_backend", "sources", "abstained", "abstain_reason"):
        assert key in result


def test_abstained_result_has_no_sources(assistant):
    result = assistant.handle("Where is my order #55123 right now?")
    assert result["abstained"] is True and result["sources"] == []


def test_non_string_query_does_not_crash(assistant):
    result = assistant.handle(None)
    assert result["abstained"] is True and result["abstain_reason"] == "empty_query"


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


def test_low_coverage_query_abstains(assistant):
    result = assistant.handle("Do you offer phone support on weekends?")
    assert result["abstained"] is True
    assert result["abstain_reason"] == "low_coverage"


def test_well_covered_query_still_answers(assistant):
    result = assistant.handle("Where can I find my invoices?")
    assert result["abstained"] is False