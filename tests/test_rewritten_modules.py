import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import pytest
from src.kb_retriever import KnowledgeBase, load_articles
from src.reply import compose_reply
from src.tone import score_tone

ARTICLES = [
    {"id": "a", "question": "How do I reset my password?", "answer": "Use the reset link on the login page."},
    {"id": "b", "question": "Where are invoices?", "answer": "Invoices are under billing settings."},
    {"id": "c", "question": "Do you ship abroad?", "answer": "Yes, we ship to many countries."},
]


def _write(tmp_path, data):
    p = tmp_path / "kb.json"
    p.write_text(json.dumps(data), encoding="utf-8")
    return p


def test_rejects_duplicate_ids(tmp_path):
    p = _write(tmp_path, [ARTICLES[0], {**ARTICLES[1], "id": "a"}])
    with pytest.raises(ValueError, match="Duplicate"):
        load_articles(p)


def test_rejects_missing_fields(tmp_path):
    p = _write(tmp_path, [{"id": "a", "question": "Q only"}])
    with pytest.raises(ValueError, match="missing"):
        load_articles(p)


def test_search_is_ordered_and_ranked():
    results = KnowledgeBase.build().search("refund for duplicate charge", top_k=5)
    sims = [r["similarity"] for r in results]
    assert len(results) == 5 and sims == sorted(sims, reverse=True)
    assert [r["rank"] for r in results] == [1, 2, 3, 4, 5]


def test_search_non_string_query_does_not_crash():
    assert isinstance(KnowledgeBase.build().search(None), list)


def test_stale_detection(tmp_path):
    p = _write(tmp_path, ARTICLES)
    kb = KnowledgeBase.build(p)
    assert kb.is_stale(p) is False
    p.write_text(json.dumps(ARTICLES + [{"id": "d", "question": "New?", "answer": "New answer."}]),
                 encoding="utf-8")
    assert kb.is_stale(p) is True


def test_reply_with_no_articles_refers_to_support():
    r = compose_reply([])
    assert r["sources"] == [] and "contact our support team" in r["answer"]
    assert "escalat" not in r["answer"].lower()


def test_reply_tone_prefix_and_second_article():
    top = {"id": "x", "answer": "Top answer.", "similarity": 0.9}
    close = {"id": "y", "answer": "Second answer.", "similarity": 0.6}
    far = {"id": "z", "answer": "Third answer.", "similarity": 0.4}
    r = compose_reply([top, close], "negative")
    assert r["answer"].startswith("I'm really sorry") and r["sources"] == ["x", "y"]
    assert compose_reply([top, far])["sources"] == ["x"]


def test_tone_labels():
    assert score_tone("This is terrible, I'm furious!")["label"] == "negative"
    assert score_tone("Thank you so much, this was great!")["label"] == "positive"
    assert score_tone("What is your refund policy?")["label"] == "neutral"