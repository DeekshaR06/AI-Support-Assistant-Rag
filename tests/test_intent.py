import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import pandas as pd
from src.intent_classifier import (
    load_classifier, load_unique_tickets, normalise_text, predict_intent,
)


def _csv(tmp_path, rows):
    path = tmp_path / "t.csv"
    pd.DataFrame(rows, columns=["text", "intent"]).to_csv(path, index=False)
    return path


def test_numbers_do_not_make_tickets_unique(tmp_path):
    path = _csv(tmp_path, [
        ("Refund order #111 now", "cancellation_refund"),
        ("refund order #222 NOW", "cancellation_refund"),
        ("Where is my parcel", "shipping_delivery"),
    ])
    _, info = load_unique_tickets(path)
    assert info["n_raw"] == 3 and info["n_unique"] == 2


def test_label_conflicts_are_counted(tmp_path):
    path = _csv(tmp_path, [("Help me", "billing"), ("help me", "technical_support")])
    _, info = load_unique_tickets(path)
    assert info["label_conflicts"] == 1


def test_normalise_text():
    assert normalise_text("  Order  #81426 ") == "order ##"


def test_predict_intent_handles_non_string():
    result = predict_intent(load_classifier(), None)
    assert isinstance(result["intent"], str)


def test_scores_sum_to_about_one():
    result = predict_intent(load_classifier(), "I was charged twice")
    assert abs(sum(result["all_scores"].values()) - 1.0) < 0.01