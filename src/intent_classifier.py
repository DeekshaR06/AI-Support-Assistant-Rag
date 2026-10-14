"""
Intent classification: TF-IDF + Logistic Regression.

Evaluation notes:
 - Duplicate tickets are removed first. Texts that differ only in digits (such as
   order numbers) count as duplicates, because they come from the same template.
 - Accuracy comes from stratified cross-validation on the unique tickets, not from
   one small random split.
 - Average confidence is compared with accuracy to check whether the score behaves
   like a probability.
Intent is shown in the UI but does NOT decide the answer.
"""
import re
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.pipeline import Pipeline

MODELS_DIR = Path(__file__).parent.parent / "models"
MODEL_PATH = MODELS_DIR / "intent_classifier.joblib"


def normalise_text(text) -> str:
    """Lower-case, replace every run of digits with '#', collapse spaces."""
    text = re.sub(r"\d+", "#", str(text).lower())
    return re.sub(r"\s+", " ", text).strip()


def load_unique_tickets(csv_path):
    """Return (unique tickets, info dict with raw/unique counts and label conflicts)."""
    df = pd.read_csv(csv_path)[["text", "intent"]].dropna()
    df["key"] = df["text"].map(normalise_text)
    conflicts = int((df.groupby("key")["intent"].nunique() > 1).sum())
    unique = df.drop_duplicates(subset="key", keep="first").reset_index(drop=True)
    info = {"n_raw": len(df), "n_unique": len(unique), "label_conflicts": conflicts}
    return unique[["text", "intent"]], info


def build_model() -> Pipeline:
    return Pipeline([
        ("tfidf", TfidfVectorizer(ngram_range=(1, 2), stop_words="english")),
        ("clf", LogisticRegression(max_iter=1000, class_weight="balanced")),
    ])


def train_intent_classifier(csv_path, save: bool = True) -> dict:
    data, info = load_unique_tickets(csv_path)
    X, y = data["text"], data["intent"].to_numpy()
    classes, counts = np.unique(y, return_counts=True)

    lines = [
        f"Tickets in file: {info['n_raw']}; unique after removing duplicates "
        f"(digits ignored): {info['n_unique']}",
        f"Texts with conflicting labels: {info['label_conflicts']}",
        "Unique tickets per intent: " + ", ".join(f"{c}={n}" for c, n in zip(classes, counts)),
    ]

    accuracy = None
    folds = int(min(5, counts.min()))
    if folds >= 2:
        cv = StratifiedKFold(n_splits=folds, shuffle=True, random_state=42)
        proba = cross_val_predict(build_model(), X, y, cv=cv, method="predict_proba")
        pred = classes[proba.argmax(axis=1)]
        accuracy = float((pred == y).mean())
        mean_conf = float(proba.max(axis=1).mean())
        lines += [
            f"\n{folds}-fold cross-validated accuracy on unique tickets: {accuracy:.3f}",
            f"Average confidence score: {mean_conf:.3f}  "
            "(if this is far from accuracy, the score is not calibrated)",
            "Caution: tickets are template-generated, so near-duplicates may remain and "
            "this figure is probably still optimistic.\n",
            classification_report(y, pred, zero_division=0),
        ]
    else:
        lines.append("Too few tickets in the smallest class for cross-validation.")

    final = build_model().fit(X, y)
    if save:
        MODELS_DIR.mkdir(parents=True, exist_ok=True)
        joblib.dump(final, MODEL_PATH)
    return {"pipeline": final, "accuracy": accuracy, "report": "\n".join(lines)}


def load_classifier(path=None):
    return joblib.load(path or MODEL_PATH)


def predict_intent(model, text) -> dict:
    text = text if isinstance(text, str) else ""
    proba = model.predict_proba([text])[0]
    classes = list(model.classes_)
    best = int(proba.argmax())
    return {
        "intent": classes[best],
        "confidence": round(float(proba[best]), 3),
        "all_scores": {c: round(float(p), 3) for c, p in zip(classes, proba)},
    }