"""
One-time setup: builds the knowledge-base index and trains the intent classifier.

    python train.py
"""
import sys
from pathlib import Path

from src.intent_classifier import train_intent_classifier
from src.kb_retriever import DEFAULT_INDEX_PATH, DEFAULT_KB_PATH, KnowledgeBase

ROOT = Path(__file__).parent
TICKETS_CSV = ROOT / "data" / "tickets.csv"


def main():
    missing = [str(p) for p in (TICKETS_CSV, DEFAULT_KB_PATH) if not p.exists()]
    if missing:
        sys.exit(f"Missing data file(s): {missing}")

    print("--- Building knowledge-base index ---")
    kb = KnowledgeBase.build(DEFAULT_KB_PATH)
    kb.save(DEFAULT_INDEX_PATH)
    print(f"Indexed {len(kb.articles)} articles -> {DEFAULT_INDEX_PATH}")

    print("\n--- Training intent classifier ---")
    result = train_intent_classifier(str(TICKETS_CSV))
    print(result["report"])
    print("\nSetup complete. Run: streamlit run app.py")


if __name__ == "__main__":
    main()