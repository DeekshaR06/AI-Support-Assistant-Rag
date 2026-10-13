"""Checks the rewritten tone and reply modules match the old sentiment and generator."""
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))
os.environ.pop("OPENAI_API_KEY", None)  # force the old generator into template mode

from src.sentiment import analyze_sentiment
from src.generator import generate_answer
from src.tone import score_tone
from src.reply import compose_reply
from src.kb_retriever import KnowledgeBase

with open(ROOT / "eval" / "eval_set.json", encoding="utf-8") as f:
    queries = [i["query"] for i in json.load(f)]
queries += ["This is terrible, I'm furious!", "Thank you so much, this was great!",
            "What is your refund policy?"]

kb = KnowledgeBase.build()
tone_bad = reply_bad = 0
for q in queries:
    old_s, new_s = analyze_sentiment(q), score_tone(q)
    if (old_s["label"], old_s["compound"]) != (new_s["label"], new_s["compound"]):
        tone_bad += 1
        print("TONE MISMATCH:", repr(q), old_s, new_s)
    retrieved = kb.search(q, top_k=3)
    old_a = generate_answer(q, retrieved, old_s["label"])["answer"]
    new_a = compose_reply(retrieved, old_s["label"])["answer"]
    if old_a != new_a:
        reply_bad += 1
        print("REPLY MISMATCH:", repr(q), "\n  old:", old_a, "\n  new:", new_a)
print(f"{len(queries)} queries: {tone_bad} tone mismatches, {reply_bad} reply mismatches")