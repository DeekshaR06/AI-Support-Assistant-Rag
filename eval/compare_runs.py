"""Usage: python eval\compare_runs.py step3 v2
Checks two saved runs behave identically. Differences are listed by id for dev
questions only; held-out differences are reported as a count."""
import json
import sys
from pathlib import Path

RES = Path(__file__).parent / "results"


def load(name):
    with open(RES / f"{name}_all.json", encoding="utf-8") as f:
        return {r["id"]: r for r in json.load(f)["rows"]}


a, b = load(sys.argv[1]), load(sys.argv[2])
fields = ["retrieved_ids", "abstained", "answer", "top_similarity"]
diff = {"dev": [], "heldout": []}
for qid, ra in a.items():
    rb = b.get(qid)
    if rb is None or any(ra[f] != rb[f] for f in fields):
        diff[ra["split"]].append(qid)
print(f"compared {len(a)} questions")
print("dev differences:", diff["dev"] or "none")
print("held-out differences (count only):", len(diff["heldout"]))