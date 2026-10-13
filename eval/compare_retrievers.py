"""Checks the rewritten retriever returns exactly the same results as the old one."""
import json
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from src.retriever import load_index, retrieve      # old version
from src.kb_retriever import KnowledgeBase          # rewritten version

old = load_index()
new = KnowledgeBase.build()

with open(ROOT / "eval" / "eval_set.json", encoding="utf-8") as f:
    queries = [item["query"] for item in json.load(f)]

def pairs(results):
    return [(r["id"], r["similarity"]) for r in results]

mismatches = 0
for q in queries:
    a, b = pairs(retrieve(q, old, top_k=3)), pairs(new.search(q, top_k=3))
    if a != b:
        mismatches += 1
        print("MISMATCH:", repr(q), "\n  old:", a, "\n  new:", b)
print(f"{len(queries)} queries compared, {mismatches} mismatches")

# save/load round trip
with tempfile.TemporaryDirectory() as d:
    path = Path(d) / "index.joblib"
    new.save(path)
    loaded = KnowledgeBase.load(path)
    same = all(pairs(new.search(q)) == pairs(loaded.search(q)) for q in queries)
    print("save/load round trip identical:", same)
    print("index stale right after build:", loaded.is_stale())