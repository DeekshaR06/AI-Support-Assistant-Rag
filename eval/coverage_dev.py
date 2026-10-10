import json, sys
from pathlib import Path
ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))
from src.retriever import load_index, retrieve
from src.answerability import term_coverage

res = load_index()
items = json.load(open(ROOT / "eval" / "eval_set.json", encoding="utf-8"))
rows = []
for it in items:
    if it["split"] != "dev" or not it["query"].strip():
        continue
    r = retrieve(it["query"], res, top_k=3)
    rows.append((term_coverage(it["query"], r), it["answerable"], it["id"], it["query"]))
for cov, ans, i, q in sorted(rows, key=lambda x: (x[0] is None, x[0])):
    print(f"{cov if cov is None else round(cov, 2)!s:>5}  answerable={ans!s:<5}  {i}  {q}")