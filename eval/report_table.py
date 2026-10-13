"""Prints before/after summary tables (aggregates only) from saved eval results."""
import json
from pathlib import Path

RESULTS = Path(__file__).parent / "results"
STAGES = [
    ("Baseline (original)", "baseline41_all.json"),
    ("Step 2: account-record + no-evidence rules", "step2_all.json"),
    ("Step 3: + term-coverage rule", "step3_all.json"),
]


def load(fname):
    with open(RESULTS / fname, encoding="utf-8") as f:
        return json.load(f)["rows"]


def pct(k, n):
    return f"{k}/{n} ({k / n:.0%})" if n else "n/a"


for split in ["all", "heldout", "dev"]:
    print(f"\n### Split: {split}\n")
    print("| Stage | Unanswerable correctly abstained | Answerable referred to support "
          "| recall@1 | recall@3 | MRR | Correct top-1 answers |")
    print("|---|---|---|---|---|---|---|")
    for name, fname in STAGES:
        rows = [r for r in load(fname) if split == "all" or r["split"] == split]
        ans = [r for r in rows if r["answerable"]]
        un = [r for r in rows if not r["answerable"]]
        ranks = [r["rank_of_first_relevant"] for r in ans]
        n = len(ans)
        mrr = sum(1 / (x + 1) for x in ranks if x is not None) / n
        correct = sum((not r["abstained"]) and r["rank_of_first_relevant"] == 0 for r in ans)
        print(f"| {name} | {pct(sum(r['abstained'] for r in un), len(un))} "
              f"| {pct(sum(r['abstained'] for r in ans), n)} "
              f"| {sum(x == 0 for x in ranks) / n:.2f} | {sum(x is not None for x in ranks) / n:.2f} "
              f"| {mrr:.2f} | {pct(correct, n)} |")