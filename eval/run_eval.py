"""
Evaluation runner. Run from the project root:
    python eval\run_eval.py --split all --label baseline

Metrics (see README/report for definitions):
  Answerable questions : recall@1, recall@k, precision@k, MRR,
                         correct_top1_answer_rate, wrongly_abstained_rate
  Unanswerable         : correct_abstention_rate, unsafe_answer_rate
The pipeline reports abstention via result["abstained"]. The ORIGINAL pipeline
has no such field, so it is treated as "never abstains" (that is the baseline).
"""
import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from src.pipeline import SupportAssistant  # noqa: E402


def load_eval(split):
    with open(ROOT / "eval" / "eval_set.json", encoding="utf-8") as f:
        items = json.load(f)
    return items if split == "all" else [i for i in items if i["split"] == split]


def safe_mean(values):
    return round(sum(values) / len(values), 3) if values else None


def run(items, top_k):
    assistant = SupportAssistant()
    rows = []
    for item in items:
        start = time.perf_counter()
        result = assistant.handle(item["query"], top_k=top_k)
        latency_ms = (time.perf_counter() - start) * 1000

        retrieved_ids = [r["id"] for r in result["retrieved"]]
        relevant = set(item["relevant_ids"])
        rank = next((i for i, d in enumerate(retrieved_ids) if d in relevant), None)
        abstained = bool(result.get("abstained", False))

        rows.append({
            "id": item["id"], "split": item["split"], "category": item["category"],
            "query": item["query"], "answerable": item["answerable"],
            "relevant_ids": item["relevant_ids"], "retrieved_ids": retrieved_ids,
            "top_similarity": result["retrieved"][0]["similarity"] if result["retrieved"] else None,
            "rank_of_first_relevant": rank, "abstained": abstained,
            "answer": result["answer"], "latency_ms": round(latency_ms, 1),
        })
    return rows


def summarize(rows, top_k):
    ans = [r for r in rows if r["answerable"]]
    unans = [r for r in rows if not r["answerable"]]
    hits = lambda r: sum(1 for d in r["retrieved_ids"] if d in r["relevant_ids"])
    return {
        "n_total": len(rows), "n_answerable": len(ans), "n_unanswerable": len(unans),
        "answerable": {
            "recall_at_1": safe_mean([r["rank_of_first_relevant"] == 0 for r in ans]),
            f"recall_at_{top_k}": safe_mean([r["rank_of_first_relevant"] is not None for r in ans]),
            f"precision_at_{top_k}": safe_mean([hits(r) / len(r["retrieved_ids"]) if r["retrieved_ids"] else 0 for r in ans]),
            "mrr": safe_mean([1 / (r["rank_of_first_relevant"] + 1) if r["rank_of_first_relevant"] is not None else 0 for r in ans]),
            "correct_top1_answer_rate": safe_mean([(not r["abstained"]) and r["rank_of_first_relevant"] == 0 for r in ans]),
            "wrongly_abstained_rate": safe_mean([r["abstained"] for r in ans]),
        },
        "unanswerable": {
            "correct_abstention_rate": safe_mean([r["abstained"] for r in unans]),
            "unsafe_answer_rate": safe_mean([not r["abstained"] for r in unans]),
        },
        "mean_latency_ms": safe_mean([r["latency_ms"] for r in rows]),
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--split", choices=["dev", "heldout", "all"], default="all")
    p.add_argument("--label", default="baseline")
    p.add_argument("--top-k", type=int, default=3)
    args = p.parse_args()

    rows = run(load_eval(args.split), args.top_k)
    summary = summarize(rows, args.top_k)

    print(json.dumps(summary, indent=2))

    # Per-question failures are shown for the DEV split only, so the held-out
    # split stays a fair final test.
    print("\nDEV failures:")
    for r in rows:
        if r["split"] != "dev":
            continue
        bad_answerable = r["answerable"] and (r["abstained"] or r["rank_of_first_relevant"] != 0)
        bad_unanswerable = (not r["answerable"]) and (not r["abstained"])
        if bad_answerable or bad_unanswerable:
            print(f"  [{r['id']}] {r['category']}: {r['query']!r} -> {r['retrieved_ids']} "
                  f"(sim {r['top_similarity']}, expected {r['relevant_ids'] or 'ABSTAIN'})")

    out_dir = ROOT / "eval" / "results"
    out_dir.mkdir(parents=True, exist_ok=True)
    out = out_dir / f"{args.label}_{args.split}.json"
    with open(out, "w", encoding="utf-8") as f:
        json.dump({"label": args.label, "split": args.split, "top_k": args.top_k,
                   "summary": summary, "rows": rows}, f, indent=2)
    print(f"\nSaved: {out}")


if __name__ == "__main__":
    main()