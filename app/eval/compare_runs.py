"""
Diffs two eval run JSON files (see run_eval.py) and prints a metric-delta table.
This is what turns "I tuned the prompt" into a measured, defensible claim --
see BLUEPRINT.md §8 and the resume bullets in §11.

Usage:
    python -m app.eval.compare_runs eval_results/run_a.json eval_results/run_b.json
"""
import argparse
import json


def compare(path_a: str, path_b: str) -> None:
    with open(path_a) as f:
        a = json.load(f)
    with open(path_b) as f:
        b = json.load(f)

    metrics = ["avg_recall_at_5", "avg_mrr", "decline_accuracy", "avg_citation_correctness"]

    print(f"\nComparing:\n  A: {path_a} (commit {a['git_commit']}, {a['timestamp']})\n"
          f"  B: {path_b} (commit {b['git_commit']}, {b['timestamp']})\n")

    print(f"{'Metric':<28}{'A':>10}{'B':>10}{'Delta':>10}")
    print("-" * 58)
    for m in metrics:
        va, vb = a.get(m, 0.0), b.get(m, 0.0)
        delta = vb - va
        sign = "+" if delta >= 0 else ""
        print(f"{m:<28}{va:>10.3f}{vb:>10.3f}{sign}{delta:>9.3f}")

    # Per-question regressions worth flagging individually
    a_by_id = {q["id"]: q for q in a["per_question"]}
    b_by_id = {q["id"]: q for q in b["per_question"]}
    regressions = [
        qid for qid in a_by_id
        if qid in b_by_id and b_by_id[qid]["recall_at_5"] < a_by_id[qid]["recall_at_5"]
    ]
    if regressions:
        print(f"\nQuestions with Recall@5 regression: {regressions}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("run_a")
    parser.add_argument("run_b")
    args = parser.parse_args()
    compare(args.run_a, args.run_b)
