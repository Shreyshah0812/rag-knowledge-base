"""
Evaluation loop per BLUEPRINT.md §8. Sends every question through the REAL
/query endpoint (not a reimplementation) so eval results reflect the exact code
path production traffic uses. Writes a timestamped, git-commit-tagged report so
runs are comparable across changes (see compare_runs.py).

Usage:
    python -m app.eval.run_eval [--api-url http://localhost:8000]
"""
import argparse
import json
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path

import requests

from app.eval.metrics import recall_at_k, mrr, citation_correctness

EVAL_SET_PATH = Path(__file__).parent / "eval_set.json"
RESULTS_DIR = Path(__file__).parent.parent.parent / "eval_results"


def get_git_commit() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "--short", "HEAD"]).decode().strip()
    except Exception:
        return "no-git"


def run_eval(api_url: str) -> dict:
    with open(EVAL_SET_PATH) as f:
        eval_data = json.load(f)

    per_question_results = []

    for q in eval_data["questions"]:
        resp = requests.post(f"{api_url}/query", json={"question": q["question"]}, timeout=60)
        resp.raise_for_status()
        result = resp.json()

        retrieved_ids = [c["chunk_id"] for c in result["citations"]]
        # Note: citations only reflect chunks the model actually cited. For a truer
        # retrieval-recall measurement, extend /query in a debug mode to also return
        # all retrieved chunk_ids pre-generation; wired here via the query log if needed.

        r_at_5 = recall_at_k(retrieved_ids, q["expected_chunk_ids"])
        rr = mrr(retrieved_ids, q["expected_chunk_ids"])

        declined = result["fallback_reason"] is not None
        correct_decline = declined == q["should_decline"]

        chunks_by_index = {c["source_index"]: "" for c in result["citations"]}  # text not returned by API; fetch via DB if deeper analysis needed
        cite_correctness = citation_correctness(result["answer"], result["citations"], chunks_by_index)

        per_question_results.append({
            "id": q["id"],
            "category": q["category"],
            "question": q["question"],
            "answer": result["answer"],
            "confidence": result["confidence"],
            "fallback_reason": result["fallback_reason"],
            "recall_at_5": r_at_5,
            "mrr": rr,
            "correct_decline_behavior": correct_decline,
            "citation_correctness": cite_correctness,
        })

    summary = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "git_commit": get_git_commit(),
        "num_questions": len(per_question_results),
        "avg_recall_at_5": sum(r["recall_at_5"] for r in per_question_results) / len(per_question_results),
        "avg_mrr": sum(r["mrr"] for r in per_question_results) / len(per_question_results),
        "decline_accuracy": sum(r["correct_decline_behavior"] for r in per_question_results) / len(per_question_results),
        "avg_citation_correctness": sum(r["citation_correctness"] for r in per_question_results) / len(per_question_results),
        "per_question": per_question_results,
    }

    return summary


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--api-url", default="http://localhost:8000")
    args = parser.parse_args()

    print(f"Running eval against {args.api_url} ...")
    summary = run_eval(args.api_url)

    RESULTS_DIR.mkdir(exist_ok=True)
    out_path = RESULTS_DIR / f"{summary['git_commit']}_{int(time.time())}.json"
    with open(out_path, "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\nResults written to {out_path}\n")
    print(f"  Avg Recall@5:           {summary['avg_recall_at_5']:.3f}")
    print(f"  Avg MRR:                {summary['avg_mrr']:.3f}")
    print(f"  Decline accuracy:       {summary['decline_accuracy']:.3f}")
    print(f"  Avg citation correctness: {summary['avg_citation_correctness']:.3f}")


if __name__ == "__main__":
    main()
