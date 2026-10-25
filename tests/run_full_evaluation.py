import os
import sys
import json
import time
from pathlib import Path

# Ensure stdout is unbuffered and uses UTF-8 to prevent Windows cp1252 crash
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace", line_buffering=True)

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from vector_db import QdrantStorage
from rag_graph import run_agentic_rag
from tests.rag_evaluator import evaluate_test_case

def precheck_vector_db():
    """Verifies that the vector database is accessible and not locked by another process."""
    try:
        store = QdrantStorage()
        stats = store.get_stats()
        print(f"Vector DB Status: {stats.status} ({stats.mode} mode, {stats.total_vectors} vectors indexed across {len(stats.sources)} document(s)).", flush=True)
        return True
    except Exception as e:
        err_msg = str(e)
        if "already accessed by another instance" in err_msg or "Permission denied" in err_msg:
            print("\n" + "!" * 70, flush=True)
            print("WARNING: Qdrant local storage is locked by another running process (such as Streamlit).", flush=True)
            print("Please stop the running Streamlit app (Ctrl+C in its terminal) before running this evaluation,", flush=True)
            print("or set QDRANT_MODE=remote to allow concurrent multi-process access.", flush=True)
            print("!" * 70 + "\n", flush=True)
        else:
            print(f"Vector DB Warning: {err_msg}", flush=True)
        return False

def run_evaluation(limit: int = None, category_filter: str = None):
    suite_path = Path(__file__).parent / "os_textbook_test_suite_50.json"
    with open(suite_path, "r", encoding="utf-8") as f:
        tests = json.load(f)

    if category_filter:
        tests = [t for t in tests if t["category"].lower() == category_filter.lower()]
        print(f"Filtered to category '{category_filter}': {len(tests)} test cases.", flush=True)

    if limit:
        tests = tests[:limit]

    print(f"\n=======================================================", flush=True)
    print(f"  Starting Red-Teaming QA Evaluation ({len(tests)} Questions)", flush=True)
    print(f"=======================================================\n", flush=True)

    precheck_vector_db()

    results = []
    passed_count = 0
    category_stats = {}

    for i, t in enumerate(tests, start=1):
        cat = t["category"]
        q = t["question"]
        expected = t["expected_behavior"]
        round_num = t.get("round", "")

        if cat not in category_stats:
            category_stats[cat] = {"total": 0, "pass": 0, "fail": 0}
        category_stats[cat]["total"] += 1

        round_label = f" (Round {round_num})" if round_num else ""
        print(f"[{i}/{len(tests)}]{round_label} [{cat}]", flush=True)
        print(f"  Q: {q}", flush=True)

        # 1. Execute RAG
        t0 = time.time()
        try:
            rag_output = run_agentic_rag(q)
            system_answer = rag_output.get("answer", "")
            citations = rag_output.get("citations", [])
            retrieved_chunks = citations
        except Exception as e:
            system_answer = f"System Error: {e}"
            retrieved_chunks = []
        elapsed = time.time() - t0

        # 2. Strict QA Judge
        verdict_obj = evaluate_test_case(
            category=cat,
            question=q,
            expected_behavior=expected,
            retrieved_chunks=retrieved_chunks,
            system_answer=system_answer,
        )

        is_pass = verdict_obj.get("verdict") == "PASS"
        if is_pass:
            passed_count += 1
            category_stats[cat]["pass"] += 1
            status_symbol = "PASS [OK]"
        else:
            category_stats[cat]["fail"] += 1
            status_symbol = "FAIL [X]"

        print(f"  -> Verdict: {status_symbol} ({elapsed:.1f}s)", flush=True)
        print(f"  -> Reason:  {verdict_obj.get('reason')}", flush=True)
        
        # If failed, show helpful debug breakdown
        if not is_pass:
            print(f"  -> Detailed Check: grounded={verdict_obj.get('grounded')}, no_fab={verdict_obj.get('no_fabrication')}, refusal={verdict_obj.get('correct_refusal')}, citation_valid={verdict_obj.get('citation_valid')}", flush=True)
            print(f"  -> System Answer Preview: {system_answer[:180]}...", flush=True)

        print("-" * 55, flush=True)

        results.append({
            "id": t["id"],
            "round": round_num,
            "category": cat,
            "question": q,
            "expected_behavior": expected,
            "system_answer": system_answer,
            "retrieved_chunks_count": len(retrieved_chunks),
            "evaluation": verdict_obj,
            "elapsed_seconds": round(elapsed, 2),
        })

    # Summary Report
    print("\n" + "=" * 55, flush=True)
    pass_pct = (passed_count / len(tests)) * 100 if tests else 0
    print(f"EVALUATION SUMMARY: {passed_count}/{len(tests)} Passed ({pass_pct:.1f}%)", flush=True)
    print("=" * 55, flush=True)
    for cat, stat in sorted(category_stats.items()):
        pct = (stat["pass"] / stat["total"]) * 100 if stat["total"] else 0
        status_icon = "OK" if pct >= 80 else "WARN"
        print(f"  [{status_icon}] {cat.ljust(22)}: {stat['pass']}/{stat['total']} ({pct:.0f}%)", flush=True)

    # Save detailed JSON output
    out_file = Path(__file__).parent / "evaluation_results.json"
    out_file.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(f"\nDetailed evaluation results saved to: {out_file}", flush=True)

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Run RAG QA Judge Evaluation")
    parser.add_argument("--limit", type=int, default=None, help="Limit number of test questions to run (e.g. 10 for 1 per category, 20 for 2 per category)")
    parser.add_argument("--category", type=str, default=None, help="Filter by specific category name")
    args = parser.parse_args()
    run_evaluation(limit=args.limit, category_filter=args.category)
