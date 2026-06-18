"""
Evaluation runner — runs the engine on each test query, scores it against the
four milestone metrics, prints an aggregate report, and saves results to JSON.

Run from the project root:
    python -m evaluation.run_eval
"""

import json
from datetime import datetime

from config import settings
from evaluation.dataset import TEST_QUERIES
from evaluation.evaluators import ALL_EVALUATORS
from graph.build_graph import build_graph

# Pass thresholds from the project spec (as fractions).
THRESHOLDS = {
    "M1 Task Planning": 0.80,
    "M2 Context Offloading": 0.80,
    "M3 Delegation & Integration": 0.80,
    "M4 Report Quality": 0.70,
}


def run() -> None:
    settings.validate_settings()
    graph = build_graph()
    per_metric: dict[str, list[bool]] = {}
    records = []

    for i, query in enumerate(TEST_QUERIES, 1):
        print(f"\n[{i}/{len(TEST_QUERIES)}] Running: {query}")
        try:
            state = graph.invoke(
                {"messages": [{"role": "user", "content": query}]}
            )
        except Exception as exc:
            print(f"   ! Run failed: {exc}")
            records.append({"query": query, "error": str(exc)})
            continue

        query_result = {"query": query, "metrics": {}}
        for evaluator in ALL_EVALUATORS:
            try:
                res = evaluator(state)
            except Exception as exc:
                res = {"metric": evaluator.__name__, "passed": False,
                       "score": 0, "reason": f"evaluator error: {exc}"}
            per_metric.setdefault(res["metric"], []).append(res["passed"])
            query_result["metrics"][res["metric"]] = res
            mark = "PASS" if res["passed"] else "FAIL"
            print(f"   {mark}  {res['metric']}: {res['reason']}")
        records.append(query_result)

    print("\n" + "=" * 62)
    print("EVALUATION SUMMARY")
    print("=" * 62)
    summary = {}
    for metric, results in per_metric.items():
        rate = sum(results) / len(results) if results else 0.0
        threshold = THRESHOLDS.get(metric, 0.0)
        status = "PASS" if rate >= threshold else "BELOW TARGET"
        summary[metric] = {"pass_rate": rate, "threshold": threshold,
                           "status": status, "n": len(results)}
        print(f"{metric:30s} {rate*100:5.1f}%  "
              f"(target {threshold*100:.0f}%)  [{status}]")

    out = {
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "model": (settings.GROQ_MODEL if settings.LLM_PROVIDER == "groq"
                  else settings.GEMINI_MODEL),
        "n_queries": len(TEST_QUERIES),
        "summary": summary,
        "records": records,
    }
    with open("evaluation/results.json", "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2, default=str)
    print("\nSaved detailed results to evaluation/results.json")


if __name__ == "__main__":
    run()