from __future__ import annotations

from collections import defaultdict
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from agent import RuleBasedClient, VectorMemoryAgent
from benchmark import load_jsonl, retrieval_metrics
from memory import HashEmbeddingModel, VectorMemoryStore

BENCHMARK = ROOT / "benchmark" / "data" / "benchmark_v0.1.jsonl"
RESULTS = ROOT / "results"


def main() -> None:
    cases = load_jsonl(BENCHMARK)
    rows = []
    metric_sums = defaultdict(float)
    metric_count = 0

    for case in cases:
        store = VectorMemoryStore(HashEmbeddingModel(dim=384))
        for turn in case.conversation:
            store.add(
                turn["content"],
                metadata={"role": turn.get("role", "user")},
                memory_id=turn["memory_id"],
            )
        agent = VectorMemoryAgent(RuleBasedClient(), store, top_k=5)
        hits = store.search(case.query, top_k=10)
        ranked_ids = [h.memory_id for h in hits]
        metrics = retrieval_metrics(ranked_ids, case.expected_memory_ids)
        if case.expected_memory_ids:
            metric_count += 1
            for key, value in metrics.items():
                metric_sums[key] += value
        pred = agent.answer(case.query)
        rows.append({
            "case_id": case.case_id,
            "category": case.category,
            "ranked_memory_ids": ranked_ids,
            "scores": [round(h.score, 6) for h in hits],
            "prediction": pred,
            "expected_answer": case.expected_answer,
            "retrieval_metrics": metrics,
        })

    RESULTS.mkdir(exist_ok=True)
    out = RESULTS / "b2_vector_memory_v0.1.json"
    out.write_text(json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8")

    print("B2 retrieval metrics on answerable cases (offline HashEmbeddingModel):")
    for key in sorted(metric_sums):
        print(f"  {key}: {metric_sums[key] / max(metric_count, 1):.3f}")
    print("NOTE: replace HashEmbeddingModel with a real embedding model for reportable experiments.")
    print(f"saved: {out}")


if __name__ == "__main__":
    main()
