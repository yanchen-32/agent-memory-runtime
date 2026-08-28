from __future__ import annotations

import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from agent import NoMemoryAgent, RuleBasedClient
from benchmark import load_jsonl

BENCHMARK = ROOT / "benchmark" / "data" / "benchmark_v0.1.jsonl"
RESULTS = ROOT / "results"


def main() -> None:
    agent = NoMemoryAgent(RuleBasedClient())
    cases = load_jsonl(BENCHMARK)
    rows = []
    correct = 0
    for case in cases:
        pred = agent.answer(case.query)
        is_correct = pred.strip() == case.expected_answer.strip()
        correct += int(is_correct)
        rows.append({
            "case_id": case.case_id,
            "category": case.category,
            "prediction": pred,
            "expected_answer": case.expected_answer,
            "correct": is_correct,
        })

    RESULTS.mkdir(exist_ok=True)
    out = RESULTS / "b0_no_memory_v0.1.json"
    out.write_text(json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8")
    print("B0 smoke-test client (RuleBasedClient)")
    print(f"exact-answer accuracy: {correct / len(cases):.3f} ({correct}/{len(cases)})")
    print("NOTE: this score is a pipeline smoke test, not the final LLM baseline score.")
    print(f"saved: {out}")


if __name__ == "__main__":
    main()
