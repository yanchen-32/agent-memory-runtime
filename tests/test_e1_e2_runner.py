from pathlib import Path

from agent import RuleBasedClient
from benchmark import load_jsonl
from benchmark.runner import run_case
from memory import HashEmbeddingModel


def test_runner_emits_historical_and_budget_fields():
    root = Path(__file__).resolve().parents[1]
    cases = load_jsonl(root / "benchmark" / "data" / "benchmark_v0.2.jsonl")
    temporal = next(case for case in cases if case.case_id == "temporal_002")
    budget = next(case for case in cases if case.case_id == "budget_001")

    temporal_row = run_case(
        "Ours", temporal, RuleBasedClient, lambda: HashEmbeddingModel(dim=64)
    )
    budget_row = run_case(
        "Ours", budget, RuleBasedClient, lambda: HashEmbeddingModel(dim=64)
    )

    assert temporal_row["historical_query_correct"] in (0, 1)
    assert temporal_row["historical_retrieval_correct"] == 1
    assert temporal_row["query_time"] is not None
    assert budget_row["budget_before_prompt_tokens"] is not None
    assert budget_row["budget_after_prompt_tokens"] is not None
    assert budget_row["budget_satisfied"] is True
    assert budget_row["budget_accuracy_delta"] in (-1, 0, 1)
