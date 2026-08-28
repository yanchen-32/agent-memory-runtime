from pathlib import Path

from benchmark import load_jsonl


def test_benchmark_v01_schema_and_categories():
    path = Path(__file__).resolve().parents[1] / "benchmark" / "data" / "benchmark_v0.1.jsonl"
    cases = load_jsonl(path)
    assert len(cases) == 16
    categories = {c.category for c in cases}
    assert categories == {
        "fact_recall", "semantic_recall", "temporal", "update",
        "conflict", "long_context", "noise", "abstention"
    }


def test_benchmark_v02_adds_long_horizon_scenarios():
    path = Path(__file__).resolve().parents[1] / "benchmark" / "data" / "benchmark_v0.2.jsonl"
    cases = load_jsonl(path)
    assert len(cases) == 28
    categories = {c.category for c in cases}
    assert {"temporal", "budget", "multi_hop", "forgetting"} <= categories
    budget_cases = [case for case in cases if case.category == "budget"]
    assert all(case.token_budget is not None for case in budget_cases)
    forgetting_cases = [case for case in cases if case.category == "forgetting"]
    assert all(case.forget_memory_ids for case in forgetting_cases)
