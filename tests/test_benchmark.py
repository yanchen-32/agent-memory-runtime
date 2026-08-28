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
