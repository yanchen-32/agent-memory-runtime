from pathlib import Path

from agent import RuleBasedClient
from benchmark import load_jsonl
from benchmark.runner import run_benchmark
from memory import HashEmbeddingModel


def test_unified_runner_emits_comparable_fields():
    root = Path(__file__).resolve().parents[1]
    cases = load_jsonl(root / "benchmark" / "data" / "benchmark_v0.2.jsonl")
    rows, summary = run_benchmark(
        cases=cases[:2],
        agent_names=("B0", "B1", "B2", "Ours"),
        client_factory=RuleBasedClient,
        embedder_factory=lambda: HashEmbeddingModel(dim=64),
    )
    assert len(rows) == 8
    assert {item["agent"] for item in summary} == {"B0", "B1", "B2", "Ours"}
    required = {
        "correct", "prompt_tokens", "token_count", "context_tokens",
        "latency_ms", "recall@1", "recall@5", "recall@10", "mrr",
    }
    assert required.issubset(rows[0])
    assert rows[0]["recall@5"] is None
    assert rows[2]["latency_ms"] >= 0


def test_unified_runner_accepts_b3():
    root = Path(__file__).resolve().parents[1]
    cases = load_jsonl(root / "benchmark" / "data" / "benchmark_v0.2.jsonl")
    rows, summary = run_benchmark(
        cases=cases[:1],
        agent_names=("B3",),
        client_factory=RuleBasedClient,
        embedder_factory=lambda: HashEmbeddingModel(dim=64),
    )
    assert len(rows) == 1
    assert rows[0]["agent"] == "B3"
    assert summary[0]["agent"] == "B3"
    assert rows[0]["retrieval_supported"] is True
