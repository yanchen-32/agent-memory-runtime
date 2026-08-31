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
        "exact_match", "normalized_match", "answer_accuracy", "answer_f1",
        "memory_latency_ms", "context_build_latency_ms", "llm_latency_ms",
        "post_latency_ms", "end_to_end_latency_ms", "execution_sequence",
    }
    assert required.issubset(rows[0])
    assert rows[0]["recall@5"] is None
    assert rows[2]["latency_ms"] >= 0


def test_unified_runner_interleaves_agent_order_by_query():
    root = Path(__file__).resolve().parents[1]
    cases = load_jsonl(root / "benchmark" / "data" / "benchmark_v0.2.jsonl")
    rows, _ = run_benchmark(
        cases=cases[:2],
        agent_names=("B1", "Ours"),
        client_factory=RuleBasedClient,
        embedder_factory=lambda: HashEmbeddingModel(dim=64),
    )
    assert [row["agent"] for row in rows] == ["B1", "Ours", "Ours", "B1"]
    assert all(row["execution_policy"] == "query_interleaved_alternating" for row in rows)


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
