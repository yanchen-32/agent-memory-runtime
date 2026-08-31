import numpy as np
import pytest

from agent import LLMRequestError, RuleBasedClient
from benchmark import JsonlRunCheckpoint, ReusableEmbeddingFactory, load_jsonl
from benchmark.runner import run_benchmark


class CountingEmbedder:
    def __init__(self, _model_name="unused"):
        self.calls: list[list[str]] = []

    def encode(self, texts: list[str]) -> np.ndarray:
        self.calls.append(list(texts))
        vectors = np.zeros((len(texts), 8), dtype=np.float32)
        for index, text in enumerate(texts):
            vectors[index, hash(text) % 8] = 1.0
        return vectors


def test_reusable_embedding_factory_loads_once_and_batches_cache_misses():
    loads: list[CountingEmbedder] = []

    def loader(model_name):
        backend = CountingEmbedder(model_name)
        loads.append(backend)
        return backend

    factory = ReusableEmbeddingFactory(
        "sentence-transformers", "test-model", backend_factory=loader
    )
    first_case = factory()
    first_case.encode(["a", "b", "a"])
    first_case.encode(["b", "c"])
    second_case = factory()
    second_case.encode(["a"])

    assert len(loads) == 1
    assert loads[0].calls == [["a", "b"], ["c"], ["a"]]


def test_checkpoint_resumes_only_matching_configuration(tmp_path):
    path = tmp_path / "checkpoint.jsonl"
    checkpoint = JsonlRunCheckpoint(path, {"benchmark": "v1", "top_k": 5})
    assert checkpoint.prepare(resume=False) == []
    row = {"case_id": "c1", "agent": "B1", "repeat": 1, "status": "succeeded"}
    checkpoint.append(row)
    assert checkpoint.prepare(resume=True) == [row]

    incompatible = JsonlRunCheckpoint(path, {"benchmark": "v1", "top_k": 10})
    with pytest.raises(ValueError, match="configuration"):
        incompatible.prepare(resume=True)


def test_runner_records_terminal_failure_and_skips_completed_row():
    case = load_jsonl("benchmark/data/benchmark_v0.2.jsonl")[0]

    class FailedClient:
        def generate(self, prompt):
            raise LLMRequestError(
                "LLM request failed with HTTP 401",
                attempts=1,
                status_code=401,
                retryable=False,
            )

    failed_rows, failed_summary = run_benchmark(
        [case],
        ["B1"],
        FailedClient,
        ReusableEmbeddingFactory("hash", "unused", hash_dim=32),
        continue_on_error=True,
    )
    assert failed_rows[0]["status"] == "failed"
    assert failed_rows[0]["error_status_code"] == 401
    assert failed_summary[0]["num_failed"] == 1
    assert failed_summary[0]["latency_mean_ms"] is None

    emitted = []
    resumed_rows, _ = run_benchmark(
        [case],
        ["B1"],
        RuleBasedClient,
        ReusableEmbeddingFactory("hash", "unused", hash_dim=32),
        existing_rows=[{**failed_rows[0], "status": "succeeded"}],
        on_row=emitted.append,
    )
    assert len(resumed_rows) == 1
    assert emitted == []
