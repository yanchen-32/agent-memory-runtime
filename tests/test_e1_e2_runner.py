from pathlib import Path

from agent import RuleBasedClient
from benchmark import BenchmarkCase, QUANTITY_ANSWER_SCORER_VERSION, load_jsonl
from benchmark.runner import run_case
from memory import HashEmbeddingModel


class TemporalContractClient:
    """Fails closed when the historical evidence contract is absent."""

    last_usage: dict = {}

    def generate(self, prompt: str) -> str:
        assert "QUERY_TIME[2026-01-15T09:00:00+08:00]" in prompt
        assert "VALID_FROM[2026-01-01T09:00:00+08:00]" in prompt
        assert "VALID_TO[2026-02-01T09:00:00+08:00]" in prompt
        assert "SQLite" in prompt
        assert "openGauss" not in prompt
        return "SQLite"


class QuantityUnitClient:
    last_usage: dict = {}

    def generate(self, prompt: str) -> str:
        return "5条。"


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


def test_ours_historical_answer_receives_query_time_and_validity_interval():
    root = Path(__file__).resolve().parents[1]
    cases = load_jsonl(root / "benchmark" / "data" / "v1.0" / "development.jsonl")
    historical = next(case for case in cases if case.case_id == "dev_update_db_history_001")

    row = run_case(
        "Ours",
        historical,
        TemporalContractClient,
        lambda: HashEmbeddingModel(dim=64),
    )

    assert row["answer_accuracy"] == 1
    assert row["historical_query_correct"] == 1
    assert row["historical_retrieval_correct"] == 1
    assert row["forbidden_retrieved_count"] == 0


def test_v1_80_token_budget_does_not_receive_historical_prompt_overhead():
    root = Path(__file__).resolve().parents[1]
    cases = load_jsonl(root / "benchmark" / "data" / "v1.0" / "development.jsonl")
    budget = next(case for case in cases if case.case_id == "dev_budget_topk_001")

    for agent_name in ("B1", "B2", "B3", "Ours"):
        row = run_case(
            agent_name,
            budget,
            RuleBasedClient,
            lambda: HashEmbeddingModel(dim=64),
        )
        assert row["budget_satisfied"] is True
        assert row["budget_after_prompt_tokens"] <= 80


def test_runner_reports_budget_semantics_format_and_joint_success_separately():
    case = BenchmarkCase(
        case_id="typed_budget_001",
        category="budget",
        conversation=[],
        query="每次取回多少条候选记忆？",
        expected_memory_ids=[],
        expected_answer="5",
        expected_version="",
        query_time="2026-09-01T00:00:00+08:00",
        difficulty="easy",
        token_budget=80,
        answer_spec={
            "type": "quantity",
            "canonical_value": "5",
            "units": ["条"],
            "unit_policy": "optional",
            "output_format": "bare_value",
            "scorer_version": QUANTITY_ANSWER_SCORER_VERSION,
        },
    )

    row = run_case(
        "B0",
        case,
        QuantityUnitClient,
        lambda: HashEmbeddingModel(dim=64),
    )

    assert row["strict_answer_accuracy"] == 0
    assert row["semantic_answer_accuracy"] == 1
    assert row["answer_format_compliance"] == 0
    assert row["budget_satisfied"] is True
    assert row["budget_task_success"] == 1
