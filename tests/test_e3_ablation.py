import json

import pytest

from agent.e3_ablation import E3AblationAgent
from experiments.run_e3_ablation import (
    build_ablation_analysis,
    validate_ablation_chain,
)
from benchmark.design_e3_v14 import (
    build_orthogonal_design,
    validate_orthogonal_design,
    write_orthogonal_design,
)
from memory import HashEmbeddingModel, MemoryRuntimeV1


class CaptureClient:
    def __init__(self):
        self.prompts = []

    def generate(self, prompt: str) -> str:
        self.prompts.append(prompt)
        return "GaussDB-01"


CONVERSATION = [
    {
        "memory_id": "fact-v1",
        "role": "user",
        "content": "极光记忆单元01数据库为 SQLite-01。",
        "valid_from": "2026-01-01T09:00:00+08:00",
    },
    {
        "memory_id": "fact-v2",
        "role": "user",
        "content": "极光记忆单元01数据库改为 GaussDB-01。",
        "valid_from": "2026-02-01T09:00:00+08:00",
    },
    {
        "memory_id": "noise-1",
        "role": "user",
        "content": "极光记忆单元01架构为 辅助配置-01。",
        "valid_from": "2026-03-01T09:00:00+08:00",
    },
    {
        "memory_id": "noise-2",
        "role": "user",
        "content": "旁路单元01数据库为 辅助值-01。",
        "valid_from": "2026-04-01T09:00:00+08:00",
    },
]
QUERY = "极光记忆单元01当前的数据库是什么？"
QUERY_TIME = "2026-05-01T09:00:00+08:00"


def _agent(variant: str) -> E3AblationAgent:
    client = CaptureClient()
    agent = E3AblationAgent(
        client,
        MemoryRuntimeV1(embedder=HashEmbeddingModel(dim=64)),
        variant=variant,
        top_k=5,
    )
    agent.ingest(CONVERSATION, user_id="benchmark", session_id="test")
    agent.answer(QUERY, query_time=QUERY_TIME)
    return agent


def test_e3_ablation_chain_is_cumulative_and_single_variable():
    validate_ablation_chain(["A0", "A1", "A2", "A3", "A4"])
    with pytest.raises(ValueError, match="single-variable"):
        validate_ablation_chain(["A0", "A2"])


def test_e3_ablation_prompt_contracts_and_header_only_control():
    a0 = _agent("A0")
    a1 = _agent("A1")
    a2 = _agent("A2")
    a3 = _agent("A3")
    a4 = _agent("A4")

    assert "数据库改为 GaussDB-01" in a0.last_context
    assert "CURRENT_VALUE[GaussDB-01]" in a1.last_context
    assert "数据库改为 GaussDB-01" not in a1.last_context
    assert "QUERY_TIME[2026-05-01T09:00:00+08:00]" in a2.last_prompt
    assert "VALID_FROM[" in a2.last_context
    assert "VALID_TO[" in a2.last_context
    assert a3.last_spo_filtered_count > 0
    assert a3.last_context.count("MEMORY[") == 1
    assert "CURRENT_VALUE[GaussDB-01]" in a3.last_context
    assert a3.last_context == a4.last_context
    assert a3.last_retrieved_contents == a4.last_retrieved_contents
    assert a3.last_prompt.splitlines()[1:] == a4.last_prompt.splitlines()[1:]
    assert a3.last_prompt.splitlines()[0] != a4.last_prompt.splitlines()[0]


def test_e3_ablation_analysis_enforces_a3_a4_context_equivalence():
    rows = []
    for variant in ("A3", "A4"):
        for repeat in range(1, 4):
            rows.append({
                "status": "succeeded",
                "agent": variant,
                "variant": variant,
                "case_id": "case-1",
                "repeat": repeat,
                "e3_stratum": "short",
                "answer_f1": 1.0,
                "correct": 1,
                "context_sha256": "same-context",
                "prompt_sha256": f"{variant}-prompt",
                "retrieved_memory_ids": ["target"],
                "context_tokens": 10,
                "prompt_tokens": 20,
                "forbidden_retrieved_count": 0,
                "recall@5": 1.0,
                "mrr": 1.0,
            })
    report = build_ablation_analysis(
        rows,
        ["A3", "A4"],
        bootstrap_samples=10,
        expected_case_count=1,
        expected_repeats=3,
    )
    assert report["integrity"]["complete"] is True
    assert report["integrity"]["a3_a4_context_and_retrieval_identical"] is True
    assert report["integrity"]["a3_a4_prompt_changed"] is True
    assert report["formal_e3_admission_claim_authorized"] is False


def test_e3_v14_design_is_full_factorial_and_deterministic(tmp_path):
    rows = build_orthogonal_design()
    report = validate_orthogonal_design(rows)
    assert report["scenario_families"] == 144
    assert report["planned_development_cases"] == 576
    assert report["full_factorial"] is True
    assert report["pairwise_orthogonal"] is True
    first = write_orthogonal_design(tmp_path)
    second = write_orthogonal_design(tmp_path)
    assert first["design_sha256"] == second["design_sha256"]
    assert json.loads((tmp_path / "design_manifest.json").read_text())["status"] == (
        "design_only_pending_candidate_generation"
    )
