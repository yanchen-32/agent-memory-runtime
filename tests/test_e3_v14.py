from __future__ import annotations

from agent import FullHistoryAgent, RuleBasedClient
from benchmark import ReusableEmbeddingFactory, load_jsonl
from benchmark.design_e3_v14 import build_orthogonal_design, validate_orthogonal_design
from benchmark.e3_v14_spec import QUERY_TEMPLATE_SPECS
from benchmark.runner import run_case
from experiments.run_e3_scaling import build_e3_analysis


DATA = "benchmark/data/v1.4-e3/development.jsonl"


def test_v14_design_is_full_factorial_with_semantic_query_modes():
    report = validate_orthogonal_design(build_orthogonal_design())
    assert report["valid"] is True
    assert report["scenario_families"] == 144
    assert report["planned_development_cases"] == 576
    assert {spec["mode"] for spec in QUERY_TEMPLATE_SPECS} >= {
        "canonical_literal",
        "predicate_alias",
        "nonliteral",
        "temporal_alias",
        "temporal_nonliteral",
    }


def test_v14_semantic_surfaces_use_structured_fast_path_and_temporal_b1_prompt():
    cases = load_jsonl(DATA)
    representatives = {}
    for case in cases:
        if case.metadata["stratum"] == "short":
            representatives.setdefault(case.metadata["query_expression_mode"], case)
    assert set(representatives) == {spec["mode"] for spec in QUERY_TEMPLATE_SPECS}

    embedder_factory = ReusableEmbeddingFactory("hash", "unused")
    for case in representatives.values():
        for agent_name in ("StructuredKV", "OursV2"):
            row = run_case(
                agent_name,
                case,
                RuleBasedClient,
                embedder_factory,
                repeat=1,
            )
            assert row["answer_f1"] == 1.0
            assert row["answer_route"] == "structured_fast_path"
            assert row["forbidden_retrieved_count"] == 0

    case = next(iter(representatives.values()))
    b1_agent = FullHistoryAgent(RuleBasedClient())
    b1_agent.answer(
        case.query,
        conversation=case.conversation,
        query_time=case.memory_query_time,
        temporal_context=True,
    )
    assert "QUERY_TIME[" in b1_agent.last_prompt
    assert "VALID_FROM[" in b1_agent.last_context
    assert "VALID_TO[" in b1_agent.last_context
    assert b1_agent.last_prompt_tokens >= 950


def test_v14_analysis_names_the_actual_treatment_agent():
    rows = []
    for stratum in ("short", "medium", "long", "very_long"):
        for agent, latency in (("B1", 100.0), ("OursV2", 40.0)):
            rows.append({
                "status": "succeeded",
                "agent": agent,
                "case_id": f"case-{stratum}",
                "repeat": 1,
                "e3_stratum": stratum,
                "correct": 1,
                "answer_f1": 1.0,
                "context_tokens": 50.0 if agent == "OursV2" else 1000.0,
                "latency_ms": latency,
                "end_to_end_latency_ms": latency,
                "forbidden_retrieved_count": 0,
            })
    report = build_e3_analysis(
        rows,
        bootstrap_samples=10,
        treatment_agent="OursV2",
    )
    assert "very_long_answer_f1_B1_vs_OursV2" in report["comparisons"]
    assert "very_long_e2e_latency_B1_vs_OursV2" in report["comparisons"]
    assert not any(key.endswith("B1_vs_Ours") for key in report["comparisons"])
