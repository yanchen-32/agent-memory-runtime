from collections import Counter
import json

import pytest

from benchmark.generate_e3_v13 import (
    DEFAULT_SEED,
    SCENARIOS_PER_STRATUM,
    STRATA,
    build_development,
    write_development,
)
from benchmark.audit_e3_v13 import write_audit
from benchmark.execution import require_frozen_benchmark
from benchmark.freeze_e3_v13 import freeze_benchmark_e3_v13
from benchmark.prereview_e3_v13 import prereview_candidates
from benchmark.validate_e3_v13 import validate_development
from experiments.run_e3_scaling import build_e3_analysis


EXPECTED_DEVELOPMENT_SHA256 = "159b9a723086e4728e3f99ab08f531bf5efcf5e3a464a41c95c82c4e6010d02f"


@pytest.fixture(scope="module")
def e3_cases():
    return build_development(DEFAULT_SEED)


def test_e3_generator_has_deterministic_matched_strata(e3_cases):
    assert len(e3_cases) == SCENARIOS_PER_STRATUM * len(STRATA)
    assert Counter(case["metadata"]["stratum"] for case in e3_cases) == Counter(
        {stratum: SCENARIOS_PER_STRATUM for stratum in STRATA}
    )
    assert [case["metadata"]["stratum"] for case in e3_cases[:4]] == list(STRATA)
    families = {}
    for case in e3_cases:
        families.setdefault(case["metadata"]["scenario_family"], []).append(case)
    assert len(families) == SCENARIOS_PER_STRATUM
    for cases in families.values():
        by_stratum = {case["metadata"]["stratum"]: case for case in cases}
        ordered = [by_stratum[stratum] for stratum in STRATA]
        assert len({case["query"] for case in ordered}) == 1
        assert len({case["expected_answer"] for case in ordered}) == 1
        assert len({tuple(case["expected_memory_ids"]) for case in ordered}) == 1
        assert len({tuple(case["forbidden_memory_ids"]) for case in ordered}) == 1
        for lower, upper in zip(ordered, ordered[1:]):
            assert upper["conversation"][: len(lower["conversation"])] == lower["conversation"]
            assert len(upper["conversation"]) > len(lower["conversation"])


def test_e3_written_candidate_validates_and_hash_is_stable(tmp_path):
    manifest = write_development(tmp_path, DEFAULT_SEED)
    report = validate_development(tmp_path / "development.jsonl")
    assert manifest["development_sha256"] == EXPECTED_DEVELOPMENT_SHA256
    assert report["sha256"] == EXPECTED_DEVELOPMENT_SHA256
    assert report["prefix_nested"] is True
    assert report["answer_leakage_free"] is True
    assert report["holdout_generated"] is False
    assert report["strata"] == {
        stratum: SCENARIOS_PER_STRATUM for stratum in STRATA
    }
    assert len(report["predicate_distribution"]) == 6
    assert len(report["query_template_distribution"]) == 8
    assert set(report["target_position_distribution"]) == {"front", "middle", "back"}


def test_e3_prereview_audit_one_signoff_and_freeze(tmp_path):
    write_development(tmp_path, DEFAULT_SEED)
    prereview = prereview_candidates(tmp_path)
    audit = write_audit(tmp_path)
    assert prereview["checked_cases"] == SCENARIOS_PER_STRATUM * len(STRATA)
    assert prereview["technical_failures"] == 0
    assert audit["valid"] is True
    assert audit["technical_fields_all_yes"] is True
    assert audit["human_signatures_present"] == 0

    with pytest.raises(ValueError, match="one-time human review signoff incomplete"):
        freeze_benchmark_e3_v13(tmp_path)

    signoff_path = tmp_path / "review_signoff.json"
    signoff = json.loads(signoff_path.read_text(encoding="utf-8"))
    signoff["reviewer"] = "test-reviewer"
    signoff["decision"] = "approved"
    signoff_path.write_text(
        json.dumps(signoff, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    frozen = freeze_benchmark_e3_v13(tmp_path)
    assert frozen["status"] == "frozen"
    assert frozen["reviewed_case_count"] == SCENARIOS_PER_STRATUM * len(STRATA)
    assert require_frozen_benchmark(tmp_path / "development.jsonl")["status"] == "frozen"


def _passing_rows():
    rows = []
    for stratum in STRATA:
        for case_index in range(24):
            case_id = f"{stratum}-{case_index}"
            for repeat in range(1, 4):
                rows.extend([
                    {
                        "status": "succeeded",
                        "agent": "B1",
                        "case_id": case_id,
                        "repeat": repeat,
                        "e3_stratum": stratum,
                        "correct": 1,
                        "answer_f1": 0.9,
                        "context_tokens": float(STRATA[stratum]),
                        "latency_ms": 100.0,
                        "end_to_end_latency_ms": 100.0,
                        "forbidden_retrieved_count": 0,
                    },
                    {
                        "status": "succeeded",
                        "agent": "Ours",
                        "case_id": case_id,
                        "repeat": repeat,
                        "e3_stratum": stratum,
                        "correct": 1,
                        "answer_f1": 1.0,
                        "context_tokens": 50.0,
                        "latency_ms": 40.0 if stratum == "very_long" else 80.0,
                        "end_to_end_latency_ms": 40.0 if stratum == "very_long" else 80.0,
                        "forbidden_retrieved_count": 0,
                    },
                ])
    return rows


def test_e3_analysis_requires_all_predeclared_gates():
    report = build_e3_analysis(_passing_rows(), bootstrap_samples=100)
    assert report["gates"]["answer_f1_non_decrease_every_stratum"]["passed"] is True
    assert report["gates"]["ours_context_tokens_stable"]["passed"] is True
    assert report["gates"]["ours_forbidden_retrieval_zero"]["passed"] is True
    assert report["gates"]["very_long_e2e_latency_reduction_50pct"]["passed"] is True
    assert report["gates"]["all_admission_gates"]["passed"] is True
    assert report["gates"]["all_admission_gates"]["latency_claim_authorized"] is True


def test_e3_candidate_pilot_cannot_become_formal_evidence():
    report = build_e3_analysis(
        _passing_rows(),
        bootstrap_samples=100,
        benchmark_review_enforced=False,
    )
    assert report["gates"]["formal_evidence_eligible"]["passed"] is False
    assert report["gates"]["all_admission_gates"]["passed"] is False
    assert report["gates"]["all_admission_gates"]["latency_claim_authorized"] is False
