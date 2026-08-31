import pytest

from benchmark import (
    paired_bootstrap_comparison,
    paired_latency_reduction,
    trace_overhead_summary,
)


def test_paired_bootstrap_uses_per_case_repeat_medians():
    rows = [
        {"agent": "B1", "case_id": "a", "answer_f1": 0.2, "repeat": 1},
        {"agent": "B1", "case_id": "a", "answer_f1": 0.4, "repeat": 2},
        {"agent": "Ours", "case_id": "a", "answer_f1": 0.6, "repeat": 1},
        {"agent": "Ours", "case_id": "a", "answer_f1": 0.8, "repeat": 2},
        {"agent": "B1", "case_id": "b", "answer_f1": 0.5, "repeat": 1},
        {"agent": "Ours", "case_id": "b", "answer_f1": 0.5, "repeat": 1},
    ]
    result = paired_bootstrap_comparison(rows, samples=100, seed=7)
    assert result["num_paired_cases"] == 2
    assert result["mean_delta"] == pytest.approx(0.2)
    assert result["point_estimate_non_decrease"] is True


def test_paired_latency_reduction_reports_target_without_claiming_it():
    rows = [
        {"agent": "B1", "case_id": "a", "end_to_end_latency_ms": 100},
        {"agent": "Ours", "case_id": "a", "end_to_end_latency_ms": 40},
    ]
    result = paired_latency_reduction(rows)
    assert result["latency_reduction"] == 0.6
    assert result["target_met"] is True


def test_trace_overhead_uses_paired_case_medians():
    rows = [
        {"case_id": "a", "trace_enabled": False, "end_to_end_latency_ms": 10},
        {"case_id": "a", "trace_enabled": True, "end_to_end_latency_ms": 12},
    ]
    result = trace_overhead_summary(rows)
    assert result["trace_overhead"] == pytest.approx(0.2)
