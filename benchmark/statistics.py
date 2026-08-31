from __future__ import annotations

from collections import defaultdict
import random
from statistics import mean, median
from typing import Iterable


def _percentile(values: list[float], probability: float) -> float:
    if not values:
        raise ValueError("percentile requires at least one value")
    ordered = sorted(values)
    position = probability * (len(ordered) - 1)
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    fraction = position - lower
    return ordered[lower] * (1.0 - fraction) + ordered[upper] * fraction


def paired_bootstrap_comparison(
    rows: Iterable[dict],
    baseline: str = "B1",
    treatment: str = "Ours",
    metric: str = "answer_f1",
    samples: int = 10_000,
    seed: int = 202601,
) -> dict[str, object]:
    """Compare per-case repeat medians with a paired case bootstrap."""
    if samples < 1:
        raise ValueError("samples must be >= 1")

    grouped: dict[tuple[str, str], list[float]] = defaultdict(list)
    for row in rows:
        value = row.get(metric)
        if row.get("agent") in {baseline, treatment} and value is not None:
            grouped[(str(row["case_id"]), str(row["agent"]))].append(float(value))

    pairs: list[tuple[str, float, float]] = []
    case_ids = sorted({case_id for case_id, _ in grouped})
    for case_id in case_ids:
        baseline_values = grouped.get((case_id, baseline))
        treatment_values = grouped.get((case_id, treatment))
        if baseline_values and treatment_values:
            pairs.append((case_id, median(baseline_values), median(treatment_values)))

    if not pairs:
        return {
            "baseline": baseline,
            "treatment": treatment,
            "metric": metric,
            "num_paired_cases": 0,
            "bootstrap_samples": samples,
            "seed": seed,
            "baseline_mean": None,
            "treatment_mean": None,
            "mean_delta": None,
            "ci95": [None, None],
            "point_estimate_non_decrease": None,
            "statistically_supported_non_decrease": None,
        }

    deltas = [treatment_value - baseline_value for _, baseline_value, treatment_value in pairs]
    rng = random.Random(seed)
    bootstrap_means = [mean(rng.choice(deltas) for _ in deltas) for _ in range(samples)]
    lower = _percentile(bootstrap_means, 0.025)
    upper = _percentile(bootstrap_means, 0.975)
    delta = mean(deltas)
    return {
        "baseline": baseline,
        "treatment": treatment,
        "metric": metric,
        "num_paired_cases": len(pairs),
        "bootstrap_samples": samples,
        "seed": seed,
        "repeat_reducer": "median",
        "baseline_mean": mean(pair[1] for pair in pairs),
        "treatment_mean": mean(pair[2] for pair in pairs),
        "mean_delta": delta,
        "ci95": [lower, upper],
        "point_estimate_non_decrease": delta >= 0.0,
        "statistically_supported_non_decrease": lower >= 0.0,
        "paired_case_ids": [pair[0] for pair in pairs],
    }


def paired_latency_reduction(
    rows: Iterable[dict],
    baseline: str = "B1",
    treatment: str = "Ours",
    metric: str = "end_to_end_latency_ms",
) -> dict[str, object]:
    """Reduce repeats by per-case median and report paired E2E reduction."""
    grouped: dict[tuple[str, str], list[float]] = defaultdict(list)
    for row in rows:
        value = row.get(metric)
        if row.get("agent") in {baseline, treatment} and value is not None:
            grouped[(str(row["case_id"]), str(row["agent"]))].append(float(value))
    pairs = []
    for case_id in sorted({case_id for case_id, _ in grouped}):
        baseline_values = grouped.get((case_id, baseline))
        treatment_values = grouped.get((case_id, treatment))
        if baseline_values and treatment_values:
            pairs.append((median(baseline_values), median(treatment_values)))
    if not pairs:
        return {
            "baseline": baseline,
            "treatment": treatment,
            "metric": metric,
            "num_paired_cases": 0,
            "latency_reduction": None,
        }
    baseline_mean = mean(pair[0] for pair in pairs)
    treatment_mean = mean(pair[1] for pair in pairs)
    return {
        "baseline": baseline,
        "treatment": treatment,
        "metric": metric,
        "num_paired_cases": len(pairs),
        "repeat_reducer": "median",
        "baseline_mean_ms": baseline_mean,
        "treatment_mean_ms": treatment_mean,
        "latency_reduction": (
            1.0 - treatment_mean / baseline_mean if baseline_mean > 0.0 else None
        ),
        "target_reduction": 0.50,
        "target_met": (
            treatment_mean <= baseline_mean * 0.50 if baseline_mean > 0.0 else None
        ),
    }


def trace_overhead_summary(
    rows: Iterable[dict],
    metric: str = "end_to_end_latency_ms",
) -> dict[str, object]:
    """Compare per-case medians for otherwise identical trace off/on runs."""
    grouped: dict[tuple[str, bool], list[float]] = defaultdict(list)
    for row in rows:
        value = row.get(metric)
        if value is not None and row.get("trace_enabled") in {True, False}:
            grouped[(str(row["case_id"]), bool(row["trace_enabled"]))].append(float(value))
    pairs = []
    for case_id in sorted({case_id for case_id, _ in grouped}):
        off = grouped.get((case_id, False))
        on = grouped.get((case_id, True))
        if off and on:
            pairs.append((median(off), median(on)))
    if not pairs:
        return {
            "metric": metric,
            "num_paired_cases": 0,
            "trace_overhead": None,
        }
    off_mean = mean(pair[0] for pair in pairs)
    on_mean = mean(pair[1] for pair in pairs)
    return {
        "metric": metric,
        "num_paired_cases": len(pairs),
        "repeat_reducer": "median",
        "trace_off_mean_ms": off_mean,
        "trace_on_mean_ms": on_mean,
        "trace_overhead": (on_mean - off_mean) / off_mean if off_mean > 0.0 else None,
    }
