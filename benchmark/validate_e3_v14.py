"""Structural, semantic-surface, temporal, and leakage validation for E3 v1.4."""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from datetime import datetime
import json
from pathlib import Path

from memory import RuleMemoryExtractor, StructuredFactResolver

from .artifacts import sha256_file
from .design_e3_v14 import build_orthogonal_design, validate_orthogonal_design
from .e3_v14_spec import (
    DESIGN_VERSION,
    STRATA,
    TARGET_POSITION_SPECS,
    render_query,
)
from .generate_e3_v13 import DISTRACTOR_KINDS, TOKEN_TOLERANCE_RATIO
from .generate_e3_v14 import (
    BENCHMARK_VERSION,
    GENERATOR_VERSION,
    SCENARIOS_PER_STRATUM,
    b1_prompt_tokens,
)
from .loader import BenchmarkCase, load_jsonl
from .validate_v1 import validate_file


def _point(value: object, field: str, case_id: str) -> datetime:
    try:
        point = datetime.fromisoformat(str(value))
    except ValueError as exc:
        raise ValueError(f"{case_id}: invalid {field}: {value}") from exc
    if point.tzinfo is None:
        raise ValueError(f"{case_id}: {field} must include timezone")
    return point


def _distractor_kind(content: str, subject: str) -> str:
    if "审计索引" in content:
        return "lexical_decoy"
    if content.startswith("旁路单元"):
        return "same_predicate_other_subject"
    if content.startswith(subject):
        return "same_subject_other_predicate"
    return "unrelated"


def _validate_case(case: BenchmarkCase, design: dict) -> None:
    metadata = case.metadata
    required = {
        "benchmark_version", "generator_version", "design_version", "scenario_family",
        "scenario_index", "stratum", "target_b1_prompt_tokens", "b1_prompt_tokens",
        "history_memory_count", "subject", "predicate", "predicate_index",
        "predicate_surface", "query_expression_mode", "old_answer",
        "target_memory_ordinal", "target_position_index", "target_position_band",
        "distractor_kinds", "query_template_index", "query_mode", "temporal_prompt",
        "method_candidate_version", "prompt_candidate_version", "review_status",
    }
    missing = sorted(required - set(metadata))
    if missing:
        raise ValueError(f"{case.case_id}: missing v1.4 metadata: {missing}")
    if case.category != "long_context" or metadata["split"] != "development":
        raise ValueError(f"{case.case_id}: must be a Development long_context case")
    if metadata["benchmark_version"] != BENCHMARK_VERSION:
        raise ValueError(f"{case.case_id}: benchmark version mismatch")
    if metadata["generator_version"] != GENERATOR_VERSION or metadata["design_version"] != DESIGN_VERSION:
        raise ValueError(f"{case.case_id}: generator/design version mismatch")
    if metadata["review_status"] != "pending_human_review":
        raise ValueError(f"{case.case_id}: candidate must remain pending human review")
    if metadata["temporal_prompt"] is not True:
        raise ValueError(f"{case.case_id}: versioned E3 cases require temporal prompts")
    for field in (
        "scenario_family", "predicate_index", "query_template_index",
        "target_position_index", "target_position_band", "query_expression_mode",
    ):
        if metadata[field] != design[field]:
            raise ValueError(f"{case.case_id}: design field changed: {field}")

    stratum = str(metadata["stratum"])
    target_tokens = STRATA[stratum]
    measured = b1_prompt_tokens(case.conversation, case.query, str(case.memory_query_time))
    if int(metadata["b1_prompt_tokens"]) != measured:
        raise ValueError(f"{case.case_id}: measured B1 token count changed")
    if abs(measured - target_tokens) > target_tokens * TOKEN_TOLERANCE_RATIO:
        raise ValueError(f"{case.case_id}: B1 token count outside stratum tolerance")
    if int(metadata["history_memory_count"]) != len(case.conversation):
        raise ValueError(f"{case.case_id}: history count mismatch")

    subject = str(metadata["subject"])
    predicate = str(metadata["predicate"])
    query, mode, surface = render_query(subject, predicate, int(metadata["query_template_index"]))
    if (case.query, metadata["query_expression_mode"], metadata["predicate_surface"]) != (query, mode, surface):
        raise ValueError(f"{case.case_id}: semantic query surface mismatch")
    if subject not in case.query or surface not in case.query:
        raise ValueError(f"{case.case_id}: query lacks its declared subject/surface")
    if mode != "canonical_literal" and predicate in case.query:
        raise ValueError(f"{case.case_id}: alias/nonliteral query leaked canonical predicate")
    resolution = StructuredFactResolver().resolve(case.query, [(subject, predicate)])
    if resolution is None or resolution.predicate != predicate:
        raise ValueError(f"{case.case_id}: schema resolver cannot map the semantic query")

    if len(case.expected_memory_ids) != 1 or len(case.forbidden_memory_ids) != 1:
        raise ValueError(f"{case.case_id}: requires one target and one stale forbidden memory")
    memory_by_id = {str(turn["memory_id"]): turn for turn in case.conversation}
    old_id, target_id = case.forbidden_memory_ids[0], case.expected_memory_ids[0]
    if old_id not in memory_by_id or target_id not in memory_by_id:
        raise ValueError(f"{case.case_id}: target/stale evidence is missing")
    old, target = memory_by_id[old_id], memory_by_id[target_id]
    if old.get("valid_to") != target.get("valid_from") or target.get("valid_to") is not None:
        raise ValueError(f"{case.case_id}: version interval is invalid")
    extracted = RuleMemoryExtractor().extract([old, target])
    if len(extracted) != 2 or any((fact.subject, fact.predicate) != (subject, predicate) for fact in extracted):
        raise ValueError(f"{case.case_id}: target versions do not share exact SPO key")
    if extracted[0].object_value != metadata["old_answer"] or extracted[1].object_value != case.expected_answer:
        raise ValueError(f"{case.case_id}: extracted object value mismatch")
    occurrences = [
        str(turn["memory_id"]) for turn in case.conversation
        if case.expected_answer in str(turn.get("content", ""))
    ]
    if occurrences != [target_id]:
        raise ValueError(f"{case.case_id}: expected answer leaked outside target")

    points = []
    for turn in case.conversation:
        created = _point(turn.get("created_at"), "created_at", case.case_id)
        valid_from = _point(turn.get("valid_from"), "valid_from", case.case_id)
        if created != valid_from:
            raise ValueError(f"{case.case_id}: created_at differs from valid_from")
        points.append(valid_from)
    if any(left >= right for left, right in zip(points, points[1:])):
        raise ValueError(f"{case.case_id}: history is not strictly chronological")
    if _point(case.memory_query_time, "memory_query_time", case.case_id) <= points[-1]:
        raise ValueError(f"{case.case_id}: query_time must follow history")

    old_index = next(i for i, turn in enumerate(case.conversation) if turn["memory_id"] == old_id)
    target_index = next(i for i, turn in enumerate(case.conversation) if turn["memory_id"] == target_id)
    if target_index != old_index + 1 or int(metadata["target_memory_ordinal"]) != target_index + 1:
        raise ValueError(f"{case.case_id}: stale/current target placement mismatch")
    _, expected_old_position = TARGET_POSITION_SPECS[int(metadata["target_position_index"])]
    if target_index != expected_old_position + 1:
        raise ValueError(f"{case.case_id}: target position band mismatch")
    kinds = {
        _distractor_kind(str(turn["content"]), subject)
        for turn in case.conversation
        if turn["memory_id"] not in {old_id, target_id}
    }
    if kinds != set(DISTRACTOR_KINDS) or set(metadata["distractor_kinds"]) != kinds:
        raise ValueError(f"{case.case_id}: distractor diversity mismatch")


def validate_development(path: str | Path) -> dict:
    target = Path(path)
    base = validate_file(target)
    cases = load_jsonl(target)
    design_rows = build_orthogonal_design()
    design_report = validate_orthogonal_design(design_rows)
    design_by_family = {row["scenario_family"]: row for row in design_rows}
    expected_cases = SCENARIOS_PER_STRATUM * len(STRATA)
    if len(cases) != expected_cases:
        raise ValueError(f"expected {expected_cases} v1.4 cases, got {len(cases)}")
    expected_order = [
        (row["scenario_family"], stratum) for row in design_rows for stratum in STRATA
    ]
    actual_order = [
        (str(case.metadata["scenario_family"]), str(case.metadata["stratum"])) for case in cases
    ]
    if actual_order != expected_order:
        raise ValueError("v1.4 cases are not scenario-major/stratum-interleaved")
    families: defaultdict[str, list[BenchmarkCase]] = defaultdict(list)
    for case in cases:
        family = str(case.metadata["scenario_family"])
        _validate_case(case, design_by_family[family])
        families[family].append(case)
    for family, family_cases in families.items():
        by_stratum = {str(case.metadata["stratum"]): case for case in family_cases}
        if set(by_stratum) != set(STRATA):
            raise ValueError(f"{family}: incomplete strata")
        ordered = [by_stratum[stratum] for stratum in STRATA]
        first = ordered[0]
        for case in ordered[1:]:
            for field in ("query", "expected_memory_ids", "forbidden_memory_ids", "expected_answer"):
                if getattr(case, field) != getattr(first, field):
                    raise ValueError(f"{family}: invariant {field} changed")
        for lower, upper in zip(ordered, ordered[1:]):
            if len(upper.conversation) <= len(lower.conversation) or upper.conversation[:len(lower.conversation)] != lower.conversation:
                raise ValueError(f"{family}: histories are not strict prefix nested")
    family_cases = cases[::len(STRATA)]
    distributions = {
        "predicate": Counter(str(case.metadata["predicate"]) for case in family_cases),
        "query_template": Counter(str(case.metadata["query_template_index"]) for case in family_cases),
        "query_expression_mode": Counter(str(case.metadata["query_expression_mode"]) for case in family_cases),
        "target_position": Counter(str(case.metadata["target_position_band"]) for case in family_cases),
    }
    token_ranges = {}
    for stratum, expected in STRATA.items():
        selected = [case for case in cases if case.metadata["stratum"] == stratum]
        values = [int(case.metadata["b1_prompt_tokens"]) for case in selected]
        counts = [len(case.conversation) for case in selected]
        token_ranges[stratum] = {
            "target_b1_prompt_tokens": expected,
            "min_b1_prompt_tokens": min(values),
            "max_b1_prompt_tokens": max(values),
            "min_history_memory_count": min(counts),
            "max_history_memory_count": max(counts),
        }
    return {
        **base,
        "benchmark_version": BENCHMARK_VERSION,
        "generator_version": GENERATOR_VERSION,
        "design_version": DESIGN_VERSION,
        "design_valid": design_report["valid"],
        "status": "pending_human_review",
        "scenario_families": len(families),
        "strata": dict(Counter(str(case.metadata["stratum"]) for case in cases)),
        "distributions": {name: dict(sorted(values.items())) for name, values in distributions.items()},
        "token_ranges": token_ranges,
        "prefix_nested": True,
        "answer_leakage_free": True,
        "semantic_surfaces_valid": True,
        "test_generated": False,
        "holdout_generated": False,
        "case_ids": [case.case_id for case in cases],
        "sha256": sha256_file(target),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate v1.4 E3 Development candidates.")
    parser.add_argument(
        "path", type=Path, nargs="?",
        default=Path(__file__).resolve().parent / "data" / "v1.4-e3" / "development.jsonl",
    )
    args = parser.parse_args()
    print(json.dumps(validate_development(args.path), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
