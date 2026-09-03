"""Structural and leakage validation for v1.3 E3 Development candidates."""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from datetime import datetime
import json
from pathlib import Path

from memory import RuleMemoryExtractor

from .artifacts import sha256_file
from .generate_e3_v13 import (
    BENCHMARK_VERSION,
    GENERATOR_VERSION,
    SCENARIOS_PER_STRATUM,
    STRATA,
    TOKEN_TOLERANCE_RATIO,
    b1_prompt_tokens,
)
from .loader import BenchmarkCase, load_jsonl
from .validate_v1 import validate_file


REQUIRED_METADATA = {
    "benchmark_version",
    "source_type",
    "derivation_type",
    "split",
    "scenario_family",
    "scenario_index",
    "stratum",
    "target_b1_prompt_tokens",
    "b1_prompt_tokens",
    "history_memory_count",
    "subject",
    "predicate",
    "old_answer",
    "query_mode",
    "author",
    "review_status",
    "generator_version",
}


def _point(value: object, field: str, case_id: str) -> datetime:
    try:
        parsed = datetime.fromisoformat(str(value))
    except ValueError as exc:
        raise ValueError(f"{case_id}: invalid {field}: {value}") from exc
    if parsed.tzinfo is None:
        raise ValueError(f"{case_id}: {field} must include timezone")
    return parsed


def _validate_case(case: BenchmarkCase) -> None:
    metadata = case.metadata
    if set(metadata) < REQUIRED_METADATA:
        missing = sorted(REQUIRED_METADATA - set(metadata))
        raise ValueError(f"{case.case_id}: missing E3 metadata: {missing}")
    if case.category != "long_context":
        raise ValueError(f"{case.case_id}: E3 category must be long_context")
    if metadata["benchmark_version"] != BENCHMARK_VERSION:
        raise ValueError(f"{case.case_id}: wrong benchmark version")
    if metadata["generator_version"] != GENERATOR_VERSION:
        raise ValueError(f"{case.case_id}: wrong generator version")
    if metadata["source_type"] != "project_owned":
        raise ValueError(f"{case.case_id}: E3 cases must be project-owned")
    if metadata["derivation_type"] != "controlled_long_context_scaling":
        raise ValueError(f"{case.case_id}: invalid derivation type")
    if metadata["split"] != "development":
        raise ValueError(f"{case.case_id}: only Development is allowed")
    if metadata["review_status"] != "pending_human_review":
        raise ValueError(f"{case.case_id}: candidate must remain pending human review")
    if metadata["query_mode"] != "current":
        raise ValueError(f"{case.case_id}: E3 query must ask for current state")

    stratum = str(metadata["stratum"])
    if stratum not in STRATA:
        raise ValueError(f"{case.case_id}: unknown stratum {stratum}")
    target_tokens = STRATA[stratum]
    if int(metadata["target_b1_prompt_tokens"]) != target_tokens:
        raise ValueError(f"{case.case_id}: target token metadata mismatch")
    measured = b1_prompt_tokens(case.conversation, case.query)
    if int(metadata["b1_prompt_tokens"]) != measured:
        raise ValueError(f"{case.case_id}: measured B1 prompt tokens changed")
    if int(metadata["history_memory_count"]) != len(case.conversation):
        raise ValueError(f"{case.case_id}: history count metadata mismatch")
    tolerance = target_tokens * TOKEN_TOLERANCE_RATIO
    if abs(measured - target_tokens) > tolerance:
        raise ValueError(
            f"{case.case_id}: {measured} B1 tokens outside {target_tokens} +/- {tolerance}"
        )

    if len(case.expected_memory_ids) != 1 or len(case.forbidden_memory_ids) != 1:
        raise ValueError(f"{case.case_id}: requires one current target and one stale forbidden memory")
    old_id = case.forbidden_memory_ids[0]
    target_id = case.expected_memory_ids[0]
    memory_by_id = {str(turn["memory_id"]): turn for turn in case.conversation}
    old, target = memory_by_id[old_id], memory_by_id[target_id]
    if case.expected_version != "v2":
        raise ValueError(f"{case.case_id}: expected version must be v2")
    if old.get("valid_to") != target.get("valid_from"):
        raise ValueError(f"{case.case_id}: stale version must close at target valid_from")
    if target.get("valid_to") is not None:
        raise ValueError(f"{case.case_id}: current target cannot have valid_to")

    points = []
    for turn in case.conversation:
        created = _point(turn.get("created_at"), "created_at", case.case_id)
        valid_from = _point(turn.get("valid_from"), "valid_from", case.case_id)
        if created != valid_from:
            raise ValueError(f"{case.case_id}: created_at and valid_from differ")
        points.append(valid_from)
    if any(left >= right for left, right in zip(points, points[1:])):
        raise ValueError(f"{case.case_id}: history is not strictly chronological")
    query_time = _point(case.memory_query_time or case.query_time, "memory_query_time", case.case_id)
    if query_time <= points[-1]:
        raise ValueError(f"{case.case_id}: query must follow the appended history")

    answer_occurrences = [
        str(turn["memory_id"])
        for turn in case.conversation
        if case.expected_answer in str(turn.get("content", ""))
    ]
    if answer_occurrences != [target_id]:
        raise ValueError(f"{case.case_id}: answer leaked outside the target memory")
    subject = str(metadata["subject"])
    predicate = str(metadata["predicate"])
    old_answer = str(metadata["old_answer"])
    if old_answer == case.expected_answer:
        raise ValueError(f"{case.case_id}: stale and current answers are identical")
    if subject not in case.query or predicate not in case.query:
        raise ValueError(f"{case.case_id}: question does not preserve subject + predicate")

    extracted = RuleMemoryExtractor().extract([old, target])
    if len(extracted) != 2:
        raise ValueError(f"{case.case_id}: extractor did not retain both versions")
    old_fact, target_fact = extracted
    expected_key = (subject, predicate)
    if (old_fact.subject, old_fact.predicate) != expected_key:
        raise ValueError(f"{case.case_id}: stale fact key mismatch")
    if (target_fact.subject, target_fact.predicate) != expected_key:
        raise ValueError(f"{case.case_id}: target fact key mismatch")
    if old_fact.object_value != old_answer or target_fact.object_value != case.expected_answer:
        raise ValueError(f"{case.case_id}: extractor-visible answer mismatch")


def _validate_family(family: str, cases: list[BenchmarkCase]) -> None:
    by_stratum = {str(case.metadata["stratum"]): case for case in cases}
    if len(cases) != len(STRATA) or set(by_stratum) != set(STRATA):
        raise ValueError(f"{family}: requires exactly one case per E3 stratum")
    ordered = [by_stratum[stratum] for stratum in STRATA]
    first = ordered[0]
    invariant_fields = (
        "query",
        "expected_memory_ids",
        "forbidden_memory_ids",
        "expected_answer",
        "expected_version",
        "answer_aliases",
    )
    for case in ordered[1:]:
        for field in invariant_fields:
            if getattr(case, field) != getattr(first, field):
                raise ValueError(f"{family}: {field} changed across strata")
    for lower, upper in zip(ordered, ordered[1:]):
        if len(upper.conversation) <= len(lower.conversation):
            raise ValueError(f"{family}: history did not grow between strata")
        if upper.conversation[: len(lower.conversation)] != lower.conversation:
            raise ValueError(f"{family}: larger stratum is not a strict prefix extension")


def validate_development(path: str | Path) -> dict:
    target = Path(path)
    base_report = validate_file(target)
    cases = load_jsonl(target)
    expected_cases = SCENARIOS_PER_STRATUM * len(STRATA)
    if len(cases) != expected_cases:
        raise ValueError(f"expected {expected_cases} E3 Development cases, got {len(cases)}")
    strata = Counter(str(case.metadata.get("stratum")) for case in cases)
    expected_strata = Counter({name: SCENARIOS_PER_STRATUM for name in STRATA})
    if strata != expected_strata:
        raise ValueError(f"unexpected E3 stratum distribution: {dict(strata)}")

    families: defaultdict[str, list[BenchmarkCase]] = defaultdict(list)
    for case in cases:
        _validate_case(case)
        families[str(case.metadata["scenario_family"])].append(case)
    if len(families) != SCENARIOS_PER_STRATUM:
        raise ValueError(f"expected {SCENARIOS_PER_STRATUM} matched scenario families")
    for family, family_cases in families.items():
        _validate_family(family, family_cases)

    token_ranges = {}
    for stratum in STRATA:
        values = [
            int(case.metadata["b1_prompt_tokens"])
            for case in cases
            if case.metadata["stratum"] == stratum
        ]
        counts = [
            len(case.conversation)
            for case in cases
            if case.metadata["stratum"] == stratum
        ]
        token_ranges[stratum] = {
            "target_b1_prompt_tokens": STRATA[stratum],
            "min_b1_prompt_tokens": min(values),
            "max_b1_prompt_tokens": max(values),
            "min_history_memory_count": min(counts),
            "max_history_memory_count": max(counts),
        }
    return {
        **base_report,
        "benchmark_version": BENCHMARK_VERSION,
        "generator_version": GENERATOR_VERSION,
        "status": "pending_human_review",
        "scenario_families": len(families),
        "strata": dict(strata),
        "token_ranges": token_ranges,
        "prefix_nested": True,
        "answer_leakage_free": True,
        "holdout_generated": False,
        "sha256": sha256_file(target),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate v1.3 E3 Development candidates.")
    parser.add_argument(
        "path",
        type=Path,
        nargs="?",
        default=Path(__file__).resolve().parent / "data" / "v1.3-e3" / "development.jsonl",
    )
    args = parser.parse_args()
    print(json.dumps(validate_development(args.path), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
