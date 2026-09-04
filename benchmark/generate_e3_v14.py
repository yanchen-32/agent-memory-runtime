"""Deterministically generate the orthogonal E3 v1.4 Development candidate."""

from __future__ import annotations

import argparse
from bisect import bisect_left
import csv
import json
from pathlib import Path
import random

from agent import (
    FullHistoryAgent,
    MEMORY_RUNTIME_METHOD_VERSION,
    MEMORY_RUNTIME_PROMPT_VERSION,
    estimate_tokens,
)
from agent.base import memory_line

from .artifacts import sha256_file
from .design_e3_v14 import (
    build_orthogonal_design,
    validate_orthogonal_design,
    write_orthogonal_design,
)
from .e3_v14_spec import DESIGN_VERSION, PREDICATE_SPECS, STRATA, TARGET_POSITION_SPECS, render_query
from .generate_e3_v13 import (
    DEFAULT_SEED,
    MAX_MEMORY_COUNT,
    REVIEW_FIELDS,
    TOKEN_TOLERANCE_RATIO,
    _distractor_content,
    _memory,
    _scenario_answers,
    _time,
)


GENERATOR_VERSION = "v1.4-e3.1"
BENCHMARK_VERSION = "1.4-e3-candidate"
SCENARIOS_PER_STRATUM = 144


class _NoopClient:
    def generate(self, prompt: str) -> str:
        return ""


def b1_prompt_tokens(
    conversation: list[dict], query: str, query_time: str
) -> int:
    agent = FullHistoryAgent(_NoopClient())
    agent.answer(
        query,
        conversation=conversation,
        query_time=query_time,
        temporal_context=True,
    )
    return agent.last_prompt_tokens


def _temporal_prefix_token_counts(
    conversation: list[dict], query: str, query_time: str
) -> list[int]:
    total = b1_prompt_tokens([], query, query_time)
    counts = []
    for index, turn in enumerate(conversation, start=1):
        total += estimate_tokens(memory_line(
            index,
            str(turn.get("content", "")).strip(),
            temporal_context=True,
            valid_from=turn.get("valid_from") or turn.get("created_at"),
            valid_to=turn.get("valid_to"),
        ))
        counts.append(total)
    return counts


def _nearest_prefix(
    conversation: list[dict], prefix_counts: list[int], target: int
) -> tuple[list[dict], int]:
    """Choose the closest prefix, avoiding one-record overshoot bias."""
    upper = bisect_left(prefix_counts, target)
    candidates = [index for index in (upper - 1, upper) if 0 <= index < len(prefix_counts)]
    index = min(candidates, key=lambda item: (abs(prefix_counts[item] - target), -item))
    return conversation[: index + 1], prefix_counts[index]


def _full_history(design: dict) -> tuple[list[dict], dict]:
    scenario_index = int(design["scenario_family"].rsplit("_", 1)[1]) - 1
    predicate_index = int(design["predicate_index"])
    predicate = PREDICATE_SPECS[predicate_index][0]
    subject = f"极光记忆单元{scenario_index + 1:03d}"
    answer, old_answer = _scenario_answers(scenario_index, predicate_index)
    family = str(design["scenario_family"])
    old_id = f"{family}_fact_v1"
    target_id = f"{family}_fact_v2"
    _, old_position = TARGET_POSITION_SPECS[int(design["target_position_index"])]
    target_position = old_position + 1
    target_time = _time(scenario_index, target_position)
    query, expression_mode, predicate_surface = render_query(
        subject, predicate, int(design["query_template_index"])
    )
    rng = random.Random(int(design["scenario_seed"]))
    conversation = []
    distractor_kinds = set()
    for offset in range(MAX_MEMORY_COUNT):
        if offset == old_position:
            turn = _memory(
                old_id,
                f"{subject}{predicate}为 {old_answer}。",
                scenario_index,
                offset,
                valid_to=target_time,
            )
        elif offset == target_position:
            turn = _memory(
                target_id,
                f"{subject}{predicate}改为 {answer}。",
                scenario_index,
                offset,
            )
        else:
            content, kind = _distractor_content(
                scenario_index,
                offset,
                rng,
                subject=subject,
                predicate=predicate,
            )
            distractor_kinds.add(kind)
            turn = _memory(
                f"{family}_noise_{offset:04d}", content, scenario_index, offset
            )
        conversation.append(turn)
    return conversation, {
        "subject": subject,
        "predicate": predicate,
        "answer": answer,
        "old_answer": old_answer,
        "old_id": old_id,
        "target_id": target_id,
        "target_position": target_position,
        "query": query,
        "query_expression_mode": expression_mode,
        "predicate_surface": predicate_surface,
        "distractor_kinds": sorted(distractor_kinds),
    }


def build_development() -> list[dict]:
    design_rows = build_orthogonal_design()
    validate_orthogonal_design(design_rows)
    cases = []
    for scenario_index, design in enumerate(design_rows):
        history, spec = _full_history(design)
        counting_query_time = _time(scenario_index, MAX_MEMORY_COUNT + 1)
        prefix_counts = _temporal_prefix_token_counts(
            history, spec["query"], counting_query_time
        )
        previous_count = 0
        for stratum, target_tokens in STRATA.items():
            conversation, measured_tokens = _nearest_prefix(
                history, prefix_counts, target_tokens
            )
            if len(conversation) <= previous_count:
                raise AssertionError("E3 strata must be strict prefix extensions")
            previous_count = len(conversation)
            query_time = _time(scenario_index, len(conversation) + 1)
            if b1_prompt_tokens(conversation, spec["query"], query_time) != measured_tokens:
                raise AssertionError("temporal B1 incremental token count drifted")
            cases.append({
                "case_id": f"{design['scenario_family']}_{stratum}",
                "category": "long_context",
                "conversation": conversation,
                "query": spec["query"],
                "expected_memory_ids": [spec["target_id"]],
                "forbidden_memory_ids": [spec["old_id"]],
                "expected_answer": spec["answer"],
                "expected_version": "v2",
                "query_time": query_time,
                "memory_query_time": query_time,
                "difficulty": "hard" if stratum in {"long", "very_long"} else "medium",
                "metadata": {
                    "benchmark_version": BENCHMARK_VERSION,
                    "source_type": "project_owned",
                    "derivation_type": "orthogonal_controlled_long_context_scaling",
                    "split": "development",
                    "scenario_family": design["scenario_family"],
                    "scenario_index": scenario_index + 1,
                    "stratum": stratum,
                    "target_b1_prompt_tokens": target_tokens,
                    "b1_prompt_tokens": measured_tokens,
                    "history_memory_count": len(conversation),
                    "subject": spec["subject"],
                    "predicate": spec["predicate"],
                    "predicate_index": design["predicate_index"],
                    "predicate_surface": spec["predicate_surface"],
                    "query_expression_mode": spec["query_expression_mode"],
                    "old_answer": spec["old_answer"],
                    "target_memory_ordinal": spec["target_position"] + 1,
                    "target_position_index": design["target_position_index"],
                    "target_position_band": design["target_position_band"],
                    "distractor_kinds": spec["distractor_kinds"],
                    "query_template_index": design["query_template_index"],
                    "query_mode": "current",
                    "temporal_prompt": True,
                    "design_version": DESIGN_VERSION,
                    "method_candidate_version": MEMORY_RUNTIME_METHOD_VERSION,
                    "prompt_candidate_version": MEMORY_RUNTIME_PROMPT_VERSION,
                    "author": "amr-e3-v14-generator",
                    "review_status": "pending_human_review",
                    "generator_version": GENERATOR_VERSION,
                },
            })
    return cases


def write_development(output_dir: str | Path) -> dict:
    root = Path(output_dir)
    root.mkdir(parents=True, exist_ok=True)
    design_manifest = write_orthogonal_design(root)
    cases = build_development()
    development_path = root / "development.jsonl"
    with development_path.open("w", encoding="utf-8") as stream:
        for case in cases:
            stream.write(json.dumps(case, ensure_ascii=False, sort_keys=True) + "\n")
    review_path = root / "review_checklist.csv"
    with review_path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=REVIEW_FIELDS, lineterminator="\n")
        writer.writeheader()
        for case in cases:
            writer.writerow({
                "case_id": case["case_id"],
                "scenario_family": case["metadata"]["scenario_family"],
                "stratum": case["metadata"]["stratum"],
                "notes": "pending-technical-prereview",
            })
    observed_ranges = {}
    for stratum, target in STRATA.items():
        selected = [case for case in cases if case["metadata"]["stratum"] == stratum]
        values = [case["metadata"]["b1_prompt_tokens"] for case in selected]
        counts = [case["metadata"]["history_memory_count"] for case in selected]
        observed_ranges[stratum] = {
            "target_b1_prompt_tokens": target,
            "min_b1_prompt_tokens": min(values),
            "max_b1_prompt_tokens": max(values),
            "min_history_memory_count": min(counts),
            "max_history_memory_count": max(counts),
        }
    manifest = {
        "benchmark_version": BENCHMARK_VERSION,
        "generator_version": GENERATOR_VERSION,
        "design_version": DESIGN_VERSION,
        "design_sha256": design_manifest["design_sha256"],
        "seed": DEFAULT_SEED,
        "split": "development",
        "status": "pending_human_review",
        "scenario_count": SCENARIOS_PER_STRATUM,
        "strata": STRATA,
        "token_tolerance_ratio": TOKEN_TOLERANCE_RATIO,
        "case_count": len(cases),
        "case_order_policy": "scenario_major_stratum_interleaved",
        "development_file": development_path.name,
        "development_sha256": sha256_file(development_path),
        "review_file": review_path.name,
        "review_sha256": sha256_file(review_path),
        "observed_ranges": observed_ranges,
        "test_generated": False,
        "holdout_generated": False,
    }
    (root / "candidate_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate v1.4 E3 Development candidates.")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(__file__).resolve().parent / "data" / "v1.4-e3",
    )
    args = parser.parse_args()
    print(json.dumps(write_development(args.output_dir), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
