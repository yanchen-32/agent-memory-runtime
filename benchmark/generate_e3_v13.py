"""Deterministically generate the v1.3 E3 long-context Development candidate.

Each scenario has four prefix-nested history variants.  The question, target
fact, expected answer, and stale forbidden version are invariant; only later,
chronologically ordered distractor memories are appended.  Strata use the
exact B1 prompt construction and the project's deterministic token estimator.
"""

from __future__ import annotations

import argparse
from bisect import bisect_left
import csv
from datetime import datetime, timedelta, timezone
import json
from pathlib import Path
import random

from agent import FullHistoryAgent, estimate_tokens

from .artifacts import sha256_file


GENERATOR_VERSION = "v1.3-e3.1"
BENCHMARK_VERSION = "1.3-e3-candidate"
DEFAULT_SEED = 20260903
SCENARIOS_PER_STRATUM = 48
STRATA = {
    "short": 1_000,
    "medium": 4_000,
    "long": 16_000,
    "very_long": 32_000,
}
TOKEN_TOLERANCE_RATIO = 0.05
BASE_TIME = datetime(2024, 1, 1, 9, tzinfo=timezone(timedelta(hours=8)))
MAX_MEMORY_COUNT = 2_000
PREDICATE_SPECS = (
    ("数据库", "GaussDB", "SQLite"),
    ("部署平台", "鲲鹏节点", "x86节点"),
    ("架构", "微服务", "单体"),
    ("项目名称", "星桥计划", "旧港计划"),
    ("截止日期", "2027年", "2026年"),
    ("答辩日期", "2028年", "2027年"),
)
QUERY_TEMPLATES = (
    "{subject}当前的{predicate}是什么？",
    "请给出{subject}现行的{predicate}？",
    "最新记录中，{subject}采用的{predicate}是什么？",
    "{subject}目前登记的{predicate}是哪一项？",
    "请从长期记录中找出{subject}最新的{predicate}？",
    "大量历史记录里，{subject}现在使用什么{predicate}？",
    "按当前有效状态回答：{subject}的{predicate}是什么？",
    "忽略已经失效的旧值，{subject}的{predicate}现在是什么？",
)
TARGET_POSITION_SPECS = (
    ("front", 1),
    ("middle", 8),
    ("back", 17),
)
DISTRACTOR_KINDS = (
    "unrelated",
    "same_predicate_other_subject",
    "same_subject_other_predicate",
    "lexical_decoy",
)
REVIEW_FIELDS = (
    "case_id",
    "scenario_family",
    "stratum",
    "question_clear",
    "answer_correct",
    "evidence_id_correct",
    "forbidden_id_correct",
    "chronology_correct",
    "prefix_nesting_correct",
    "token_stratum_correct",
    "answer_leakage_free",
    "notes",
)


class _NoopClient:
    def generate(self, prompt: str) -> str:
        return ""


def b1_prompt_tokens(conversation: list[dict], query: str) -> int:
    """Measure the actual unconstrained B1 prompt with the canonical counter."""
    agent = FullHistoryAgent(_NoopClient())
    agent.answer(query, conversation=conversation)
    return agent.last_prompt_tokens


def _time(scenario_index: int, offset: int) -> str:
    # Scenario timelines are disjoint while each stays strictly chronological.
    point = BASE_TIME + timedelta(days=scenario_index * 60, minutes=offset)
    return point.isoformat()


def _memory(
    memory_id: str,
    content: str,
    scenario_index: int,
    offset: int,
    *,
    valid_to: str | None = None,
) -> dict:
    timestamp = _time(scenario_index, offset)
    record = {
        "memory_id": memory_id,
        "role": "user",
        "content": content,
        "created_at": timestamp,
        "valid_from": timestamp,
    }
    if valid_to is not None:
        record["valid_to"] = valid_to
    return record


def _scenario_answers(scenario_index: int, predicate_index: int) -> tuple[str, str]:
    predicate, current_prefix, old_prefix = PREDICATE_SPECS[predicate_index]
    if predicate in {"截止日期", "答辩日期"}:
        month = scenario_index % 12 + 1
        day = scenario_index % 27 + 1
        return (
            f"{current_prefix}{month}月{day}日",
            f"{old_prefix}{month}月{day}日",
        )
    return f"{current_prefix}-{scenario_index + 1:02d}", f"{old_prefix}-{scenario_index + 1:02d}"


def _distractor_content(
    scenario_index: int,
    offset: int,
    rng: random.Random,
    *,
    subject: str,
    predicate: str,
) -> tuple[str, str]:
    kind = DISTRACTOR_KINDS[(offset + scenario_index) % len(DISTRACTOR_KINDS)]
    subjects = ("巡检作业", "备份任务", "文档流水线", "日志归档", "依赖扫描", "构建批次")
    actions = (
        "完成校验并登记了摘要与责任队列",
        "核对状态后归档了清单与普通运行记录",
        "完成例行检查并更新了非关键审计备注",
        "保存了构建日志、依赖摘要与备份索引",
    )
    action = actions[rng.randrange(len(actions))]
    if kind == "unrelated":
        noise_subject = subjects[(offset + scenario_index) % len(subjects)]
        content = f"{noise_subject}{scenario_index + 1:02d}-{offset:04d}{action}。"
    elif kind == "same_predicate_other_subject":
        content = f"旁路单元{scenario_index + 1:02d}-{offset:04d}{predicate}为 辅助值-{offset:04d}。"
    elif kind == "same_subject_other_predicate":
        predicates = [item[0] for item in PREDICATE_SPECS if item[0] != predicate]
        other_predicate = predicates[(offset + scenario_index) % len(predicates)]
        content = f"{subject}{other_predicate}为 辅助配置-{offset:04d}。"
    else:
        content = (
            f"{subject}的审计索引{offset:04d}包含“{predicate}”字段名称，"
            "但该索引不保存字段取值。"
        )
    return content, kind


def _full_history(scenario_index: int, rng: random.Random) -> tuple[list[dict], dict]:
    predicate_index = scenario_index % len(PREDICATE_SPECS)
    predicate = PREDICATE_SPECS[predicate_index][0]
    subject = f"极光记忆单元{scenario_index + 1:02d}"
    answer, old_answer = _scenario_answers(scenario_index, predicate_index)
    family = f"v13_e3_dev_family_{scenario_index + 1:03d}"
    old_id = f"{family}_fact_v1"
    target_id = f"{family}_fact_v2"
    position_band, old_position = TARGET_POSITION_SPECS[
        scenario_index % len(TARGET_POSITION_SPECS)
    ]
    target_position = old_position + 1
    target_time = _time(scenario_index, target_position)
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
                f"{family}_noise_{offset:04d}",
                content,
                scenario_index,
                offset,
            )
        conversation.append(turn)
    return conversation, {
        "subject": subject,
        "predicate": predicate,
        "predicate_index": predicate_index,
        "answer": answer,
        "old_answer": old_answer,
        "family": family,
        "old_id": old_id,
        "target_id": target_id,
        "query": QUERY_TEMPLATES[scenario_index % len(QUERY_TEMPLATES)].format(
            subject=subject,
            predicate=predicate,
        ),
        "query_template_index": scenario_index % len(QUERY_TEMPLATES),
        "target_position": target_position,
        "target_position_band": position_band,
        "distractor_kinds": sorted(distractor_kinds),
    }


def _prefix_token_counts(conversation: list[dict], query: str) -> list[int]:
    total = b1_prompt_tokens([], query)
    counts = []
    for index, turn in enumerate(conversation, start=1):
        content = str(turn.get("content", "")).strip()
        timestamp = turn.get("valid_from") or turn.get("created_at")
        line = f"MEMORY[{index}] TIME[{timestamp}] {content}"
        total += estimate_tokens(line)
        counts.append(total)
    return counts


def _prefix_for_target(
    conversation: list[dict],
    query: str,
    prefix_counts: list[int],
    target: int,
) -> tuple[list[dict], int]:
    if b1_prompt_tokens(conversation, query) < target:
        raise ValueError(f"MAX_MEMORY_COUNT cannot reach {target} B1 prompt tokens")
    prefix_length = bisect_left(prefix_counts, target) + 1
    prefix = conversation[:prefix_length]
    measured = b1_prompt_tokens(prefix, query)
    if measured != prefix_counts[prefix_length - 1]:
        raise AssertionError("incremental B1 token measurement drifted from actual prompt")
    return prefix, measured


def build_development(seed: int = DEFAULT_SEED) -> list[dict]:
    """Build 48 matched scenarios at each of four B1 prompt-token strata."""
    cases: list[dict] = []
    for scenario_index in range(SCENARIOS_PER_STRATUM):
        rng = random.Random(seed + scenario_index * 10_007)
        history, spec = _full_history(scenario_index, rng)
        prefix_counts = _prefix_token_counts(history, spec["query"])
        previous_count = 0
        for stratum, target_tokens in STRATA.items():
            conversation, measured_tokens = _prefix_for_target(
                history, spec["query"], prefix_counts, target_tokens
            )
            if len(conversation) <= previous_count:
                raise AssertionError("E3 strata must add at least one memory")
            previous_count = len(conversation)
            query_time = _time(scenario_index, len(conversation) + 1)
            cases.append({
                "case_id": f"{spec['family']}_{stratum}",
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
                    "derivation_type": "controlled_long_context_scaling",
                    "split": "development",
                    "scenario_family": spec["family"],
                    "scenario_index": scenario_index + 1,
                    "stratum": stratum,
                    "target_b1_prompt_tokens": target_tokens,
                    "b1_prompt_tokens": measured_tokens,
                    "history_memory_count": len(conversation),
                    "subject": spec["subject"],
                    "predicate": spec["predicate"],
                    "predicate_index": spec["predicate_index"],
                    "old_answer": spec["old_answer"],
                    "target_memory_ordinal": spec["target_position"] + 1,
                    "target_position_band": spec["target_position_band"],
                    "distractor_kinds": spec["distractor_kinds"],
                    "query_template_index": spec["query_template_index"],
                    "query_mode": "current",
                    "author": "amr-e3-v13-generator",
                    "review_status": "pending_human_review",
                    "generator_version": GENERATOR_VERSION,
                },
            })
    # Scenario-major order interleaves all four scales every four cases. This
    # limits confounding between remote-service load drift and context stratum.
    return cases


def write_development(output_dir: str | Path, seed: int = DEFAULT_SEED) -> dict:
    root = Path(output_dir)
    root.mkdir(parents=True, exist_ok=True)
    cases = build_development(seed)
    development_path = root / "development.jsonl"
    with development_path.open("w", encoding="utf-8") as stream:
        for case in cases:
            stream.write(json.dumps(case, ensure_ascii=False, sort_keys=True) + "\n")

    checklist_path = root / "review_checklist.csv"
    with checklist_path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=REVIEW_FIELDS, lineterminator="\n")
        writer.writeheader()
        for case in cases:
            writer.writerow({
                "case_id": case["case_id"],
                "scenario_family": case["metadata"]["scenario_family"],
                "stratum": case["metadata"]["stratum"],
                "notes": "pending-technical-prereview",
            })

    token_ranges = {}
    for stratum, target in STRATA.items():
        values = [
            case["metadata"]["b1_prompt_tokens"]
            for case in cases
            if case["metadata"]["stratum"] == stratum
        ]
        counts = [
            case["metadata"]["history_memory_count"]
            for case in cases
            if case["metadata"]["stratum"] == stratum
        ]
        token_ranges[stratum] = {
            "target_b1_prompt_tokens": target,
            "min_b1_prompt_tokens": min(values),
            "max_b1_prompt_tokens": max(values),
            "min_history_memory_count": min(counts),
            "max_history_memory_count": max(counts),
        }
    manifest = {
        "benchmark_version": BENCHMARK_VERSION,
        "generator_version": GENERATOR_VERSION,
        "seed": seed,
        "split": "development",
        "status": "pending_human_review",
        "scenario_count": SCENARIOS_PER_STRATUM,
        "strata": STRATA,
        "token_tolerance_ratio": TOKEN_TOLERANCE_RATIO,
        "case_count": len(cases),
        "case_order_policy": "scenario_major_stratum_interleaved",
        "development_file": development_path.name,
        "development_sha256": sha256_file(development_path),
        "review_file": checklist_path.name,
        "review_sha256": sha256_file(checklist_path),
        "observed_ranges": token_ranges,
        "holdout_generated": False,
    }
    (root / "candidate_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate v1.3 E3 Development candidates.")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(__file__).resolve().parent / "data" / "v1.3-e3",
    )
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    args = parser.parse_args()
    print(json.dumps(write_development(args.output_dir, args.seed), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
