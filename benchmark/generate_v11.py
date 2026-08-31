"""Deterministically build the project-owned AMR-CN Benchmark v1.1 candidates.

The generator intentionally uses independent scenario primitives.  It does not
read, translate, or paraphrase upstream benchmark conversations or labels.
"""

from __future__ import annotations

import argparse
import copy
import csv
from datetime import datetime, timedelta, timezone
import json
from pathlib import Path
import random

from .artifacts import sha256_file


GENERATOR_VERSION = "v1.1.1"
DEFAULT_SEED = 20260831
SPLIT = "development"
TIMEZONE = timezone(timedelta(hours=8))
BASE_TIME = datetime(2026, 1, 1, 9, tzinfo=TIMEZONE)
NON_GOVERNANCE_CATEGORIES = (
    "fact_recall",
    "semantic_recall",
    "multi_hop",
    "abstention",
    "noise",
    "budget",
    "forgetting",
    "long_context",
)
SPLIT_SPECS = {
    "development": {"seed_offset": 0, "governance_pairs": 48, "per_category": 24},
    "test": {"seed_offset": 101, "governance_pairs": 16, "per_category": 8},
    "holdout": {"seed_offset": 202, "governance_pairs": 16, "per_category": 8},
}
SUBJECT_REWRITES = {
    "test": {
        "星图研发组": "澄明数据中心",
        "北辰项目": "雾港方案",
        "天枢助手": "织云知识库",
        "远航部署单元": "松风部署站",
        "云舟服务": "澜海服务",
        "启明检索器": "清河检索器",
        "松涛检索器": "明光检索器",
        "临川调试服务": "流云调试服务",
        "星海发布单元": "云岚发布单元",
    },
    "holdout": {
        "星图研发组": "玄岳供应站",
        "北辰项目": "鹭岛项目",
        "天枢助手": "寰宇档案员",
        "远航部署单元": "北斗节点",
        "云舟服务": "玉衡服务",
        "启明检索器": "天问检索器",
        "松涛检索器": "玄光检索器",
        "临川调试服务": "霜叶调试服务",
        "星海发布单元": "瀚海发布单元",
    },
}


def _time(day: int) -> str:
    return (BASE_TIME + timedelta(days=day)).isoformat()


def _memory(memory_id: str, content: str, day: int, *, valid_to: str | None = None) -> dict:
    record = {
        "memory_id": memory_id,
        "role": "user",
        "content": content,
        "created_at": _time(day),
        "valid_from": _time(day),
    }
    if valid_to is not None:
        record["valid_to"] = valid_to
    return record


def _metadata(family: str, **extra: object) -> dict:
    return {
        "benchmark_version": "1.1-candidate",
        "source_type": "project_owned",
        "derivation_type": "independent_authoring_from_capability_spec",
        "split": SPLIT,
        "scenario_family": family,
        "author": "amr-v11-generator",
        "review_status": "pending_human_review",
        "generator_version": GENERATOR_VERSION,
        **extra,
    }


def _case(
    *,
    case_id: str,
    category: str,
    conversation: list[dict],
    query: str,
    expected_memory_ids: list[str],
    expected_answer: str,
    day: int,
    family: str,
    difficulty: str,
    subject: str | None = None,
    forbidden_memory_ids: list[str] | None = None,
    expected_version: str = "v1",
    token_budget: int | None = None,
    forget_memory_ids: list[str] | None = None,
    memory_metadata: dict | None = None,
    answer_aliases: list[str] | None = None,
    metadata: dict | None = None,
) -> dict:
    payload = {
        "case_id": case_id,
        "category": category,
        "conversation": conversation,
        "query": query,
        "expected_memory_ids": expected_memory_ids,
        "forbidden_memory_ids": forbidden_memory_ids or [],
        "expected_answer": expected_answer,
        "expected_version": expected_version,
        "query_time": _time(day),
        "memory_query_time": _time(day),
        "difficulty": difficulty,
        "metadata": metadata or _metadata(family, **({"subject": subject} if subject else {})),
    }
    if token_budget is not None:
        payload["token_budget"] = token_budget
    if forget_memory_ids:
        payload["forget_memory_ids"] = forget_memory_ids
    if memory_metadata:
        payload["memory_metadata"] = memory_metadata
    if answer_aliases:
        payload["answer_aliases"] = answer_aliases
    return payload


def _governance_pair(index: int, rng: random.Random) -> list[dict]:
    predicate_specs = (
        ("数据库", ("SQLite", "PostgreSQL", "openGauss", "TiDB", "OceanBase", "MySQL")),
        ("部署平台", ("x86_64", "鲲鹏 ARM64", "飞腾 ARM64", "RISC-V", "AArch64", "LoongArch")),
        ("架构", ("单体架构", "微服务架构", "事件驱动架构", "分层架构", "插件化架构", "无服务器架构")),
    )
    predicate, values = predicate_specs[index % len(predicate_specs)]
    old_value, new_value = rng.sample(values, 2)
    subject = f"星图研发组{index + 1:02d}"
    family = f"v11_dev_governance_{index + 1:03d}"
    pair_id = f"v11_dev_pair_{index + 1:03d}"
    old_id = f"v11_dev_gov_{index + 1:03d}_v1"
    new_id = f"v11_dev_gov_{index + 1:03d}_v2"
    old_day = 2 * index
    new_day = old_day + 30
    query_day = new_day + 20
    old_time, new_time = _time(old_day), _time(new_day)
    conversation = [
        _memory(old_id, f"{subject}{predicate}为 {old_value}。", old_day, valid_to=new_time),
        _memory(new_id, f"{subject}{predicate}改为 {new_value}。", new_day),
    ]
    common = {
        "pair_id": pair_id,
        "subject": subject,
        "predicate": predicate,
        "old_memory_id": old_id,
        "new_memory_id": new_id,
    }
    current_category = "update" if index % 2 == 0 else "conflict"
    current_queries = (
        f"{subject}当前的{predicate}是什么？",
        f"请给出{subject}现行的{predicate}？",
        f"最新记录中，{subject}采用什么{predicate}？",
        f"{subject}目前登记的{predicate}是哪一项？",
    )
    current = _case(
        case_id=f"v11_dev_{current_category}_{index + 1:03d}_current",
        category=current_category,
        conversation=conversation,
        query=current_queries[index % len(current_queries)],
        expected_memory_ids=[new_id],
        forbidden_memory_ids=[old_id],
        expected_answer=new_value,
        expected_version="v2",
        day=query_day,
        family=family,
        difficulty="medium",
        metadata=_metadata(family, query_mode="current", **common),
    )
    historical_day = old_day + 10
    history_date = BASE_TIME + timedelta(days=historical_day)
    historical_queries = (
        f"2026年{history_date.month}月{history_date.day}日时，{subject}的{predicate}是什么？",
        f"请按2026年{history_date.month}月{history_date.day}日的记录回答：{subject}当时采用什么{predicate}？",
        f"回看2026年{history_date.month}月{history_date.day}日，{subject}登记的{predicate}是哪一项？",
        f"在2026年{history_date.month}月{history_date.day}日这个时间点，{subject}的{predicate}为何？",
    )
    historical = _case(
        case_id=f"v11_dev_temporal_{index + 1:03d}_history",
        category="temporal",
        conversation=conversation,
        query=historical_queries[index % len(historical_queries)],
        expected_memory_ids=[old_id],
        forbidden_memory_ids=[new_id],
        expected_answer=old_value,
        expected_version="v1",
        day=historical_day,
        family=family,
        difficulty="hard",
        metadata=_metadata(family, query_mode="historical", **common),
    )
    return [current, historical]


def _fact_case(index: int) -> dict:
    subject = f"北辰项目{index + 1:02d}"
    answer = f"蓝海-{index + 1:02d}"
    target = f"v11_dev_fact_{index + 1:03d}_target"
    family = f"v11_dev_fact_{index + 1:03d}"
    queries = (
        f"{subject}的项目名称是什么？",
        f"请说出{subject}登记的项目名称？",
        f"{subject}以什么名称立项？",
        f"记录中{subject}叫什么项目？",
        f"{subject}正式项目名是什么？",
        f"请给出{subject}的项目名称？",
    )
    return _case(
        case_id=f"v11_dev_fact_{index + 1:03d}",
        category="fact_recall",
        conversation=[
            _memory(target, f"{subject}项目名称为 {answer}。", index),
            _memory(f"v11_dev_fact_{index + 1:03d}_noise", f"{subject}本周完成了例行文档整理。", index + 1),
        ],
        query=queries[index % len(queries)],
        expected_memory_ids=[target],
        expected_answer=answer,
        day=index + 60,
        family=family,
        difficulty="easy",
        subject=subject,
    )


def _semantic_case(index: int) -> dict:
    subject = f"天枢助手{index + 1:02d}"
    values = ("语义记忆", "情景记忆", "程序记忆", "工作记忆")
    answer = values[index % len(values)]
    target = f"v11_dev_semantic_{index + 1:03d}_target"
    family = f"v11_dev_semantic_{index + 1:03d}"
    queries = (
        f"{subject}用哪类记忆保存长期稳定知识？",
        f"哪一种记忆承载{subject}的长期稳定知识？",
        f"请指出{subject}保存稳定知识所用的记忆类型？",
        f"{subject}把长期稳定知识放在哪类记忆中？",
        f"记录显示{subject}依靠什么记忆保存稳定知识？",
        f"{subject}的长期知识存储属于哪种记忆？",
    )
    return _case(
        case_id=f"v11_dev_semantic_{index + 1:03d}",
        category="semantic_recall",
        conversation=[
            _memory(target, f"{subject}用{answer}保存长期稳定知识。", index + 80),
            _memory(f"v11_dev_semantic_{index + 1:03d}_noise", f"{subject}的会话缓冲只保存当前任务状态。", index + 81),
        ],
        query=queries[index % len(queries)],
        expected_memory_ids=[target],
        expected_answer=answer,
        day=index + 140,
        family=family,
        difficulty="easy",
        subject=subject,
    )


def _multihop_case(index: int) -> dict:
    subject = f"远航部署单元{index + 1:02d}"
    os_name = ("openEuler", "Ubuntu", "Debian", "Anolis OS")[index % 4]
    architecture = ("ARM64", "x86_64", "LoongArch", "RISC-V")[index % 4]
    family = f"v11_dev_multihop_{index + 1:03d}"
    os_id = f"v11_dev_multihop_{index + 1:03d}_os"
    arch_id = f"v11_dev_multihop_{index + 1:03d}_arch"
    queries = (
        f"{subject}使用什么操作系统和架构？",
        f"请同时给出{subject}的操作系统与处理器架构？",
        f"{subject}运行在哪种系统和架构组合上？",
        f"记录中的{subject}采用哪套操作系统及架构？",
        f"{subject}的系统名称和架构分别是什么？",
        f"部署{subject}需要匹配什么操作系统与架构？",
    )
    return _case(
        case_id=f"v11_dev_multihop_{index + 1:03d}",
        category="multi_hop",
        conversation=[
            _memory(os_id, f"{subject}的操作系统是 {os_name}。", index + 110),
            _memory(arch_id, f"{subject}的架构为 {architecture}。", index + 111),
            _memory(f"v11_dev_multihop_{index + 1:03d}_noise", f"{subject}的部署说明存放在 docs 目录。", index + 112),
        ],
        query=queries[index % len(queries)],
        expected_memory_ids=[os_id, arch_id],
        expected_answer=f"{os_name} {architecture}",
        answer_aliases=[f"{os_name}和{architecture}"],
        day=index + 180,
        family=family,
        difficulty="hard",
        subject=subject,
    )


def _abstention_case(index: int) -> dict:
    subject = f"云舟服务{index + 1:02d}"
    family = f"v11_dev_abstention_{index + 1:03d}"
    queries = (
        f"{subject}负责人的手机号是多少？",
        f"请提供{subject}负责人的联系电话？",
        f"记录里{subject}负责人使用什么手机号码？",
        f"如何电话联系{subject}负责人？",
        f"{subject}负责人的移动电话是什么？",
        f"请查出{subject}负责人的号码？",
    )
    return _case(
        case_id=f"v11_dev_abstention_{index + 1:03d}",
        category="abstention",
        conversation=[_memory(f"v11_dev_abs_{index + 1:03d}", f"{subject}使用 HTTPS 接口。", index + 140)],
        query=queries[index % len(queries)],
        expected_memory_ids=[],
        expected_answer="UNKNOWN",
        expected_version="",
        day=index + 200,
        family=family,
        difficulty="medium",
        subject=subject,
    )


def _noise_case(index: int) -> dict:
    subject = f"启明检索器{index + 1:02d}"
    answer = str(256 * ((index % 4) + 1))
    target = f"v11_dev_noise_{index + 1:03d}_target"
    family = f"v11_dev_noise_{index + 1:03d}"
    conversation = [
        _memory(f"v11_dev_noise_{index + 1:03d}_n1", f"{subject}的会议纪要使用 Markdown 格式。", index + 160),
        _memory(target, f"{subject}的向量维度确定为 {answer}。", index + 161),
        _memory(f"v11_dev_noise_{index + 1:03d}_n2", f"{subject}的测试日志按日期归档。", index + 162),
        _memory(f"v11_dev_noise_{index + 1:03d}_n3", f"{subject}的演示页面采用深色主题。", index + 163),
    ]
    queries = (
        f"{subject}的向量维度是多少？",
        f"请给出{subject}配置的向量维度？",
        f"{subject}使用多少维向量？",
        f"记录中{subject}的嵌入维度为何？",
        f"{subject}向量配置包含多少个维度？",
        f"请查找{subject}的向量维数？",
    )
    return _case(
        case_id=f"v11_dev_noise_{index + 1:03d}",
        category="noise",
        conversation=conversation,
        query=queries[index % len(queries)],
        expected_memory_ids=[target],
        expected_answer=answer,
        day=index + 220,
        family=family,
        difficulty="medium",
        subject=subject,
    )


def _budget_case(index: int) -> dict:
    subject = f"松涛检索器{index + 1:02d}"
    answer = str((index % 5) + 3)
    target = f"v11_dev_budget_{index + 1:03d}_target"
    family = f"v11_dev_budget_{index + 1:03d}"
    queries = (
        f"{subject}的检索 Top-K 是多少？",
        f"请给出{subject}设定的 Top-K？",
        f"{subject}每次取回多少条候选记忆？",
        f"记录中{subject}的 Top-K 参数为何？",
        f"{subject}固定检索几条结果？",
        f"请查找{subject}的检索数量上限？",
    )
    return _case(
        case_id=f"v11_dev_budget_{index + 1:03d}",
        category="budget",
        conversation=[
            _memory(target, f"{subject}检索 Top-K 为 {answer}。", index + 180),
            _memory(f"v11_dev_budget_{index + 1:03d}_n1", "代码仓库使用 Git。", index + 181),
            _memory(f"v11_dev_budget_{index + 1:03d}_n2", "报告包含运行说明。", index + 182),
        ],
        query=queries[index % len(queries)],
        expected_memory_ids=[target],
        expected_answer=answer,
        day=index + 240,
        family=family,
        difficulty="medium",
        subject=subject,
        token_budget=80,
    )


def _forgetting_case(index: int) -> dict:
    subject = f"临川调试服务{index + 1:02d}"
    target = f"v11_dev_forget_{index + 1:03d}_target"
    family = f"v11_dev_forgetting_{index + 1:03d}"
    created = _time(index)
    queries = (
        f"{subject}临时调试端口是多少？",
        f"请查找{subject}已经清除的临时端口？",
        f"{subject}过去的调试端口是什么？",
        f"记录能否提供{subject}的临时调试端口？",
        f"{subject}曾使用哪个临时端口？",
        f"请回答{subject}临时开放的端口号？",
    )
    return _case(
        case_id=f"v11_dev_forgetting_{index + 1:03d}",
        category="forgetting",
        conversation=[
            _memory(target, f"{subject}临时调试端口是 {8000 + index}。", index),
            _memory(f"v11_dev_forget_{index + 1:03d}_keep", f"{subject}使用 HTTPS 接口。", index + 200),
        ],
        query=queries[index % len(queries)],
        expected_memory_ids=[],
        forbidden_memory_ids=[target],
        expected_answer="UNKNOWN",
        expected_version="",
        day=index + 260,
        family=family,
        difficulty="medium",
        subject=subject,
        forget_memory_ids=[target],
        memory_metadata={target: {"importance": 0.0, "utility": 0.0, "created_at": created}},
    )


def _long_context_case(index: int, rng: random.Random) -> dict:
    subject = f"星海发布单元{index + 1:02d}"
    answer = f"苍穹{index + 1:02d}"
    family = f"v11_dev_long_context_{index + 1:03d}"
    target = f"v11_dev_long_{index + 1:03d}_target"
    target_position = rng.randrange(64)
    conversation = []
    for offset in range(64):
        if offset == target_position:
            conversation.append(_memory(target, f"{subject}的发布代号是 {answer}。", index + 210 + offset))
        else:
            conversation.append(
                _memory(
                    f"v11_dev_long_{index + 1:03d}_n{offset:02d}",
                    f"{subject}第{offset + 1}次例行记录完成了构建、日志与文档检查。",
                    index + 210 + offset,
                )
            )
    queries = (
        f"{subject}的发布代号是什么？",
        f"请从长期记录中找出{subject}的发布代号？",
        f"{subject}以哪个代号发布？",
        f"大量记录中登记的{subject}发布代号是哪一个？",
        f"请回答{subject}正式使用的发布代号？",
        f"{subject}发布时采用什么代号？",
    )
    return _case(
        case_id=f"v11_dev_long_context_{index + 1:03d}",
        category="long_context",
        conversation=conversation,
        query=queries[index % len(queries)],
        expected_memory_ids=[target],
        expected_answer=answer,
        day=index + 300,
        family=family,
        difficulty="hard",
        subject=subject,
    )


def build_development(seed: int = DEFAULT_SEED) -> list[dict]:
    """Return the fixed 288-case Development candidate set for a given seed."""
    rng = random.Random(seed)
    cases = [case for index in range(48) for case in _governance_pair(index, rng)]
    factories = (
        _fact_case,
        _semantic_case,
        _multihop_case,
        _abstention_case,
        _noise_case,
        _budget_case,
        _forgetting_case,
    )
    for factory in factories:
        cases.extend(factory(index) for index in range(24))
    cases.extend(_long_context_case(index, rng) for index in range(24))
    return cases


def _replace_ids(value: object, split: str) -> object:
    if isinstance(value, str):
        return value.replace("v11_dev", f"v11_{split}")
    if isinstance(value, list):
        return [_replace_ids(item, split) for item in value]
    if isinstance(value, dict):
        return {str(_replace_ids(key, split)): _replace_ids(item, split) for key, item in value.items()}
    return value


def _rewrite_text(value: str, split: str) -> str:
    if split == "development":
        return value
    for old, new in SUBJECT_REWRITES[split].items():
        value = value.replace(old, new)
    if split == "test":
        replacements = (
            ("数据库改为 ", "数据库后来是 "),
            ("数据库为 ", "数据库是 "),
            ("部署平台改为 ", "部署平台改成 "),
            ("部署平台为 ", "部署平台是 "),
            ("架构改为 ", "架构改成 "),
            ("架构为 ", "架构是 "),
            ("项目名称为 ", "项目名称是 "),
            ("操作系统是 ", "操作系统为 "),
            ("向量维度确定为 ", "向量维度设为 "),
            ("检索 Top-K 为 ", "检索 Top-K 是 "),
            ("临时调试端口是 ", "临时调试端口为 "),
            ("发布代号是 ", "发布代号为 "),
            ("本周完成了", "测试记录完成了"),
            ("工作记忆只保存", "评审说明：工作记忆只保存"),
            ("部署说明存放", "运行指引存放"),
            ("会议纪要使用", "归档记录：会议纪要使用"),
            ("测试日志按日期", "监测日志按日期"),
            ("演示页面采用", "控制台采用"),
            ("代码仓库使用", "代码资产使用"),
            ("报告包含", "交付报告包含"),
        )
    else:
        replacements = (
            ("数据库改为 ", "数据库改成 "),
            ("数据库为 ", "数据库确定为 "),
            ("部署平台改为 ", "部署平台改成 "),
            ("部署平台为 ", "部署平台确定为 "),
            ("架构改为 ", "架构改成 "),
            ("架构为 ", "架构确定为 "),
            ("项目名称为 ", "项目名称定为 "),
            ("操作系统是 ", "操作系统为 "),
            ("向量维度确定为 ", "向量维度配置为 "),
            ("检索 Top-K 为 ", "检索 Top-K 固定为 "),
            ("临时调试端口是 ", "临时调试端口配置为 "),
            ("发布代号是 ", "发布代号定为 "),
            ("本周完成了", "巡检记录完成了"),
            ("工作记忆只保存", "运行规范：工作记忆只保存"),
            ("部署说明存放", "现场手册存放"),
            ("会议纪要使用", "质检记录：会议纪要使用"),
            ("测试日志按日期", "审计日志按日期"),
            ("演示页面采用", "操作台采用"),
            ("代码仓库使用", "版本库使用"),
            ("报告包含", "验收报告包含"),
        )
    for old, new in replacements:
        value = value.replace(old, new)
    if split == "test":
        value = value.replace("当前的", "最新登记的")
        value = value.replace("是什么？", "请给出答案？")
        if value.startswith("2026年"):
            value = "请依据历史记录回答：" + value
    else:
        value = value.replace("当前的", "现行的")
        value = value.replace("是什么？", "请按记录作答？")
        if value.startswith("2026年"):
            value += "请按当时状态作答。"
    return value


def _adapt_case_for_split(case: dict, split: str) -> dict:
    adapted = copy.deepcopy(_replace_ids(case, split))
    adapted["metadata"]["split"] = split
    adapted["metadata"]["scenario_family"] = adapted["metadata"]["scenario_family"].replace(
        "v11_dev", f"v11_{split}"
    )
    if adapted["metadata"].get("pair_id"):
        adapted["metadata"]["pair_id"] = adapted["metadata"]["pair_id"].replace(
            "v11_dev", f"v11_{split}"
        )
    for old, new in SUBJECT_REWRITES[split].items():
        if adapted["metadata"].get("subject"):
            adapted["metadata"]["subject"] = adapted["metadata"]["subject"].replace(old, new)
    adapted["query"] = _rewrite_text(adapted["query"], split)
    for turn in adapted["conversation"]:
        turn["content"] = _rewrite_text(turn["content"], split)
    return adapted


def build_split(split: str, seed: int = DEFAULT_SEED) -> list[dict]:
    """Build one v1.1 split with split-disjoint entities and text templates."""
    if split not in SPLIT_SPECS:
        raise ValueError(f"unsupported split: {split}")
    if split == "development":
        return build_development(seed)
    spec = SPLIT_SPECS[split]
    source = build_development(seed + int(spec["seed_offset"]))
    selected = source[: 2 * int(spec["governance_pairs"])]
    for category in NON_GOVERNANCE_CATEGORIES:
        category_cases = [case for case in source if case["category"] == category]
        selected.extend(category_cases[: int(spec["per_category"])])
    return [_adapt_case_for_split(case, split) for case in selected]


def build_all_splits(seed: int = DEFAULT_SEED) -> dict[str, list[dict]]:
    return {split: build_split(split, seed) for split in SPLIT_SPECS}


REVIEW_FIELDS = (
    "case_id",
    "question_clear",
    "answer_correct",
    "evidence_ids_correct",
    "forbidden_ids_correct",
    "timestamps_correct",
    "aliases_checked",
    "notes",
)


def write_development(output_dir: str | Path, seed: int = DEFAULT_SEED) -> dict:
    """Write the deterministic Development candidate and blank review checklist."""
    root = Path(output_dir)
    root.mkdir(parents=True, exist_ok=True)
    cases = build_development(seed)
    development_path = root / "development.jsonl"
    with development_path.open("w", encoding="utf-8") as stream:
        for case in cases:
            stream.write(json.dumps(case, ensure_ascii=False, sort_keys=True) + "\n")
    checklist_path = root / "review_checklist.csv"
    with checklist_path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=REVIEW_FIELDS)
        writer.writeheader()
        for case in cases:
            writer.writerow({"case_id": case["case_id"], "notes": "pending-technical-prereview"})
    manifest = {
        "benchmark_version": "1.1-candidate",
        "generator_version": GENERATOR_VERSION,
        "seed": seed,
        "split": SPLIT,
        "status": "pending_human_review",
        "case_count": len(cases),
        "development_file": development_path.name,
        "development_sha256": sha256_file(development_path),
        "review_file": checklist_path.name,
        "review_sha256": sha256_file(checklist_path),
    }
    (root / "candidate_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return manifest


def write_all_splits(output_dir: str | Path, seed: int = DEFAULT_SEED) -> dict:
    """Write all v1.1 candidate splits and one blank global review checklist."""
    root = Path(output_dir)
    root.mkdir(parents=True, exist_ok=True)
    splits = build_all_splits(seed)
    split_manifest = {}
    all_cases = []
    for split, cases in splits.items():
        path = root / f"{split}.jsonl"
        with path.open("w", encoding="utf-8") as stream:
            for case in cases:
                stream.write(json.dumps(case, ensure_ascii=False, sort_keys=True) + "\n")
        split_manifest[split] = {"file": path.name, "case_count": len(cases), "sha256": sha256_file(path)}
        all_cases.extend(cases)
    checklist_path = root / "review_checklist.csv"
    with checklist_path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=REVIEW_FIELDS)
        writer.writeheader()
        for case in all_cases:
            writer.writerow({"case_id": case["case_id"], "notes": "pending-technical-prereview"})
    manifest = {
        "benchmark_version": "1.1-candidate",
        "generator_version": GENERATOR_VERSION,
        "seed": seed,
        "status": "pending_human_review",
        "total_case_count": len(all_cases),
        "splits": split_manifest,
        "review_file": checklist_path.name,
        "review_sha256": sha256_file(checklist_path),
    }
    (root / "candidate_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate AMR-CN Benchmark v1.1 Development candidates.")
    parser.add_argument("--output-dir", type=Path, default=Path(__file__).resolve().parent / "data" / "v1.1")
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--development-only", action="store_true")
    args = parser.parse_args()
    writer = write_development if args.development_only else write_all_splits
    print(json.dumps(writer(args.output_dir, args.seed), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
