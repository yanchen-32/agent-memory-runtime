from __future__ import annotations

from collections import defaultdict
from datetime import datetime, timezone
from statistics import mean, stdev
import time
from typing import Callable, Iterable

from agent import (
    FullHistoryAgent,
    HybridMemoryAgent,
    MemoryRuntimeAgent,
    NoMemoryAgent,
    VectorMemoryAgent,
)
from memory import (
    AdaptiveConsolidationPolicy,
    HashEmbeddingModel,
    MemoryRuntimeV1,
    MemoryType,
    VectorMemoryStore,
)

from .loader import BenchmarkCase
from .metrics import answer_metrics, estimate_tokens, retrieval_metrics


AGENT_NAMES = ("B0", "B1", "B2", "B3", "Ours")
RETRIEVAL_KEYS = (
    "recall@1", "precision@1",
    "recall@5", "precision@5",
    "recall@10", "precision@10",
    "mrr",
)


def _parse_time(value: datetime | str | None) -> datetime | None:
    if value is None:
        return None
    if isinstance(value, datetime):
        parsed = value
    else:
        try:
            parsed = datetime.fromisoformat(value)
        except ValueError:
            return None
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed


def _case_content_map(case: BenchmarkCase) -> dict[str, str]:
    return {
        str(turn.get("content", "")): str(turn.get("memory_id", ""))
        for turn in case.conversation
        if turn.get("content") and turn.get("memory_id")
    }


def _apply_forgetting_setup(runtime: MemoryRuntimeV1, case: BenchmarkCase) -> list[str]:
    if not case.forget_memory_ids:
        return []

    content_to_id = {
        str(turn.get("content", "")): str(turn.get("memory_id", ""))
        for turn in case.conversation
    }
    target_ids = set(case.forget_memory_ids)
    attributes = case.memory_metadata or {}
    query_time = _parse_time(case.query_time)

    for record in runtime.store.list_all():
        benchmark_id = content_to_id.get(record.content)
        if benchmark_id not in target_ids:
            continue
        config = attributes.get(benchmark_id, {})
        record.importance = float(config.get("importance", 0.0))
        record.utility = float(config.get("utility", 0.0))
        created_at = _parse_time(config.get("created_at")) if config else None
        if created_at is not None:
            record.created_at = created_at
        elif query_time is not None:
            record.created_at = query_time
        runtime.store.update(record)

    return runtime.forgetting.run(user_id="benchmark", now=query_time)


def _map_runtime_ids(agent: MemoryRuntimeAgent, case: BenchmarkCase) -> list[str]:
    content_to_id = _case_content_map(case)
    return [
        content_to_id[content]
        for content in agent.last_retrieved_contents
        if content in content_to_id
    ]


def _empty_retrieval_metrics() -> dict[str, float | None]:
    return {key: None for key in RETRIEVAL_KEYS}


def _invoke(
    agent_name: str,
    agent,
    case: BenchmarkCase,
    token_budget: int | None,
    trace_id: str | None = None,
):
    if agent_name == "B0":
        return agent.answer(case.query)
    if agent_name == "B1":
        return agent.answer(
            case.query,
            conversation=case.conversation,
            token_budget=token_budget,
        )
    if agent_name == "B2":
        return agent.answer(case.query, token_budget=token_budget)
    if agent_name == "B3":
        return agent.answer(case.query, token_budget=token_budget)
    return agent.answer(
        case.query,
        token_budget=token_budget,
        query_time=case.memory_query_time,
        trace_id=trace_id,
        query_id=case.case_id,
    )


def _snapshot(agent_name: str, agent, case: BenchmarkCase) -> dict:
    if agent_name == "Ours":
        ranked_ids = _map_runtime_ids(agent, case)
    else:
        ranked_ids = list(getattr(agent, "last_retrieved_ids", []))
    usage = dict(getattr(getattr(agent, "client", None), "last_usage", {}) or {})
    return {
        "prompt": getattr(agent, "last_prompt", ""),
        "context": getattr(agent, "last_context", ""),
        "ranked_ids": ranked_ids,
        "memory_latency_ms": float(getattr(agent, "last_memory_latency_ms", 0.0)),
        "context_latency_ms": float(getattr(agent, "last_context_latency_ms", 0.0)),
        "llm_latency_ms": float(getattr(agent, "last_llm_latency_ms", 0.0)),
        "api_usage": usage,
    }


def _run_once(
    agent_name: str,
    agent,
    case: BenchmarkCase,
    token_budget: int | None,
    trace_id: str | None = None,
) -> tuple[str, float, dict]:
    started = time.perf_counter()
    prediction = _invoke(agent_name, agent, case, token_budget, trace_id=trace_id)
    latency_ms = (time.perf_counter() - started) * 1000
    snapshot = _snapshot(agent_name, agent, case)
    measured = (
        snapshot["memory_latency_ms"]
        + snapshot["context_latency_ms"]
        + snapshot["llm_latency_ms"]
    )
    snapshot["post_latency_ms"] = max(0.0, latency_ms - measured)
    snapshot["prediction"] = prediction
    return prediction, latency_ms, snapshot


def run_case(
    agent_name: str,
    case: BenchmarkCase,
    client_factory: Callable[[], object],
    embedder_factory: Callable[[], object],
    top_k: int = 5,
    token_budget: int | None = None,
    repeat: int = 1,
    consolidation_strategy: str = "fixed",
    consolidation_policy: AdaptiveConsolidationPolicy | None = None,
    trace_enabled: bool = False,
) -> dict:
    if agent_name not in AGENT_NAMES:
        raise ValueError(f"unknown agent: {agent_name}")

    client = client_factory()
    budget = case.token_budget if case.token_budget is not None else token_budget
    setup_started = time.perf_counter()
    archived_ids: list[str] = []
    case_trace_id: str | None = None

    if agent_name == "B0":
        agent = NoMemoryAgent(client)
        retrieval_supported = False
    elif agent_name == "B1":
        agent = FullHistoryAgent(client)
        retrieval_supported = False
    elif agent_name == "B2":
        store = VectorMemoryStore(embedder_factory())
        for turn in case.conversation:
            store.add(
                str(turn.get("content", "")),
                metadata={"role": turn.get("role", "user")},
                memory_id=turn.get("memory_id"),
            )
        agent = VectorMemoryAgent(client, store, top_k=top_k)
        retrieval_supported = True
    elif agent_name == "B3":
        agent = HybridMemoryAgent(
            client,
            embedder=embedder_factory(),
            top_k=top_k,
        )
        agent.ingest(case.conversation)
        retrieval_supported = True
    else:
        runtime = MemoryRuntimeV1(
            embedder=embedder_factory(),
            consolidation_policy=consolidation_policy,
            trace_enabled=trace_enabled,
        )
        if trace_enabled:
            case_trace_id = runtime.trace_recorder.new_trace_id()
        agent = MemoryRuntimeAgent(client, runtime, top_k=top_k)
        agent.ingest(
            case.conversation,
            user_id="benchmark",
            session_id=f"{case.case_id}-repeat-{repeat}",
            memory_type_override=(
                MemoryType.EPISODIC if case.category == "consolidation" else None
            ),
            preserve_duplicates=case.category == "consolidation",
        )
        consolidation_report = None
        consolidation_latency_ms = None
        if case.category == "consolidation":
            consolidation_started = time.perf_counter()
            consolidation_report = runtime.consolidate(
                user_id="benchmark",
                strategy=consolidation_strategy,
                trace_id=case_trace_id,
            )
            consolidation_latency_ms = (time.perf_counter() - consolidation_started) * 1000
        archived_ids = _apply_forgetting_setup(runtime, case)
        retrieval_supported = True

    setup_ms = (time.perf_counter() - setup_started) * 1000
    is_budget_case = budget is not None and case.category == "budget"

    if is_budget_case:
        prediction_before, latency_before, before = _run_once(
            agent_name, agent, case, None, trace_id=case_trace_id
        )
        prediction_after, latency_after, after = _run_once(
            agent_name, agent, case, budget, trace_id=case_trace_id
        )
    else:
        prediction_after, latency_after, after = _run_once(
            agent_name, agent, case, budget, trace_id=case_trace_id
        )
        prediction_before = None
        latency_before = None
        before = None

    after_answer = answer_metrics(
        prediction_after,
        case.expected_answer,
        case.answer_aliases,
    )
    answer_correct = int(after_answer["answer_accuracy"])
    historical_retrieval_correct = (
        int(bool(set(case.expected_memory_ids) & set(after["ranked_ids"])))
        if case.category == "temporal"
        else None
    )
    budget_after_tokens = estimate_tokens(after["prompt"])
    budget_before_tokens = (
        estimate_tokens(before["prompt"]) if before is not None else None
    )
    row = {
        "agent": agent_name,
        "case_id": case.case_id,
        "category": case.category,
        "difficulty": case.difficulty,
        "repeat": repeat,
        "query": case.query,
        "query_time": case.memory_query_time,
        "expected_answer": case.expected_answer,
        "answer_aliases": case.answer_aliases,
        "expected_version": case.expected_version,
        "prediction": prediction_after,
        "correct": answer_correct,
        "exact_match": int(after_answer["exact_match"]),
        "normalized_match": int(after_answer["normalized_match"]),
        "answer_match": int(after_answer["answer_match"]),
        "answer_accuracy": answer_correct,
        "answer_precision": after_answer["answer_precision"],
        "answer_recall": after_answer["answer_recall"],
        "answer_f1": after_answer["answer_f1"],
        "normalized_prediction": after_answer["normalized_prediction"],
        "answer_candidate": after_answer["answer_candidate"],
        "normalized_expected": after_answer["normalized_expected"],
        "matched_target": after_answer["matched_target"],
        "historical_query_correct": (
            answer_correct if case.category == "temporal" else None
        ),
        "historical_retrieval_correct": historical_retrieval_correct,
        "prompt_tokens": budget_after_tokens,
        "token_count": budget_after_tokens,
        "context_tokens": estimate_tokens(after["context"]),
        "setup_latency_ms": round(setup_ms, 4),
        "latency_ms": round(latency_after, 4),
        "end_to_end_latency_ms": round(latency_after, 4),
        "memory_latency_ms": round(after["memory_latency_ms"], 4),
        "context_build_latency_ms": round(after["context_latency_ms"], 4),
        "llm_latency_ms": round(after["llm_latency_ms"], 4),
        "post_latency_ms": round(after["post_latency_ms"], 4),
        "ttft_ms": None,
        "api_prompt_tokens": after["api_usage"].get("prompt_tokens"),
        "api_completion_tokens": after["api_usage"].get("completion_tokens"),
        "api_total_tokens": after["api_usage"].get("total_tokens"),
        "retrieval_supported": retrieval_supported,
        "trace_enabled": trace_enabled if agent_name == "Ours" else False,
        "retrieved_memory_ids": after["ranked_ids"],
        "expected_memory_ids": case.expected_memory_ids,
        "forbidden_memory_ids": case.forbidden_memory_ids,
        "complete_chain_hit": (
            int(set(case.expected_memory_ids).issubset(set(after["ranked_ids"])))
            if retrieval_supported and case.expected_memory_ids
            else None
        ),
        "forbidden_retrieved_count": len(
            set(case.forbidden_memory_ids) & set(after["ranked_ids"])
        ),
        "stale_memory_retrieval": (
            int(bool(set(case.forbidden_memory_ids) & set(after["ranked_ids"])))
            if case.forbidden_memory_ids
            else None
        ),
        "budget_before_prompt_tokens": budget_before_tokens,
        "budget_after_prompt_tokens": (
            budget_after_tokens if is_budget_case else None
        ),
        "budget_before_context_tokens": (
            estimate_tokens(before["context"]) if before is not None else None
        ),
        "budget_after_context_tokens": (
            estimate_tokens(after["context"]) if is_budget_case else None
        ),
        "budget_satisfied": (
            budget_after_tokens <= budget if is_budget_case else None
        ),
        "accuracy_before_budget": (
            int(answer_metrics(
                prediction_before,
                case.expected_answer,
                case.answer_aliases,
            )["answer_accuracy"])
            if prediction_before is not None
            else None
        ),
        "accuracy_after_budget": answer_correct if is_budget_case else None,
        "budget_accuracy_delta": (
            answer_correct
            - int(answer_metrics(
                prediction_before,
                case.expected_answer,
                case.answer_aliases,
            )["answer_accuracy"])
            if prediction_before is not None
            else None
        ),
        "budget_token_delta": (
            budget_after_tokens - budget_before_tokens
            if budget_before_tokens is not None
            else None
        ),
        "latency_before_budget_ms": (
            round(latency_before, 4) if latency_before is not None else None
        ),
        "latency_after_budget_ms": (
            round(latency_after, 4) if is_budget_case else None
        ),
    }
    if retrieval_supported:
        row.update(retrieval_metrics(after["ranked_ids"], case.expected_memory_ids))
    else:
        row.update(_empty_retrieval_metrics())
    if agent_name == "Ours":
        row["trace_id"] = agent.last_trace_id
        row["trace_events"] = (
            runtime.trace(trace_id=agent.last_trace_id)
            if trace_enabled and agent.last_trace_id is not None
            else []
        )
        row["archived_memory_ids"] = archived_ids
        row["consolidation_strategy"] = (
            consolidation_strategy if case.category == "consolidation" else None
        )
        if consolidation_report is not None:
            row["consolidation_created_ids"] = consolidation_report.created_ids
            row["consolidation_updated_ids"] = consolidation_report.updated_ids
            row["consolidation_source_count"] = consolidation_report.source_count
            row["consolidation_fidelity"] = consolidation_report.fidelity
            row["consolidation_latency_ms"] = round(consolidation_latency_ms or 0.0, 4)
            row["consolidation_skipped_groups"] = consolidation_report.skipped_groups
            row["consolidation_skipped_by_policy"] = consolidation_report.skipped_by_policy
            row["consolidation_conflict_blocked_groups"] = consolidation_report.conflict_blocked_groups
            row["consolidation_groups"] = [
                {
                    "group_key": group.group_key,
                    "source_ids": group.source_ids,
                    "semantic_memory_id": group.semantic_memory_id,
                    "fidelity": group.fidelity,
                    "action": group.action,
                    "trigger_score": group.trigger_score,
                    "granularity_score": group.granularity_score,
                    "granularity_level": group.granularity_level,
                    "policy_version": group.policy_version,
                    "tokens_before": group.tokens_before,
                    "tokens_after": group.tokens_after,
                    "compression_ratio": group.compression_ratio,
                }
                for group in consolidation_report.groups
            ]
            semantic_records = [
                record
                for record in runtime.store.list_all("benchmark")
                if record.memory_type == MemoryType.SEMANTIC
            ]
            source_ids = {
                source_id
                for group in consolidation_report.groups
                for source_id in group.source_ids
            }
            source_contents = {
                record.content
                for record in runtime.store.list_all("benchmark")
                if record.memory_id in source_ids
            }
            source_benchmark_ids = {
                _case_content_map(case)[content]
                for content in source_contents
                if content in _case_content_map(case)
            }
            expected_source_ids = set(case.expected_memory_ids)
            row["consolidation_key_fact_recall"] = (
                len(source_benchmark_ids & expected_source_ids) / len(expected_source_ids)
                if expected_source_ids
                else None
            )
            row["semantic_memory_tokens"] = sum(
                estimate_tokens(record.content) for record in semantic_records
            )
            tokens_before = sum(group.tokens_before for group in consolidation_report.groups)
            tokens_after = sum(group.tokens_after for group in consolidation_report.groups)
            row["consolidation_compression_ratio"] = (
                1.0 - tokens_after / tokens_before if tokens_before else None
            )
        else:
            row["consolidation_created_ids"] = []
            row["consolidation_updated_ids"] = []
            row["consolidation_source_count"] = None
            row["consolidation_fidelity"] = None
            row["consolidation_latency_ms"] = None
            row["consolidation_groups"] = []
            row["consolidation_key_fact_recall"] = None
            row["semantic_memory_tokens"] = None
            row["consolidation_compression_ratio"] = None
    return row


def _std(values: list[float]) -> float:
    return stdev(values) if len(values) > 1 else 0.0


def _percentile(values: list[float], quantile: float) -> float:
    ordered = sorted(values)
    index = int(round(quantile * (len(ordered) - 1)))
    return ordered[index]


def _numeric_summary(item: dict, rows: list[dict], key: str) -> None:
    values = [float(row[key]) for row in rows if row.get(key) is not None]
    item[f"avg_{key}"] = mean(values) if values else None
    item[f"std_{key}"] = _std(values) if values else None


def summarize(rows: Iterable[dict]) -> list[dict]:
    grouped: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        grouped[row["agent"]].append(row)

    summary: list[dict] = []
    for agent_name in AGENT_NAMES:
        agent_rows = grouped.get(agent_name, [])
        if not agent_rows:
            continue
        item = {
            "agent": agent_name,
            "num_runs": len(agent_rows),
            "num_cases": len({row["case_id"] for row in agent_rows}),
        }
        _numeric_summary(item, agent_rows, "correct")
        item["accuracy"] = item["avg_correct"]
        item["accuracy_std"] = item["std_correct"]
        for key in (
            "exact_match",
            "normalized_match",
            "answer_accuracy",
            "answer_precision",
            "answer_recall",
            "answer_f1",
        ):
            _numeric_summary(item, agent_rows, key)
        for key in ("prompt_tokens", "token_count", "context_tokens"):
            _numeric_summary(item, agent_rows, key)

        latencies = [float(row["latency_ms"]) for row in agent_rows]
        item["latency_mean_ms"] = mean(latencies)
        item["latency_std_ms"] = _std(latencies)
        item["latency_p50_ms"] = _percentile(latencies, 0.50)
        item["latency_p95_ms"] = _percentile(latencies, 0.95)
        item["latency_p99_ms"] = _percentile(latencies, 0.99)

        for key in (
            "end_to_end_latency_ms",
            "memory_latency_ms",
            "context_build_latency_ms",
            "llm_latency_ms",
            "post_latency_ms",
        ):
            _numeric_summary(item, agent_rows, key)

        for key in RETRIEVAL_KEYS:
            _numeric_summary(item, agent_rows, key)

        historical_answer = [
            float(row["historical_query_correct"])
            for row in agent_rows
            if row.get("historical_query_correct") is not None
        ]
        item["historical_query_accuracy"] = (
            mean(historical_answer) if historical_answer else None
        )
        item["historical_query_accuracy_std"] = (
            _std(historical_answer) if historical_answer else None
        )
        historical_retrieval = [
            float(row["historical_retrieval_correct"])
            for row in agent_rows
            if row.get("historical_retrieval_correct") is not None
        ]
        item["historical_retrieval_accuracy"] = (
            mean(historical_retrieval) if historical_retrieval else None
        )
        item["historical_retrieval_accuracy_std"] = (
            _std(historical_retrieval) if historical_retrieval else None
        )

        for key in (
            "budget_before_prompt_tokens",
            "budget_after_prompt_tokens",
            "budget_before_context_tokens",
            "budget_after_context_tokens",
            "budget_satisfied",
            "accuracy_before_budget",
            "accuracy_after_budget",
            "budget_accuracy_delta",
            "budget_token_delta",
            "latency_before_budget_ms",
            "latency_after_budget_ms",
            "consolidation_fidelity",
            "consolidation_latency_ms",
            "consolidation_key_fact_recall",
            "semantic_memory_tokens",
            "consolidation_compression_ratio",
            "complete_chain_hit",
            "forbidden_retrieved_count",
            "stale_memory_retrieval",
        ):
            _numeric_summary(item, agent_rows, key)
        summary.append(item)
    return summary


def run_benchmark(
    cases: Iterable[BenchmarkCase],
    agent_names: Iterable[str],
    client_factory: Callable[[], object],
    embedder_factory: Callable[[], object],
    top_k: int = 5,
    token_budget: int | None = None,
    repeats: int = 1,
    consolidation_strategy: str = "fixed",
    consolidation_policy: AdaptiveConsolidationPolicy | None = None,
    trace_enabled: bool = False,
) -> tuple[list[dict], list[dict]]:
    if repeats < 1:
        raise ValueError("repeats must be >= 1")
    cases = list(cases)
    agent_names = list(agent_names)
    rows: list[dict] = []
    sequence = 0
    for repeat in range(1, repeats + 1):
        for case_index, case in enumerate(cases):
            # Query-level alternating order limits time-correlated API bias.
            ordered_agents = (
                agent_names
                if (case_index + repeat - 1) % 2 == 0
                else list(reversed(agent_names))
            )
            for order_in_query, agent_name in enumerate(ordered_agents, start=1):
                row = run_case(
                    agent_name=agent_name,
                    case=case,
                    client_factory=client_factory,
                    embedder_factory=embedder_factory,
                    top_k=top_k,
                    token_budget=token_budget,
                    repeat=repeat,
                    consolidation_strategy=consolidation_strategy,
                    consolidation_policy=consolidation_policy,
                    trace_enabled=trace_enabled,
                )
                sequence += 1
                row["execution_sequence"] = sequence
                row["execution_order_in_query"] = order_in_query
                row["execution_policy"] = "query_interleaved_alternating"
                rows.append(row)
    return rows, summarize(rows)
