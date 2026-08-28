from __future__ import annotations

from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from statistics import mean
import time
from typing import Callable, Iterable

from agent import (
    FullHistoryAgent,
    MemoryRuntimeAgent,
    NoMemoryAgent,
    OpenAICompatibleClient,
    RuleBasedClient,
    VectorMemoryAgent,
)
from memory import HashEmbeddingModel, MemoryRuntimeV1, VectorMemoryStore

from .loader import BenchmarkCase
from .metrics import answer_metrics, estimate_tokens, retrieval_metrics


AGENT_NAMES = ("B0", "B1", "B2", "Ours")
RETRIEVAL_KEYS = ("recall@1", "recall@5", "recall@10", "mrr")
NUMERIC_KEYS = (
    "correct",
    "prompt_tokens",
    "context_tokens",
    "latency_ms",
    *RETRIEVAL_KEYS,
)


def _parse_time(value: str | None) -> datetime | None:
    if not value:
        return None
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
    """Apply benchmark-controlled low-value attributes before lifecycle testing."""
    if not case.forget_memory_ids:
        return []

    content_to_id = {
        str(turn.get("content", "")): str(turn.get("memory_id", ""))
        for turn in case.conversation
    }
    target_ids = set(case.forget_memory_ids)
    attributes = case.memory_metadata or {}
    old_time = _parse_time(case.query_time)

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
        elif old_time is not None:
            record.created_at = old_time
        runtime.store.update(record)

    return runtime.forgetting.run(
        user_id="benchmark",
        now=_parse_time(case.query_time),
    )


def _build_client(
    client_mode: str,
    model: str | None,
    base_url: str,
    api_key: str,
    timeout: int,
):
    if client_mode == "rule":
        return RuleBasedClient()
    return OpenAICompatibleClient(
        model=model,
        base_url=base_url,
        api_key=api_key,
        timeout=timeout,
    )


def _map_runtime_ids(agent: MemoryRuntimeAgent, case: BenchmarkCase) -> list[str]:
    content_to_id = _case_content_map(case)
    mapped: list[str] = []
    for content in agent.last_retrieved_contents:
        mapped.append(content_to_id.get(content, ""))
    return [memory_id for memory_id in mapped if memory_id]


def _empty_retrieval_metrics() -> dict[str, float | None]:
    return {key: None for key in RETRIEVAL_KEYS}


def run_case(
    agent_name: str,
    case: BenchmarkCase,
    client_factory: Callable[[], object],
    embedder_factory: Callable[[], object],
    top_k: int = 5,
    token_budget: int | None = None,
    repeat: int = 1,
) -> dict:
    if agent_name not in AGENT_NAMES:
        raise ValueError(f"unknown agent: {agent_name}")

    client = client_factory()
    budget = case.token_budget if case.token_budget is not None else token_budget
    setup_started = time.perf_counter()

    if agent_name == "B0":
        agent = NoMemoryAgent(client)
        setup_ms = (time.perf_counter() - setup_started) * 1000
        started = time.perf_counter()
        prediction = agent.answer(case.query)
        latency_ms = (time.perf_counter() - started) * 1000
        ranked_ids: list[str] = []
        retrieval_supported = False

    elif agent_name == "B1":
        agent = FullHistoryAgent(client)
        setup_ms = (time.perf_counter() - setup_started) * 1000
        started = time.perf_counter()
        prediction = agent.answer(
            case.query,
            conversation=case.conversation,
            token_budget=budget,
        )
        latency_ms = (time.perf_counter() - started) * 1000
        ranked_ids = []
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
        setup_ms = (time.perf_counter() - setup_started) * 1000
        started = time.perf_counter()
        prediction = agent.answer(case.query, token_budget=budget)
        latency_ms = (time.perf_counter() - started) * 1000
        ranked_ids = list(agent.last_retrieved_ids)
        retrieval_supported = True

    else:
        runtime = MemoryRuntimeV1(embedder=embedder_factory())
        agent = MemoryRuntimeAgent(client, runtime, top_k=top_k)
        agent.ingest(
            case.conversation,
            user_id="benchmark",
            session_id=f"{case.case_id}-repeat-{repeat}",
        )
        archived_ids = _apply_forgetting_setup(runtime, case)
        setup_ms = (time.perf_counter() - setup_started) * 1000
        started = time.perf_counter()
        prediction = agent.answer(case.query, token_budget=budget)
        latency_ms = (time.perf_counter() - started) * 1000
        ranked_ids = _map_runtime_ids(agent, case)
        retrieval_supported = True

    prompt = getattr(agent, "last_prompt", "")
    context = getattr(agent, "last_context", "")
    row = {
        "agent": agent_name,
        "case_id": case.case_id,
        "category": case.category,
        "difficulty": case.difficulty,
        "repeat": repeat,
        "query": case.query,
        "expected_answer": case.expected_answer,
        "expected_version": case.expected_version,
        "prediction": prediction,
        "correct": int(answer_metrics(prediction, case.expected_answer)["exact_match"]),
        "prompt_tokens": estimate_tokens(prompt),
        "token_count": estimate_tokens(prompt),
        "context_tokens": estimate_tokens(context),
        "setup_latency_ms": round(setup_ms, 4),
        "latency_ms": round(latency_ms, 4),
        "retrieval_supported": retrieval_supported,
        "retrieved_memory_ids": ranked_ids,
        "expected_memory_ids": case.expected_memory_ids,
    }
    if retrieval_supported:
        row.update(retrieval_metrics(ranked_ids, case.expected_memory_ids))
    else:
        row.update(_empty_retrieval_metrics())
    if agent_name == "Ours":
        row["archived_memory_ids"] = archived_ids
    return row


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
            "accuracy": mean(row["correct"] for row in agent_rows),
        }
        for key in ("prompt_tokens", "token_count", "context_tokens"):
            item[f"avg_{key}"] = mean(row[key] for row in agent_rows)
        latencies = sorted(row["latency_ms"] for row in agent_rows)
        item["latency_mean_ms"] = mean(latencies)
        item["latency_p50_ms"] = latencies[int(0.50 * (len(latencies) - 1))]
        item["latency_p95_ms"] = latencies[int(0.95 * (len(latencies) - 1))]
        for key in RETRIEVAL_KEYS:
            values = [
                row[key]
                for row in agent_rows
                if row.get(key) is not None
            ]
            item[f"avg_{key}"] = mean(values) if values else None
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
) -> tuple[list[dict], list[dict]]:
    cases = list(cases)
    agent_names = list(agent_names)
    rows: list[dict] = []
    for repeat in range(1, repeats + 1):
        for agent_name in agent_names:
            for case in cases:
                rows.append(
                    run_case(
                        agent_name=agent_name,
                        case=case,
                        client_factory=client_factory,
                        embedder_factory=embedder_factory,
                        top_k=top_k,
                        token_budget=token_budget,
                        repeat=repeat,
                    )
                )
    return rows, summarize(rows)
