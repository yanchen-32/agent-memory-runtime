from __future__ import annotations

from collections import defaultdict
from datetime import datetime, timezone
from statistics import mean, stdev
import time
from typing import Callable, Iterable

from agent import FullHistoryAgent, MemoryRuntimeAgent, NoMemoryAgent, VectorMemoryAgent
from memory import HashEmbeddingModel, MemoryRuntimeV1, VectorMemoryStore

from .loader import BenchmarkCase
from .metrics import answer_metrics, estimate_tokens, retrieval_metrics


AGENT_NAMES = ("B0", "B1", "B2", "Ours")
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


def _invoke(agent_name: str, agent, case: BenchmarkCase, token_budget: int | None):
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
    return agent.answer(
        case.query,
        token_budget=token_budget,
        query_time=case.memory_query_time,
    )


def _snapshot(agent_name: str, agent, case: BenchmarkCase) -> dict:
    if agent_name == "Ours":
        ranked_ids = _map_runtime_ids(agent, case)
    else:
        ranked_ids = list(getattr(agent, "last_retrieved_ids", []))
    return {
        "prompt": getattr(agent, "last_prompt", ""),
        "context": getattr(agent, "last_context", ""),
        "ranked_ids": ranked_ids,
    }


def _run_once(
    agent_name: str,
    agent,
    case: BenchmarkCase,
    token_budget: int | None,
) -> tuple[str, float, dict]:
    started = time.perf_counter()
    prediction = _invoke(agent_name, agent, case, token_budget)
    latency_ms = (time.perf_counter() - started) * 1000
    snapshot = _snapshot(agent_name, agent, case)
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
) -> dict:
    if agent_name not in AGENT_NAMES:
        raise ValueError(f"unknown agent: {agent_name}")

    client = client_factory()
    budget = case.token_budget if case.token_budget is not None else token_budget
    setup_started = time.perf_counter()
    archived_ids: list[str] = []

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
    else:
        runtime = MemoryRuntimeV1(embedder=embedder_factory())
        agent = MemoryRuntimeAgent(client, runtime, top_k=top_k)
        agent.ingest(
            case.conversation,
            user_id="benchmark",
            session_id=f"{case.case_id}-repeat-{repeat}",
        )
        archived_ids = _apply_forgetting_setup(runtime, case)
        retrieval_supported = True

    setup_ms = (time.perf_counter() - setup_started) * 1000
    is_budget_case = budget is not None and case.category == "budget"

    if is_budget_case:
        prediction_before, latency_before, before = _run_once(
            agent_name, agent, case, None
        )
        prediction_after, latency_after, after = _run_once(
            agent_name, agent, case, budget
        )
    else:
        prediction_after, latency_after, after = _run_once(
            agent_name, agent, case, budget
        )
        prediction_before = None
        latency_before = None
        before = None

    answer_correct = int(
        answer_metrics(prediction_after, case.expected_answer)["exact_match"]
    )
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
        "expected_version": case.expected_version,
        "prediction": prediction_after,
        "correct": answer_correct,
        "historical_query_correct": (
            answer_correct if case.category == "temporal" else None
        ),
        "historical_retrieval_correct": historical_retrieval_correct,
        "prompt_tokens": budget_after_tokens,
        "token_count": budget_after_tokens,
        "context_tokens": estimate_tokens(after["context"]),
        "setup_latency_ms": round(setup_ms, 4),
        "latency_ms": round(latency_after, 4),
        "retrieval_supported": retrieval_supported,
        "retrieved_memory_ids": after["ranked_ids"],
        "expected_memory_ids": case.expected_memory_ids,
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
            int(answer_metrics(prediction_before, case.expected_answer)["exact_match"])
            if prediction_before is not None
            else None
        ),
        "accuracy_after_budget": answer_correct if is_budget_case else None,
        "budget_accuracy_delta": (
            answer_correct
            - int(answer_metrics(prediction_before, case.expected_answer)["exact_match"])
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
        row["archived_memory_ids"] = archived_ids
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
        for key in ("prompt_tokens", "token_count", "context_tokens"):
            _numeric_summary(item, agent_rows, key)

        latencies = [float(row["latency_ms"]) for row in agent_rows]
        item["latency_mean_ms"] = mean(latencies)
        item["latency_std_ms"] = _std(latencies)
        item["latency_p50_ms"] = _percentile(latencies, 0.50)
        item["latency_p95_ms"] = _percentile(latencies, 0.95)

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
) -> tuple[list[dict], list[dict]]:
    if repeats < 1:
        raise ValueError("repeats must be >= 1")
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
