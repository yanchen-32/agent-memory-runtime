from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from agent import ANSWER_FORMAT_VERSION, OpenAICompatibleClient, RuleBasedClient
from benchmark import (
    LEGACY_ANSWER_SCORER_VERSION,
    QUANTITY_ANSWER_SCORER_VERSION,
    JsonlRunCheckpoint,
    ReusableEmbeddingFactory,
    collect_environment,
    load_jsonl,
    paired_bootstrap_comparison,
    paired_latency_reduction,
    require_frozen_benchmark,
    sha256_file,
    write_formal_artifacts,
)
from benchmark.runner import AGENT_NAMES, run_benchmark, summarize


E1_CATEGORIES = {
    "fact_recall", "semantic_recall", "long_context", "noise",
    "abstention", "budget", "multi_hop", "forgetting",
}
E2_CATEGORIES = {"update", "conflict", "temporal"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run formal E1 long-term and E2 update/conflict experiments."
    )
    parser.add_argument(
        "--benchmark",
        type=Path,
        default=ROOT / "benchmark" / "data" / "benchmark_v0.2.jsonl",
    )
    parser.add_argument("--output-dir", type=Path, default=ROOT / "results")
    parser.add_argument("--agents", default="B0,B1,B2,B3,Ours")
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--token-budget", type=int, default=None)
    parser.add_argument("--protocol-version", default="1.1")
    parser.add_argument("--bootstrap-samples", type=int, default=10_000)
    parser.add_argument("--client", choices=("rule", "openai"), default="rule")
    parser.add_argument("--model", default=os.getenv("LLM_MODEL"))
    parser.add_argument(
        "--base-url",
        default=os.getenv("LLM_BASE_URL", "http://localhost:8000/v1"),
    )
    parser.add_argument("--api-key", default=os.getenv("LLM_API_KEY", "EMPTY"))
    parser.add_argument("--timeout", type=int, default=60)
    parser.add_argument("--max-retries", type=int, default=2)
    parser.add_argument("--retry-backoff-seconds", type=float, default=1.0)
    parser.add_argument(
        "--continue-on-error",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument("--resume", action="store_true")
    parser.add_argument(
        "--allow-unreviewed-benchmark",
        action="store_true",
        help="Development pilot only; formal claims are forbidden.",
    )
    parser.add_argument("--trace", action="store_true", help="Record full mechanistic trace events.")
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top-p", type=float, default=1.0)
    parser.add_argument("--max-tokens", type=int, default=None)
    parser.add_argument(
        "--thinking",
        choices=("disabled", "enabled", "omit"),
        default="disabled",
    )
    parser.add_argument(
        "--embedding",
        choices=("hash", "sentence-transformers"),
        default="hash",
    )
    parser.add_argument(
        "--embedding-model",
        default="BAAI/bge-small-zh-v1.5",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.repeats < 3:
        raise ValueError("Formal E1/E2 experiments require --repeats >= 3")
    if not args.allow_unreviewed_benchmark:
        require_frozen_benchmark(args.benchmark)

    agent_names = [name.strip() for name in args.agents.split(",") if name.strip()]
    unknown = sorted(set(agent_names) - set(AGENT_NAMES))
    if unknown:
        raise ValueError(f"unknown agents: {unknown}")

    cases = load_jsonl(args.benchmark)

    def client_factory():
        if args.client == "rule":
            return RuleBasedClient()
        return OpenAICompatibleClient(
            model=args.model,
            base_url=args.base_url,
            api_key=args.api_key,
            timeout=args.timeout,
            temperature=args.temperature,
            top_p=args.top_p,
            max_tokens=args.max_tokens,
            thinking=None if args.thinking == "omit" else args.thinking,
            max_retries=args.max_retries,
            retry_backoff_seconds=args.retry_backoff_seconds,
        )

    embedder_factory = ReusableEmbeddingFactory(
        args.embedding,
        args.embedding_model,
        hash_dim=384,
    )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_config = {
        "protocol_version": args.protocol_version,
        "benchmark": str(args.benchmark.resolve()),
        "benchmark_sha256": sha256_file(args.benchmark),
        "agents": agent_names,
        "repeats": args.repeats,
        "top_k": args.top_k,
        "token_budget": args.token_budget,
        "client": args.client,
        "model": args.model if args.client == "openai" else "RuleBasedClient",
        "base_url": args.base_url if args.client == "openai" else None,
        "timeout": args.timeout,
        "max_retries": args.max_retries,
        "retry_backoff_seconds": args.retry_backoff_seconds,
        "temperature": args.temperature,
        "top_p": args.top_p,
        "max_tokens": args.max_tokens,
        "thinking": args.thinking,
        "embedding": args.embedding,
        "embedding_model": args.embedding_model,
        "trace": args.trace,
        "answer_format_version": ANSWER_FORMAT_VERSION,
        "answer_scorer_versions": [
            LEGACY_ANSWER_SCORER_VERSION,
            QUANTITY_ANSWER_SCORER_VERSION,
        ],
        "benchmark_review_enforced": not args.allow_unreviewed_benchmark,
    }
    checkpoint = JsonlRunCheckpoint(
        args.output_dir / "run_checkpoint.jsonl",
        checkpoint_config,
    )
    existing_rows = checkpoint.prepare(resume=args.resume, retry_failed=True)

    rows, _ = run_benchmark(
        cases=cases,
        agent_names=agent_names,
        client_factory=client_factory,
        embedder_factory=embedder_factory,
        top_k=args.top_k,
        token_budget=args.token_budget,
        repeats=args.repeats,
        trace_enabled=args.trace,
        existing_rows=existing_rows,
        on_row=checkpoint.append,
        continue_on_error=args.continue_on_error,
    )

    e1_rows = [row for row in rows if row["category"] in E1_CATEGORIES]
    e2_rows = [row for row in rows if row["category"] in E2_CATEGORIES]
    budget_rows = [row for row in e1_rows if row["category"] == "budget"]
    generated_at = datetime.now(timezone.utc).isoformat()
    environment = collect_environment(ROOT, args.benchmark)

    experiment_summaries = {
        "E1": summarize(e1_rows),
        "E2": summarize(e2_rows),
    }
    comparisons = {
        "E1_answer_f1_B1_vs_Ours": paired_bootstrap_comparison(
            e1_rows, samples=args.bootstrap_samples
        ),
        "E2_answer_f1_B1_vs_Ours": paired_bootstrap_comparison(
            e2_rows, samples=args.bootstrap_samples
        ),
        "Budget_semantic_accuracy_B1_vs_Ours": paired_bootstrap_comparison(
            budget_rows,
            metric="semantic_answer_accuracy",
            samples=args.bootstrap_samples,
        ),
        "Budget_task_success_B1_vs_Ours": paired_bootstrap_comparison(
            budget_rows,
            metric="budget_task_success",
            samples=args.bootstrap_samples,
        ),
        "E1_e2e_latency_B1_vs_Ours": paired_latency_reduction(e1_rows),
        "E2_e2e_latency_B1_vs_Ours": paired_latency_reduction(e2_rows),
    }
    for experiment, experiment_rows in (("E1", e1_rows), ("E2", e2_rows)):
        failure_count = sum(row.get("status") == "failed" for row in experiment_rows)
        for suffix in ("answer_f1_B1_vs_Ours", "e2e_latency_B1_vs_Ours"):
            comparison = comparisons[f"{experiment}_{suffix}"]
            comparison["formal_comparison_valid"] = failure_count == 0
            comparison["failure_count"] = failure_count
            if failure_count:
                comparison["invalid_reason"] = "terminal run failures present"
    budget_failure_count = sum(row.get("status") == "failed" for row in budget_rows)
    for key in (
        "Budget_semantic_accuracy_B1_vs_Ours",
        "Budget_task_success_B1_vs_Ours",
    ):
        comparisons[key]["formal_comparison_valid"] = budget_failure_count == 0
        comparisons[key]["failure_count"] = budget_failure_count
        if budget_failure_count:
            comparisons[key]["invalid_reason"] = "terminal Budget run failures present"
    safe_config = {
        "protocol_version": args.protocol_version,
        "benchmark": str(args.benchmark.resolve()),
        "agents": agent_names,
        "repeats": args.repeats,
        "top_k": args.top_k,
        "token_budget": args.token_budget,
        "client": args.client,
        "model": args.model if args.client == "openai" else "RuleBasedClient",
        "base_url": args.base_url if args.client == "openai" else None,
        "timeout": args.timeout,
        "max_retries": args.max_retries,
        "retry_backoff_seconds": args.retry_backoff_seconds,
        "continue_on_error": args.continue_on_error,
        "temperature": args.temperature,
        "top_p": args.top_p,
        "max_tokens": args.max_tokens,
        "thinking": args.thinking,
        "embedding": args.embedding,
        "embedding_model": (
            args.embedding_model if args.embedding == "sentence-transformers"
            else "HashEmbeddingModel(dim=384)"
        ),
        "execution_policy": "query_interleaved_alternating",
        "trace_enabled": args.trace,
        "bootstrap_samples": args.bootstrap_samples,
        "bootstrap_seed": 202601,
        "answer_format_version": ANSWER_FORMAT_VERSION,
        "answer_scorer_versions": [
            LEGACY_ANSWER_SCORER_VERSION,
            QUANTITY_ANSWER_SCORER_VERSION,
        ],
        "benchmark_review_enforced": not args.allow_unreviewed_benchmark,
    }

    benchmark_label = (
        f"v1.0_{args.benchmark.stem}"
        if args.benchmark.parent.name == "v1.0"
        else args.benchmark.stem.removeprefix("benchmark_")
    )
    outputs = [
        (
            f"e1_long_term_accuracy_{benchmark_label}.json",
            "E1",
            E1_CATEGORIES,
            e1_rows,
        ),
        (
            f"e2_update_conflict_{benchmark_label}.json",
            "E2",
            E2_CATEGORIES,
            e2_rows,
        ),
    ]
    for filename, experiment, categories, experiment_rows in outputs:
        payload = {
            "experiment": experiment,
            "protocol_version": args.protocol_version,
            "generated_at": generated_at,
            "benchmark": str(args.benchmark),
            "agents": agent_names,
            "repeats": args.repeats,
            "categories": sorted(categories),
            "rows": experiment_rows,
            "summary": experiment_summaries[experiment],
            "answer_f1_comparison": comparisons[
                f"{experiment}_answer_f1_B1_vs_Ours"
            ],
            "e2e_latency_comparison": comparisons[
                f"{experiment}_e2e_latency_B1_vs_Ours"
            ],
            "configuration": safe_config,
            "environment": environment,
        }
        (args.output_dir / filename).write_text(
            json.dumps(payload, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

    manifest = {
        "protocol_version": args.protocol_version,
        "generated_at": generated_at,
        "configuration": safe_config,
        "environment": environment,
        "secrets_recorded": False,
    }
    write_formal_artifacts(
        args.output_dir,
        rows=e1_rows + e2_rows,
        summary_payload={
            "protocol_version": args.protocol_version,
            "generated_at": generated_at,
            "experiments": experiment_summaries,
            "comparisons": comparisons,
        },
        manifest=manifest,
    )

    print(
        f"E1 rows: {len(e1_rows)}, E2 rows: {len(e2_rows)}, "
        f"agents: {', '.join(agent_names)}, repeats: {args.repeats}"
    )
    for label, experiment_rows in (("E1", e1_rows), ("E2", e2_rows)):
        print(f"{label} summary:")
        for item in summarize(experiment_rows):
            latency_mean = item["latency_mean_ms"]
            latency_p50 = item["latency_p50_ms"]
            latency_p95 = item["latency_p95_ms"]
            print(
                f"  {item['agent']}: succeeded={item['num_succeeded']} "
                f"failed={item['num_failed']}, "
                f"accuracy={item['accuracy'] if item['accuracy'] is not None else 'NA'} "
                f"+/- {item['accuracy_std'] if item['accuracy_std'] is not None else 'NA'}, "
                f"latency_mean={latency_mean if latency_mean is not None else 'NA'} ms, "
                f"p50={latency_p50 if latency_p50 is not None else 'NA'} ms, "
                f"p95={latency_p95 if latency_p95 is not None else 'NA'} ms"
            )
    print("Budget scoring summary:")
    for item in summarize(budget_rows):
        print(
            f"  {item['agent']}: "
            f"semantic_accuracy={item['avg_semantic_answer_accuracy']}, "
            f"format_compliance={item['avg_answer_format_compliance']}, "
            f"constraint_satisfaction={item['avg_budget_satisfied']}, "
            f"task_success={item['avg_budget_task_success']}"
        )


if __name__ == "__main__":
    main()
