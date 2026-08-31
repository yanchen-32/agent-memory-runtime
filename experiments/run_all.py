from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from agent import ANSWER_FORMAT_VERSION, OpenAICompatibleClient, RuleBasedClient
from benchmark import (
    JsonlRunCheckpoint,
    ReusableEmbeddingFactory,
    load_jsonl,
    sha256_file,
)
from benchmark.runner import AGENT_NAMES, run_benchmark


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run B0/B1/B2/Ours with one comparable benchmark runner."
    )
    parser.add_argument(
        "--benchmark",
        type=Path,
        default=ROOT / "benchmark" / "data" / "benchmark_v0.2.jsonl",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=ROOT / "results",
    )
    parser.add_argument(
        "--case-id",
        default=None,
        help="Run exactly one benchmark case by ID (useful for real-model smoke tests).",
    )
    parser.add_argument(
        "--output-prefix",
        default=None,
        help="Optional prefix for detail and summary JSON files.",
    )
    parser.add_argument(
        "--agents",
        default="B0,B1,B2,B3,Ours",
        help="Comma-separated subset of B0,B1,B2,B3,Ours.",
    )
    parser.add_argument("--repeats", type=int, default=1)
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument(
        "--token-budget",
        type=int,
        default=None,
        help="Global prompt budget; case-level budget takes precedence.",
    )
    parser.add_argument(
        "--client",
        choices=("rule", "openai"),
        default="rule",
        help="rule is offline deterministic; openai uses an OpenAI-compatible endpoint.",
    )
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
        help="Record terminal failures and continue remaining runs.",
    )
    parser.add_argument("--resume", action="store_true")
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
    if args.repeats < 1:
        raise ValueError("--repeats must be >= 1")

    agent_names = [name.strip() for name in args.agents.split(",") if name.strip()]
    unknown = sorted(set(agent_names) - set(AGENT_NAMES))
    if unknown:
        raise ValueError(f"unknown agents: {unknown}")

    cases = load_jsonl(args.benchmark)
    if args.case_id is not None:
        cases = [case for case in cases if case.case_id == args.case_id]
        if not cases:
            raise ValueError(f"case ID not found in benchmark: {args.case_id}")

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
    prefix = args.output_prefix or "unified_v0.2"
    checkpoint_config = {
        "benchmark": str(args.benchmark.resolve()),
        "benchmark_sha256": sha256_file(args.benchmark),
        "case_ids": [case.case_id for case in cases],
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
    }
    checkpoint = JsonlRunCheckpoint(
        args.output_dir / f"{prefix}_checkpoint.jsonl",
        checkpoint_config,
    )
    existing_rows = checkpoint.prepare(resume=args.resume, retry_failed=True)

    rows, summary = run_benchmark(
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

    if args.output_prefix is None:
        detail_path = args.output_dir / "unified_results_v0.2.json"
        summary_path = args.output_dir / "unified_summary_v0.2.json"
    else:
        detail_path = args.output_dir / f"{args.output_prefix}_results.json"
        summary_path = args.output_dir / f"{args.output_prefix}_summary.json"
    detail_path.write_text(
        json.dumps(rows, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    summary_path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print(f"cases: {len(cases)}, agents: {', '.join(agent_names)}, repeats: {args.repeats}")
    print("agent | succeeded/failed | accuracy | avg_prompt_tokens | avg_context_tokens | p50_ms | p95_ms | avg_recall@5 | avg_mrr")
    def display(value, digits=4):
        return "NA" if value is None else f"{value:.{digits}f}"

    for item in summary:
        print(
            f"{item['agent']} | "
            f"{item['num_succeeded']}/{item['num_failed']} | "
            f"{display(item['accuracy'], 3)} | "
            f"{display(item['avg_prompt_tokens'], 1)} | "
            f"{display(item['avg_context_tokens'], 1)} | "
            f"{display(item['latency_p50_ms'])} | "
            f"{display(item['latency_p95_ms'])} | "
            f"{item['avg_recall@5'] if item['avg_recall@5'] is not None else 'NA'} | "
            f"{item['avg_mrr'] if item['avg_mrr'] is not None else 'NA'}"
        )
    print(f"detail results: {detail_path}")
    print(f"summary results: {summary_path}")


if __name__ == "__main__":
    main()
