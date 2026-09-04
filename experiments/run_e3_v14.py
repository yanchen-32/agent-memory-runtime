"""Run frozen E3 v1.4 Development with B1, StructuredKV, and OursV2."""

from __future__ import annotations

# Direct execution adds the repository root before project imports.
# ruff: noqa: E402

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
from statistics import mean
import sys


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from agent import (
    ANSWER_FORMAT_VERSION,
    MEMORY_RUNTIME_METHOD_VERSION,
    MEMORY_RUNTIME_PROMPT_VERSION,
    OpenAICompatibleClient,
    RuleBasedClient,
    STRUCTURED_KV_BASELINE_VERSION,
)
from benchmark import (
    JsonlRunCheckpoint,
    ReusableEmbeddingFactory,
    collect_environment,
    load_jsonl,
    require_frozen_benchmark,
    sha256_file,
    write_formal_artifacts,
)
from benchmark.e3_v14_spec import DESIGN_VERSION, STRATA
from benchmark.runner import AGENT_NAMES, run_benchmark
from benchmark.validate_e3_v14 import validate_development
from experiments.run_e3_scaling import build_e3_analysis


DEFAULT_AGENTS = ("B1", "StructuredKV", "OursV2")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run frozen E3 v1.4 Development.")
    parser.add_argument(
        "--benchmark", type=Path,
        default=ROOT / "benchmark/data/v1.4-e3/development.jsonl",
    )
    parser.add_argument(
        "--output-dir", type=Path,
        default=ROOT / "results/formal/e3_v14_development",
    )
    parser.add_argument("--agents", default=",".join(DEFAULT_AGENTS))
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--bootstrap-samples", type=int, default=10_000)
    parser.add_argument("--client", choices=("rule", "openai"), default="rule")
    parser.add_argument("--model", default=os.getenv("LLM_MODEL"))
    parser.add_argument("--base-url", default=os.getenv("LLM_BASE_URL", "https://api.deepseek.com"))
    parser.add_argument("--api-key", default=os.getenv("LLM_API_KEY", "EMPTY"))
    parser.add_argument("--timeout", type=int, default=180)
    parser.add_argument("--max-retries", type=int, default=4)
    parser.add_argument("--retry-backoff-seconds", type=float, default=2.0)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top-p", type=float, default=1.0)
    parser.add_argument("--max-tokens", type=int, default=64)
    parser.add_argument("--thinking", choices=("disabled", "enabled", "omit"), default="disabled")
    parser.add_argument("--embedding", choices=("hash", "sentence-transformers"), default="hash")
    parser.add_argument("--embedding-model", default="BAAI/bge-small-zh-v1.5")
    parser.add_argument("--context-stability-tolerance", type=float, default=0.10)
    parser.add_argument("--continue-on-error", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument(
        "--resume-after-interruption", action="store_true",
        help="Only resume the exact append-only checkpoint after an interrupted Development run.",
    )
    return parser.parse_args()


def _enrich(row: dict, metadata: dict[str, dict]) -> dict:
    source = metadata[str(row["case_id"])]
    return {
        **row,
        "e3_scenario_family": source["scenario_family"],
        "e3_stratum": source["stratum"],
        "e3_predicate": source["predicate"],
        "e3_predicate_surface": source["predicate_surface"],
        "e3_query_expression_mode": source["query_expression_mode"],
        "e3_target_position_band": source["target_position_band"],
        "e3_target_b1_prompt_tokens": source["target_b1_prompt_tokens"],
        "e3_generated_b1_prompt_tokens": source["b1_prompt_tokens"],
        "e3_history_memory_count": source["history_memory_count"],
    }


def _route_summary(rows: list[dict]) -> dict:
    output = {}
    for stratum in STRATA:
        output[stratum] = {}
        for agent in DEFAULT_AGENTS:
            selected = [
                row for row in rows
                if row.get("status") == "succeeded"
                and row.get("e3_stratum") == stratum
                and row.get("agent") == agent
            ]
            fast = [row for row in selected if row.get("answer_route") == "structured_fast_path"]
            fallback = [row for row in selected if row.get("answer_route") == "llm_fallback"]
            hit = sum(int(row.get("api_prompt_cache_hit_tokens") or 0) for row in selected)
            miss = sum(int(row.get("api_prompt_cache_miss_tokens") or 0) for row in selected)
            output[stratum][agent] = {
                "runs": len(selected),
                "fast_path_hits": len(fast),
                "fast_path_hit_rate": len(fast) / len(selected) if selected else None,
                "fast_path_latency_mean_ms": mean(float(row["fast_path_latency_ms"]) for row in fast) if fast else None,
                "llm_fallback_runs": len(fallback),
                "llm_fallback_latency_mean_ms": mean(float(row["llm_fallback_latency_ms"]) for row in fallback) if fallback else None,
                "prompt_cache_hit_tokens": hit,
                "prompt_cache_miss_tokens": miss,
                "prompt_cache_token_hit_rate": hit / (hit + miss) if hit + miss else None,
            }
    return output


def main() -> None:
    args = parse_args()
    if args.repeats < 3:
        raise ValueError("v1.4 Development requires at least three repeats")
    validation = validate_development(args.benchmark)
    frozen_manifest = require_frozen_benchmark(args.benchmark)
    agents = [name.strip() for name in args.agents.split(",") if name.strip()]
    if sorted(agents) != sorted(DEFAULT_AGENTS):
        raise ValueError(f"formal v1.4 agents are locked to {DEFAULT_AGENTS}")
    if any(name not in AGENT_NAMES for name in agents):
        raise ValueError("unknown agent in v1.4 run")
    if args.output_dir.exists() and any(args.output_dir.iterdir()) and not args.resume_after_interruption:
        raise ValueError("output directory is not empty; start in a new directory")

    cases = load_jsonl(args.benchmark)
    metadata = {case.case_id: case.metadata for case in cases}

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
            persistent_connections=True,
        )

    embedder_factory = ReusableEmbeddingFactory(args.embedding, args.embedding_model)
    configuration = {
        "experiment": "E3-v1.4-Development",
        "benchmark": str(args.benchmark.resolve()),
        "benchmark_sha256": sha256_file(args.benchmark),
        "frozen_manifest_sha256": sha256_file(args.benchmark.parent / "frozen_manifest.json"),
        "design_version": DESIGN_VERSION,
        "agents": agents,
        "repeats": args.repeats,
        "top_k": args.top_k,
        "client": args.client,
        "model": args.model if args.client == "openai" else "RuleBasedClient",
        "base_url": args.base_url if args.client == "openai" else None,
        "persistent_http_connections": args.client == "openai",
        "prompt_cache_metrics_recorded": True,
        "timeout": args.timeout,
        "max_retries": args.max_retries,
        "retry_backoff_seconds": args.retry_backoff_seconds,
        "temperature": args.temperature,
        "top_p": args.top_p,
        "max_tokens": args.max_tokens,
        "thinking": args.thinking,
        "embedding": args.embedding,
        "embedding_model": args.embedding_model,
        "method_version": MEMORY_RUNTIME_METHOD_VERSION,
        "prompt_version": MEMORY_RUNTIME_PROMPT_VERSION,
        "structured_kv_version": STRUCTURED_KV_BASELINE_VERSION,
        "answer_format_version": ANSWER_FORMAT_VERSION,
        "execution_policy": "query_interleaved_alternating",
        "bootstrap_samples": args.bootstrap_samples,
        "test_accessed": False,
    }
    checkpoint = JsonlRunCheckpoint(args.output_dir / "run_checkpoint.jsonl", configuration)
    existing = checkpoint.prepare(
        resume=args.resume_after_interruption,
        retry_failed=args.resume_after_interruption,
    )

    def append(row: dict) -> None:
        checkpoint.append(_enrich(row, metadata))

    rows, _ = run_benchmark(
        cases=cases,
        agent_names=agents,
        client_factory=client_factory,
        embedder_factory=embedder_factory,
        top_k=args.top_k,
        repeats=args.repeats,
        existing_rows=existing,
        on_row=append,
        continue_on_error=args.continue_on_error,
    )
    rows = [_enrich(row, metadata) for row in rows]
    analysis = build_e3_analysis(
        rows,
        bootstrap_samples=args.bootstrap_samples,
        context_stability_tolerance=args.context_stability_tolerance,
        benchmark_review_enforced=True,
        expected_case_ids={case.case_id for case in cases},
        expected_agents=set(agents),
        expected_repeats=args.repeats,
        treatment_agent="OursV2",
    )
    routes = _route_summary(rows)
    generated_at = datetime.now(timezone.utc).isoformat()
    payload = {
        "experiment": "E3-v1.4-Development",
        "generated_at": generated_at,
        "configuration": configuration,
        "benchmark_validation": validation,
        "frozen_manifest": frozen_manifest,
        "environment": collect_environment(ROOT, args.benchmark),
        "route_and_cache_metrics": routes,
        "rows": rows,
        **analysis,
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "e3_v14_development.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    write_formal_artifacts(
        args.output_dir,
        rows=rows,
        summary_payload={
            "generated_at": generated_at,
            "summaries": analysis["summaries"],
            "comparisons": analysis["comparisons"],
            "gates": analysis["gates"],
            "route_and_cache_metrics": routes,
        },
        manifest={
            "generated_at": generated_at,
            "configuration": configuration,
            "benchmark_sha256": validation["sha256"],
            "secrets_recorded": False,
        },
    )
    print(json.dumps(analysis["gates"], ensure_ascii=False, indent=2))
    print(f"Artifacts: {args.output_dir}")


if __name__ == "__main__":
    main()
