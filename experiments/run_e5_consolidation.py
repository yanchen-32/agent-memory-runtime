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

from agent import OpenAICompatibleClient, RuleBasedClient
from benchmark import (
    collect_environment,
    load_jsonl,
    paired_bootstrap_comparison,
    write_formal_artifacts,
)
from benchmark.runner import run_benchmark
from memory import (
    AdaptiveConsolidationPolicy,
    HashEmbeddingModel,
    SentenceTransformerEmbedder,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run E5 Fixed vs Adaptive Consolidation with frozen settings."
    )
    parser.add_argument(
        "--benchmark",
        type=Path,
        default=ROOT / "benchmark" / "data" / "consolidation_v0.1.jsonl",
    )
    parser.add_argument("--output-dir", type=Path, default=ROOT / "results" / "formal" / "E5_consolidation")
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--token-budget", type=int, default=None)
    parser.add_argument("--protocol-version", default="1.1")
    parser.add_argument("--bootstrap-samples", type=int, default=10_000)
    parser.add_argument("--client", choices=("rule", "openai"), default="rule")
    parser.add_argument("--model", default=os.getenv("LLM_MODEL"))
    parser.add_argument("--base-url", default=os.getenv("LLM_BASE_URL", "https://api.deepseek.com"))
    parser.add_argument("--api-key", default=os.getenv("LLM_API_KEY", "EMPTY"))
    parser.add_argument("--timeout", type=int, default=60)
    parser.add_argument("--trace", action="store_true", help="Record full mechanistic trace events.")
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top-p", type=float, default=1.0)
    parser.add_argument("--max-tokens", type=int, default=None)
    parser.add_argument("--thinking", choices=("disabled", "enabled", "omit"), default="disabled")
    parser.add_argument("--embedding", choices=("hash", "sentence-transformers"), default="hash")
    parser.add_argument("--embedding-model", default="BAAI/bge-small-zh-v1.5")
    parser.add_argument("--adaptive-trigger-threshold", type=float, default=0.35)
    parser.add_argument("--adaptive-conflict-threshold", type=float, default=0.5)
    parser.add_argument("--adaptive-target-cluster-size", type=int, default=4)
    parser.add_argument("--adaptive-age-horizon-days", type=float, default=30.0)
    parser.add_argument("--adaptive-storage-token-capacity", type=int, default=100_000)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.repeats < 3:
        raise ValueError("Formal E5 experiments require --repeats >= 3")
    cases = load_jsonl(args.benchmark)
    if any(case.category != "consolidation" for case in cases):
        raise ValueError("E5 benchmark must contain only consolidation cases")

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
        )

    def embedder_factory():
        if args.embedding == "hash":
            return HashEmbeddingModel(dim=384)
        return SentenceTransformerEmbedder(args.embedding_model)

    policy = AdaptiveConsolidationPolicy(
        trigger_threshold=args.adaptive_trigger_threshold,
        conflict_threshold=args.adaptive_conflict_threshold,
        target_cluster_size=args.adaptive_target_cluster_size,
        age_horizon_days=args.adaptive_age_horizon_days,
        storage_token_capacity=args.adaptive_storage_token_capacity,
    )
    fixed_rows, fixed_summary = run_benchmark(
        cases,
        ("Ours",),
        client_factory,
        embedder_factory,
        top_k=args.top_k,
        token_budget=args.token_budget,
        repeats=args.repeats,
        consolidation_strategy="fixed",
        trace_enabled=args.trace,
    )
    adaptive_rows, adaptive_summary = run_benchmark(
        cases,
        ("Ours",),
        client_factory,
        embedder_factory,
        top_k=args.top_k,
        token_budget=args.token_budget,
        repeats=args.repeats,
        consolidation_strategy="adaptive",
        consolidation_policy=policy,
        trace_enabled=args.trace,
    )
    comparison_rows = [
        {**row, "agent": "Fixed"} for row in fixed_rows
    ] + [
        {**row, "agent": "Adaptive"} for row in adaptive_rows
    ]
    comparison = paired_bootstrap_comparison(
        comparison_rows,
        baseline="Fixed",
        treatment="Adaptive",
        samples=args.bootstrap_samples,
    )

    generated_at = datetime.now(timezone.utc).isoformat()
    environment = collect_environment(ROOT, args.benchmark)
    safe_config = {
        "protocol_version": args.protocol_version,
        "benchmark": str(args.benchmark.resolve()),
        "repeats": args.repeats,
        "top_k": args.top_k,
        "client": args.client,
        "model": args.model if args.client == "openai" else "RuleBasedClient",
        "base_url": args.base_url if args.client == "openai" else None,
        "temperature": args.temperature,
        "top_p": args.top_p,
        "max_tokens": args.max_tokens,
        "thinking": args.thinking,
        "embedding": args.embedding,
        "embedding_model": args.embedding_model if args.embedding == "sentence-transformers" else "HashEmbeddingModel(dim=384)",
        "adaptive_policy_version": policy.version,
        "adaptive_trigger_threshold": policy.trigger_threshold,
        "adaptive_conflict_threshold": policy.conflict_threshold,
        "adaptive_target_cluster_size": policy.target_cluster_size,
        "adaptive_age_horizon_days": policy.age_horizon_days,
        "adaptive_storage_token_capacity": policy.storage_token_capacity,
        "adaptive_trigger_weights": policy.trigger_weights,
        "adaptive_granularity_weights": policy.granularity_weights,
        "trace_enabled": args.trace,
    }
    payload = {
        "experiment": "E5",
        "generated_at": generated_at,
        "protocol_version": args.protocol_version,
        "configuration": safe_config,
        "environment": environment,
        "fixed": {"rows": fixed_rows, "summary": fixed_summary},
        "adaptive": {"rows": adaptive_rows, "summary": adaptive_summary},
        "answer_f1_comparison": comparison,
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "e5_consolidation_v0.1.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    write_formal_artifacts(
        args.output_dir,
        rows=comparison_rows,
        summary_payload={
            "protocol_version": args.protocol_version,
            "generated_at": generated_at,
            "experiments": {"E5_fixed": fixed_summary, "E5_adaptive": adaptive_summary},
            "comparisons": {"answer_f1_fixed_vs_adaptive": comparison},
        },
        manifest={
            "protocol_version": args.protocol_version,
            "generated_at": generated_at,
            "configuration": safe_config,
            "environment": environment,
            "secrets_recorded": False,
        },
    )
    print(
        f"E5 cases={len(cases)}, repeats={args.repeats}, "
        f"fixed_rows={len(fixed_rows)}, adaptive_rows={len(adaptive_rows)}"
    )
    print(f"Answer F1 delta Adaptive-Fixed: {comparison['mean_delta']}")
    print(f"Artifacts: {args.output_dir}")


if __name__ == "__main__":
    main()
