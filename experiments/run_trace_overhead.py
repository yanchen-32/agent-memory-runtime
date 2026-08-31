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
    trace_overhead_summary,
    write_formal_artifacts,
)
from benchmark.runner import run_case
from memory import HashEmbeddingModel, SentenceTransformerEmbedder


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Measure full Memory Observatory trace overhead.")
    parser.add_argument("--benchmark", type=Path, default=ROOT / "benchmark" / "data" / "benchmark_v0.2.jsonl")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "results" / "formal" / "trace_overhead")
    parser.add_argument("--repeats", type=int, default=30)
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--token-budget", type=int, default=None)
    parser.add_argument("--client", choices=("rule", "openai"), default="rule")
    parser.add_argument("--model", default=os.getenv("LLM_MODEL"))
    parser.add_argument("--base-url", default=os.getenv("LLM_BASE_URL", "https://api.deepseek.com"))
    parser.add_argument("--api-key", default=os.getenv("LLM_API_KEY", "EMPTY"))
    parser.add_argument("--timeout", type=int, default=60)
    parser.add_argument("--embedding", choices=("hash", "sentence-transformers"), default="hash")
    parser.add_argument("--embedding-model", default="BAAI/bge-small-zh-v1.5")
    parser.add_argument("--protocol-version", default="1.1")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.repeats < 30:
        raise ValueError("Trace overhead microbenchmark requires --repeats >= 30")
    cases = load_jsonl(args.benchmark)

    def client_factory():
        if args.client == "rule":
            return RuleBasedClient()
        return OpenAICompatibleClient(
            model=args.model,
            base_url=args.base_url,
            api_key=args.api_key,
            timeout=args.timeout,
            thinking="disabled",
        )

    def embedder_factory():
        if args.embedding == "hash":
            return HashEmbeddingModel(dim=384)
        return SentenceTransformerEmbedder(args.embedding_model)

    rows = []
    sequence = 0
    for repeat in range(1, args.repeats + 1):
        for case_index, case in enumerate(cases):
            conditions = (False, True) if (case_index + repeat) % 2 else (True, False)
            for trace_enabled in conditions:
                row = run_case(
                    "Ours",
                    case,
                    client_factory,
                    embedder_factory,
                    top_k=args.top_k,
                    token_budget=args.token_budget,
                    repeat=repeat,
                    trace_enabled=trace_enabled,
                )
                sequence += 1
                row["execution_sequence"] = sequence
                row["execution_policy"] = "trace_condition_interleaved_alternating"
                rows.append(row)

    overhead = trace_overhead_summary(rows)
    generated_at = datetime.now(timezone.utc).isoformat()
    environment = collect_environment(ROOT, args.benchmark)
    config = {
        "protocol_version": args.protocol_version,
        "benchmark": str(args.benchmark.resolve()),
        "repeats": args.repeats,
        "top_k": args.top_k,
        "client": args.client,
        "model": args.model if args.client == "openai" else "RuleBasedClient",
        "embedding": args.embedding,
        "embedding_model": args.embedding_model if args.embedding == "sentence-transformers" else "HashEmbeddingModel(dim=384)",
        "execution_policy": "trace_condition_interleaved_alternating",
    }
    payload = {
        "experiment": "Memory Observatory Trace Overhead",
        "generated_at": generated_at,
        "configuration": config,
        "environment": environment,
        "overhead": overhead,
        "rows": rows,
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "trace_overhead.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    write_formal_artifacts(
        args.output_dir,
        rows,
        summary_payload={
            "protocol_version": args.protocol_version,
            "generated_at": generated_at,
            "experiments": {},
            "comparisons": {"trace_overhead": overhead},
        },
        manifest={
            "protocol_version": args.protocol_version,
            "generated_at": generated_at,
            "configuration": config,
            "environment": environment,
            "secrets_recorded": False,
        },
    )
    print(f"paired_cases={overhead['num_paired_cases']}")
    print(f"trace_overhead={overhead['trace_overhead']}")
    print(f"artifacts={args.output_dir}")


if __name__ == "__main__":
    main()
