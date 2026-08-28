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
from benchmark import load_jsonl
from benchmark.runner import AGENT_NAMES, run_benchmark, summarize
from memory import HashEmbeddingModel, SentenceTransformerEmbedder


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
    parser.add_argument("--client", choices=("rule", "openai"), default="rule")
    parser.add_argument("--model", default=os.getenv("LLM_MODEL"))
    parser.add_argument(
        "--base-url",
        default=os.getenv("LLM_BASE_URL", "http://localhost:8000/v1"),
    )
    parser.add_argument("--api-key", default=os.getenv("LLM_API_KEY", "EMPTY"))
    parser.add_argument("--timeout", type=int, default=60)
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
        )

    def embedder_factory():
        if args.embedding == "hash":
            return HashEmbeddingModel(dim=384)
        return SentenceTransformerEmbedder(args.embedding_model)

    rows, _ = run_benchmark(
        cases=cases,
        agent_names=agent_names,
        client_factory=client_factory,
        embedder_factory=embedder_factory,
        top_k=args.top_k,
        token_budget=args.token_budget,
        repeats=args.repeats,
    )

    e1_rows = [row for row in rows if row["category"] in E1_CATEGORIES]
    e2_rows = [row for row in rows if row["category"] in E2_CATEGORIES]
    generated_at = datetime.now(timezone.utc).isoformat()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    outputs = [
        (
            "e1_long_term_accuracy_v0.2.json",
            "E1",
            E1_CATEGORIES,
            e1_rows,
        ),
        (
            "e2_update_conflict_v0.2.json",
            "E2",
            E2_CATEGORIES,
            e2_rows,
        ),
    ]
    for filename, experiment, categories, experiment_rows in outputs:
        payload = {
            "experiment": experiment,
            "generated_at": generated_at,
            "benchmark": str(args.benchmark),
            "agents": agent_names,
            "repeats": args.repeats,
            "categories": sorted(categories),
            "rows": experiment_rows,
            "summary": summarize(experiment_rows),
        }
        (args.output_dir / filename).write_text(
            json.dumps(payload, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

    print(
        f"E1 rows: {len(e1_rows)}, E2 rows: {len(e2_rows)}, "
        f"agents: {', '.join(agent_names)}, repeats: {args.repeats}"
    )
    for label, experiment_rows in (("E1", e1_rows), ("E2", e2_rows)):
        print(f"{label} summary:")
        for item in summarize(experiment_rows):
            print(
                f"  {item['agent']}: accuracy={item['accuracy']:.3f} "
                f"+/- {item['accuracy_std']:.3f}, "
                f"latency_mean={item['latency_mean_ms']:.4f} ms, "
                f"p50={item['latency_p50_ms']:.4f} ms, "
                f"p95={item['latency_p95_ms']:.4f} ms"
            )


if __name__ == "__main__":
    main()
