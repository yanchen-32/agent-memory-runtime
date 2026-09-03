from __future__ import annotations

# The executable supports direct invocation from ``experiments/`` and must add
# the repository root before importing project packages.
# ruff: noqa: E402

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
from statistics import mean
import sys
import time


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from agent import ANSWER_FORMAT_VERSION, OpenAICompatibleClient, RuleBasedClient
from agent.e3_ablation import (
    E3_ABLATION_PROMPT_VERSION,
    E3_ABLATION_SPECS,
    E3AblationAgent,
)
from benchmark import (
    JsonlRunCheckpoint,
    ReusableEmbeddingFactory,
    answer_metrics,
    collect_environment,
    estimate_tokens,
    load_jsonl,
    paired_bootstrap_comparison,
    require_frozen_benchmark,
    retrieval_metrics,
    sha256_file,
    write_formal_artifacts,
)
from benchmark.generate_e3_v13 import STRATA
from memory import MemoryRuntimeV1


DEFAULT_VARIANTS = tuple(E3_ABLATION_SPECS)
CHAIN_FIELDS = (
    "canonical_current_fact",
    "version_time_context",
    "exact_spo_filter",
    "identity_title",
)


def validate_ablation_chain(variants: list[str] | tuple[str, ...]) -> None:
    unknown = sorted(set(variants) - set(E3_ABLATION_SPECS))
    if unknown:
        raise ValueError(f"unknown E3 ablation variants: {unknown}")
    if len(variants) != len(set(variants)):
        raise ValueError("E3 ablation variants must be unique")
    canonical_order = [name for name in DEFAULT_VARIANTS if name in variants]
    if list(variants) != canonical_order:
        raise ValueError(f"E3 ablations must follow canonical order: {canonical_order}")
    for baseline, treatment in zip(variants, variants[1:]):
        left = E3_ABLATION_SPECS[baseline]
        right = E3_ABLATION_SPECS[treatment]
        changed = [field for field in CHAIN_FIELDS if getattr(left, field) != getattr(right, field)]
        if len(changed) != 1 or changed[0] != right.changed_variable:
            raise ValueError(
                f"{baseline}->{treatment} is not a declared single-variable transition"
            )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run cumulative, single-variable v1.3-E3 Development ablations."
    )
    parser.add_argument(
        "--benchmark",
        type=Path,
        default=ROOT / "benchmark/data/v1.3-e3/development.jsonl",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=ROOT / "results/development/e3_v13_ablation_a0_a4",
    )
    parser.add_argument("--variants", default=",".join(DEFAULT_VARIANTS))
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
    parser.add_argument(
        "--continue-on-error",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top-p", type=float, default=1.0)
    parser.add_argument("--max-tokens", type=int, default=64)
    parser.add_argument("--thinking", choices=("disabled", "enabled", "omit"), default="disabled")
    parser.add_argument("--embedding", choices=("hash", "sentence-transformers"), default="hash")
    parser.add_argument("--embedding-model", default="BAAI/bge-small-zh-v1.5")
    return parser.parse_args()


def _sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _mapped_ids(agent: E3AblationAgent, case) -> list[str]:
    content_to_id = {
        str(turn["content"]): str(turn["memory_id"])
        for turn in case.conversation
    }
    return [
        content_to_id[content]
        for content in agent.last_retrieved_contents
        if content in content_to_id
    ]


def run_ablation_case(
    variant: str,
    case,
    repeat: int,
    client_factory,
    embedder_factory,
    *,
    top_k: int,
) -> dict:
    client = client_factory()
    setup_started = time.perf_counter()
    runtime = MemoryRuntimeV1(embedder=embedder_factory())
    agent = E3AblationAgent(client, runtime, variant=variant, top_k=top_k)
    agent.ingest(
        case.conversation,
        user_id="benchmark",
        session_id=f"{case.case_id}-{variant}-repeat-{repeat}",
    )
    setup_latency_ms = (time.perf_counter() - setup_started) * 1000
    started = time.perf_counter()
    prediction = agent.answer(
        case.query,
        query_time=case.memory_query_time,
        query_id=case.case_id,
    )
    end_to_end_latency_ms = (time.perf_counter() - started) * 1000
    measured = (
        agent.last_memory_latency_ms
        + agent.last_context_latency_ms
        + agent.last_llm_latency_ms
    )
    scored = answer_metrics(
        prediction,
        case.expected_answer,
        case.answer_aliases,
        case.answer_spec,
    )
    ranked_ids = _mapped_ids(agent, case)
    retrieval = retrieval_metrics(ranked_ids, case.expected_memory_ids)
    usage = dict(getattr(client, "last_usage", {}) or {})
    spec = E3_ABLATION_SPECS[variant]
    return {
        "status": "succeeded",
        "agent": variant,
        "variant": variant,
        "changed_variable": spec.changed_variable,
        "case_id": case.case_id,
        "repeat": repeat,
        "e3_scenario_family": case.metadata["scenario_family"],
        "e3_stratum": case.metadata["stratum"],
        "e3_predicate": case.metadata["predicate"],
        "e3_query_template_index": case.metadata["query_template_index"],
        "e3_target_position_band": case.metadata["target_position_band"],
        "query": case.query,
        "expected_answer": case.expected_answer,
        "prediction": prediction,
        "correct": int(scored["answer_accuracy"]),
        "answer_accuracy": int(scored["answer_accuracy"]),
        "exact_match": int(scored["exact_match"]),
        "semantic_answer_accuracy": int(scored["semantic_answer_accuracy"]),
        "answer_f1": scored["answer_f1"],
        "answer_scorer_version": scored["answer_scorer_version"],
        "retrieved_memory_ids": ranked_ids,
        "expected_memory_ids": case.expected_memory_ids,
        "forbidden_memory_ids": case.forbidden_memory_ids,
        "forbidden_retrieved_count": len(
            set(case.forbidden_memory_ids) & set(ranked_ids)
        ),
        "context_tokens": estimate_tokens(agent.last_context),
        "prompt_tokens": estimate_tokens(agent.last_prompt),
        "context_sha256": _sha256_text(agent.last_context),
        "prompt_sha256": _sha256_text(agent.last_prompt),
        "setup_latency_ms": round(setup_latency_ms, 4),
        "memory_latency_ms": round(agent.last_memory_latency_ms, 4),
        "context_build_latency_ms": round(agent.last_context_latency_ms, 4),
        "llm_latency_ms": round(agent.last_llm_latency_ms, 4),
        "post_latency_ms": round(max(0.0, end_to_end_latency_ms - measured), 4),
        "end_to_end_latency_ms": round(end_to_end_latency_ms, 4),
        "api_prompt_tokens": usage.get("prompt_tokens"),
        "api_completion_tokens": usage.get("completion_tokens"),
        "api_total_tokens": usage.get("total_tokens"),
        "api_attempts": getattr(client, "last_attempts", None),
        "spo_filtered_count": agent.last_spo_filtered_count,
        **retrieval,
    }


def _failure_row(variant: str, case, repeat: int, exc: Exception) -> dict:
    return {
        "status": "failed",
        "agent": variant,
        "variant": variant,
        "changed_variable": E3_ABLATION_SPECS[variant].changed_variable,
        "case_id": case.case_id,
        "repeat": repeat,
        "e3_scenario_family": case.metadata["scenario_family"],
        "e3_stratum": case.metadata["stratum"],
        "e3_predicate": case.metadata["predicate"],
        "e3_query_template_index": case.metadata["query_template_index"],
        "e3_target_position_band": case.metadata["target_position_band"],
        "answer_f1": None,
        "correct": None,
        "retrieved_memory_ids": [],
        "forbidden_retrieved_count": None,
        "error_type": type(exc).__name__,
        "error_message": str(exc)[:500],
        "error_status_code": getattr(exc, "status_code", None),
        "api_attempts": getattr(exc, "attempts", None),
    }


def _variant_summary(rows: list[dict], variant: str) -> dict:
    selected = [row for row in rows if row.get("variant") == variant]
    succeeded = [row for row in selected if row.get("status") == "succeeded"]
    item = {
        "variant": variant,
        "changed_variable": E3_ABLATION_SPECS[variant].changed_variable,
        "runs": len(selected),
        "succeeded": len(succeeded),
        "failed": len(selected) - len(succeeded),
        "cases": len({row["case_id"] for row in selected}),
    }
    for metric in (
        "correct",
        "answer_f1",
        "context_tokens",
        "prompt_tokens",
        "memory_latency_ms",
        "context_build_latency_ms",
        "llm_latency_ms",
        "end_to_end_latency_ms",
        "forbidden_retrieved_count",
        "recall@5",
        "mrr",
    ):
        values = [float(row[metric]) for row in succeeded if row.get(metric) is not None]
        item[f"avg_{metric}"] = mean(values) if values else None
    item["unknown_rate"] = (
        mean(row.get("prediction") == "UNKNOWN" for row in succeeded)
        if succeeded
        else None
    )
    return item


def build_ablation_analysis(
    rows: list[dict],
    variants: list[str],
    *,
    bootstrap_samples: int = 10_000,
    expected_case_count: int | None = None,
    expected_repeats: int | None = None,
) -> dict:
    summaries = {}
    comparisons = {}
    for stratum in STRATA:
        subset = [row for row in rows if row.get("e3_stratum") == stratum]
        summaries[stratum] = [_variant_summary(subset, variant) for variant in variants]
        for baseline, treatment in zip(variants, variants[1:]):
            comparisons[f"{stratum}_{baseline}_vs_{treatment}_answer_f1"] = (
                paired_bootstrap_comparison(
                    subset,
                    baseline=baseline,
                    treatment=treatment,
                    metric="answer_f1",
                    samples=bootstrap_samples,
                )
            )
    for baseline, treatment in zip(variants, variants[1:]):
        comparisons[f"overall_{baseline}_vs_{treatment}_answer_f1"] = (
            paired_bootstrap_comparison(
                rows,
                baseline=baseline,
                treatment=treatment,
                metric="answer_f1",
                samples=bootstrap_samples,
            )
        )

    successful = [row for row in rows if row.get("status") == "succeeded"]
    by_key = {
        (str(row["case_id"]), int(row["repeat"]), str(row["variant"])): row
        for row in successful
    }
    a3_a4_pairs = []
    if {"A3", "A4"} <= set(variants):
        for case_id, repeat, variant in list(by_key):
            if variant != "A3":
                continue
            a3 = by_key[(case_id, repeat, "A3")]
            a4 = by_key.get((case_id, repeat, "A4"))
            if a4 is not None:
                a3_a4_pairs.append((a3, a4))
    context_identical = bool(a3_a4_pairs) and all(
        left["context_sha256"] == right["context_sha256"]
        and left["retrieved_memory_ids"] == right["retrieved_memory_ids"]
        for left, right in a3_a4_pairs
    )
    prompt_changed = bool(a3_a4_pairs) and all(
        left["prompt_sha256"] != right["prompt_sha256"]
        for left, right in a3_a4_pairs
    )

    observed_keys = {
        (str(row["case_id"]), str(row["variant"]), int(row["repeat"]))
        for row in rows
    }
    expected_rows = (
        expected_case_count * len(variants) * expected_repeats
        if expected_case_count is not None and expected_repeats is not None
        else None
    )
    complete = (
        expected_rows is not None
        and len(rows) == expected_rows
        and len(observed_keys) == expected_rows
        and all(row.get("status") == "succeeded" for row in rows)
    )
    return {
        "development_diagnostic_only": True,
        "formal_e3_admission_claim_authorized": False,
        "variants": {
            name: {
                field: getattr(E3_ABLATION_SPECS[name], field)
                for field in (*CHAIN_FIELDS, "changed_variable")
            }
            for name in variants
        },
        "summaries": summaries,
        "comparisons": comparisons,
        "integrity": {
            "complete": complete,
            "expected_rows": expected_rows,
            "observed_rows": len(rows),
            "terminal_failures": sum(row.get("status") == "failed" for row in rows),
            "a3_a4_paired_rows": len(a3_a4_pairs),
            "a3_a4_context_and_retrieval_identical": context_identical,
            "a3_a4_prompt_changed": prompt_changed,
        },
    }


def main() -> None:
    args = parse_args()
    variants = [name.strip() for name in args.variants.split(",") if name.strip()]
    validate_ablation_chain(variants)
    if args.repeats < 3:
        raise ValueError("E3 Development ablations require --repeats >= 3")
    require_frozen_benchmark(args.benchmark)
    if args.output_dir.exists() and any(args.output_dir.iterdir()) and not args.resume:
        raise ValueError("output directory is not empty; use a new directory or exact --resume")
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

    embedder_factory = ReusableEmbeddingFactory(args.embedding, args.embedding_model)
    configuration = {
        "experiment": "E3-Development-Ablation",
        "diagnostic_only": True,
        "benchmark": str(args.benchmark.resolve()),
        "benchmark_sha256": sha256_file(args.benchmark),
        "variants": variants,
        "variant_specs": {
            name: {
                field: getattr(E3_ABLATION_SPECS[name], field)
                for field in (*CHAIN_FIELDS, "changed_variable")
            }
            for name in variants
        },
        "repeats": args.repeats,
        "top_k": args.top_k,
        "client": args.client,
        "model": args.model if args.client == "openai" else "RuleBasedClient",
        "base_url": args.base_url if args.client == "openai" else None,
        "temperature": args.temperature,
        "top_p": args.top_p,
        "max_tokens": args.max_tokens,
        "thinking": args.thinking,
        "max_retries": args.max_retries,
        "retry_backoff_seconds": args.retry_backoff_seconds,
        "embedding": args.embedding,
        "embedding_model": args.embedding_model,
        "answer_format_version": ANSWER_FORMAT_VERSION,
        "ablation_prompt_version": E3_ABLATION_PROMPT_VERSION,
        "execution_policy": "query_interleaved_alternating_variants",
        "bootstrap_samples": args.bootstrap_samples,
    }
    checkpoint = JsonlRunCheckpoint(
        args.output_dir / "run_checkpoint.jsonl", configuration
    )
    rows = checkpoint.prepare(resume=args.resume, retry_failed=False)
    completed = {
        (str(row["case_id"]), str(row["variant"]), int(row["repeat"]))
        for row in rows
    }
    sequence = max((int(row.get("execution_sequence", 0)) for row in rows), default=0)
    for repeat in range(1, args.repeats + 1):
        for case_index, case in enumerate(cases):
            ordered = variants if (case_index + repeat - 1) % 2 == 0 else list(reversed(variants))
            for order_in_query, variant in enumerate(ordered, start=1):
                key = (case.case_id, variant, repeat)
                if key in completed:
                    continue
                try:
                    row = run_ablation_case(
                        variant,
                        case,
                        repeat,
                        client_factory,
                        embedder_factory,
                        top_k=args.top_k,
                    )
                except Exception as exc:
                    if not args.continue_on_error:
                        raise
                    row = _failure_row(variant, case, repeat, exc)
                sequence += 1
                row["execution_sequence"] = sequence
                row["execution_order_in_query"] = order_in_query
                row["execution_policy"] = "query_interleaved_alternating_variants"
                rows.append(row)
                checkpoint.append(row)
                completed.add(key)

    analysis = build_ablation_analysis(
        rows,
        variants,
        bootstrap_samples=args.bootstrap_samples,
        expected_case_count=len(cases),
        expected_repeats=args.repeats,
    )
    generated_at = datetime.now(timezone.utc).isoformat()
    environment = collect_environment(ROOT, args.benchmark)
    payload = {
        "generated_at": generated_at,
        "configuration": configuration,
        "environment": environment,
        "rows": rows,
        **analysis,
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "e3_development_ablation.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    write_formal_artifacts(
        args.output_dir,
        rows=rows,
        summary_payload={
            "generated_at": generated_at,
            "development_diagnostic_only": True,
            "experiments": {
                f"E3_ablation_{stratum}": analysis["summaries"][stratum]
                for stratum in STRATA
            },
            "comparisons": analysis["comparisons"],
            "integrity": analysis["integrity"],
            "formal_e3_admission_claim_authorized": False,
        },
        manifest={
            "generated_at": generated_at,
            "configuration": configuration,
            "environment": environment,
            "development_diagnostic_only": True,
            "formal_e3_admission_claim_authorized": False,
            "secrets_recorded": False,
        },
    )
    print(
        f"E3 Development ablation rows={len(rows)}, variants={','.join(variants)}, "
        f"failures={analysis['integrity']['terminal_failures']}"
    )
    for stratum in STRATA:
        values = analysis["summaries"][stratum]
        print(
            stratum
            + ": "
            + ", ".join(
                f"{item['variant']} F1={item['avg_answer_f1']:.4f}"
                for item in values
                if item["avg_answer_f1"] is not None
            )
        )
    print(json.dumps(analysis["integrity"], ensure_ascii=False, indent=2))
    print(f"Artifacts: {args.output_dir}")


if __name__ == "__main__":
    main()
