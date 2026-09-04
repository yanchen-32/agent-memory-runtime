from __future__ import annotations

# Direct execution adds the repository root before project imports.
# ruff: noqa: E402

import argparse
from collections import defaultdict
from datetime import datetime, timezone
import json
import os
from pathlib import Path
from statistics import mean, median
import sys


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from agent import ANSWER_FORMAT_VERSION, OpenAICompatibleClient, RuleBasedClient
from benchmark import (
    LEGACY_ANSWER_SCORER_VERSION,
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
from benchmark.generate_e3_v13 import STRATA
from benchmark.runner import AGENT_NAMES, run_benchmark, summarize
from benchmark.validate_e3_v13 import validate_development


CONTEXT_STABILITY_TOLERANCE = 0.10


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run v1.3 E3 long-context scaling by frozen B1 prompt-token stratum."
    )
    parser.add_argument(
        "--benchmark",
        type=Path,
        default=ROOT / "benchmark" / "data" / "v1.3-e3" / "development.jsonl",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=ROOT / "results" / "formal" / "e3_v13_development",
    )
    parser.add_argument("--agents", default="B0,B1,B2,B3,Ours")
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--token-budget", type=int, default=None)
    parser.add_argument("--protocol-version", default="1.3-e3")
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
    parser.add_argument(
        "--allow-unreviewed-benchmark",
        action="store_true",
        help="Candidate Development pilot only; formal admission remains ineligible.",
    )
    parser.add_argument("--trace", action="store_true")
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top-p", type=float, default=1.0)
    parser.add_argument("--max-tokens", type=int, default=64)
    parser.add_argument("--thinking", choices=("disabled", "enabled", "omit"), default="disabled")
    parser.add_argument("--embedding", choices=("hash", "sentence-transformers"), default="hash")
    parser.add_argument("--embedding-model", default="BAAI/bge-small-zh-v1.5")
    parser.add_argument(
        "--context-stability-tolerance",
        type=float,
        default=CONTEXT_STABILITY_TOLERANCE,
        help="Maximum relative spread of Ours case-median context tokens across strata.",
    )
    return parser.parse_args()


def _enrich_row(row: dict, metadata_by_case: dict[str, dict]) -> dict:
    metadata = metadata_by_case[str(row["case_id"])]
    return {
        **row,
        "e3_scenario_family": metadata["scenario_family"],
        "e3_stratum": metadata["stratum"],
        "e3_target_b1_prompt_tokens": metadata["target_b1_prompt_tokens"],
        "e3_generated_b1_prompt_tokens": metadata["b1_prompt_tokens"],
        "e3_history_memory_count": metadata["history_memory_count"],
    }


def _case_median_means(rows: list[dict], agent: str, metric: str) -> dict[str, float | None]:
    grouped: dict[tuple[str, str], list[float]] = defaultdict(list)
    for row in rows:
        value = row.get(metric)
        stratum = row.get("e3_stratum")
        if row.get("agent") == agent and value is not None and stratum in STRATA:
            grouped[(str(stratum), str(row["case_id"]))].append(float(value))
    values_by_stratum: dict[str, list[float]] = defaultdict(list)
    for (stratum, _), values in grouped.items():
        values_by_stratum[stratum].append(median(values))
    return {
        stratum: mean(values_by_stratum[stratum]) if values_by_stratum[stratum] else None
        for stratum in STRATA
    }


def build_e3_analysis(
    rows: list[dict],
    *,
    bootstrap_samples: int = 10_000,
    context_stability_tolerance: float = CONTEXT_STABILITY_TOLERANCE,
    benchmark_review_enforced: bool = True,
    expected_case_ids: set[str] | None = None,
    expected_agents: set[str] | None = None,
    expected_repeats: int | None = None,
    treatment_agent: str = "Ours",
) -> dict:
    """Build per-stratum statistics and the predeclared E3 admission gates."""
    if context_stability_tolerance < 0:
        raise ValueError("context stability tolerance must be non-negative")
    comparison_label = f"B1_vs_{treatment_agent}"
    summaries = {}
    comparisons = {}
    stratum_failure_counts = {}
    for stratum in STRATA:
        stratum_rows = [row for row in rows if row.get("e3_stratum") == stratum]
        summaries[stratum] = summarize(stratum_rows)
        failure_count = sum(row.get("status") == "failed" for row in stratum_rows)
        stratum_failure_counts[stratum] = failure_count
        f1 = paired_bootstrap_comparison(
            stratum_rows,
            baseline="B1",
            treatment=treatment_agent,
            metric="answer_f1",
            samples=bootstrap_samples,
        )
        latency = paired_latency_reduction(
            stratum_rows, baseline="B1", treatment=treatment_agent
        )
        for comparison in (f1, latency):
            comparison["formal_comparison_valid"] = (
                failure_count == 0 and benchmark_review_enforced
            )
            comparison["failure_count"] = failure_count
            if failure_count:
                comparison["invalid_reason"] = "terminal run failures present"
            elif not benchmark_review_enforced:
                comparison["invalid_reason"] = "unreviewed Development candidate"
        comparisons[f"{stratum}_answer_f1_{comparison_label}"] = f1
        comparisons[f"{stratum}_e2e_latency_{comparison_label}"] = latency

    overall_f1 = paired_bootstrap_comparison(
        rows,
        baseline="B1",
        treatment=treatment_agent,
        metric="answer_f1",
        samples=bootstrap_samples,
    )
    total_failures = sum(row.get("status") == "failed" for row in rows)
    overall_f1["formal_comparison_valid"] = total_failures == 0 and benchmark_review_enforced
    overall_f1["failure_count"] = total_failures
    if total_failures:
        overall_f1["invalid_reason"] = "terminal run failures present"
    elif not benchmark_review_enforced:
        overall_f1["invalid_reason"] = "unreviewed Development candidate"
    comparisons[f"overall_answer_f1_{comparison_label}"] = overall_f1

    ours_context = _case_median_means(rows, treatment_agent, "context_tokens")
    defined_context = [value for value in ours_context.values() if value is not None]
    if len(defined_context) == len(STRATA) and min(defined_context) > 0:
        context_relative_spread = (max(defined_context) - min(defined_context)) / min(defined_context)
        context_stable = context_relative_spread <= context_stability_tolerance
    else:
        context_relative_spread = None
        context_stable = False

    ours_rows = [row for row in rows if row.get("agent") == treatment_agent]
    ours_failures = sum(row.get("status") == "failed" for row in ours_rows)
    forbidden_count = sum(
        int(row.get("forbidden_retrieved_count") or 0)
        for row in ours_rows
        if row.get("status", "succeeded") == "succeeded"
    )
    forbidden_zero = bool(ours_rows) and ours_failures == 0 and forbidden_count == 0

    f1_gate = all(
        comparisons[f"{stratum}_answer_f1_{comparison_label}"].get(
            "statistically_supported_non_decrease"
        ) is True
        and stratum_failure_counts[stratum] == 0
        for stratum in STRATA
    )
    very_long_latency = comparisons[f"very_long_e2e_latency_{comparison_label}"]
    very_long_latency_gate = (
        very_long_latency.get("target_met") is True
        and stratum_failure_counts["very_long"] == 0
    )
    basic_data_complete = (
        all(summaries[stratum] for stratum in STRATA)
        and total_failures == 0
        and bool(ours_rows)
    )
    observed_keys = [
        (str(row["case_id"]), str(row["agent"]), int(row.get("repeat", 1)))
        for row in rows
    ]
    duplicate_run_keys = len(observed_keys) - len(set(observed_keys))
    expected_row_count = None
    coverage_complete = basic_data_complete and duplicate_run_keys == 0
    if (
        expected_case_ids is not None
        and expected_agents is not None
        and expected_repeats is not None
    ):
        expected_keys = {
            (case_id, agent, repeat)
            for case_id in expected_case_ids
            for agent in expected_agents
            for repeat in range(1, expected_repeats + 1)
        }
        expected_row_count = len(expected_keys)
        coverage_complete = coverage_complete and set(observed_keys) == expected_keys
    evidence_eligible = benchmark_review_enforced and coverage_complete
    gates = {
        "answer_f1_non_decrease_every_stratum": {
            "passed": f1_gate,
            "criterion": "CI95(delta F1 Ours-B1).lower >= 0 in every stratum",
        },
        "ours_context_tokens_stable": {
            "passed": context_stable,
            "criterion": f"relative spread of stratum case-median means <= {context_stability_tolerance:.2%}",
            "case_median_mean_context_tokens": ours_context,
            "relative_spread": context_relative_spread,
        },
        "ours_forbidden_retrieval_zero": {
            "passed": forbidden_zero,
            "criterion": "zero forbidden retrievals and zero terminal Ours failures",
            "forbidden_retrieved_count": forbidden_count,
            "ours_terminal_failures": ours_failures,
        },
        "very_long_e2e_latency_reduction_50pct": {
            "passed": very_long_latency_gate,
            "criterion": "paired Very Long E2E latency reduction >= 50%",
            "latency_reduction": very_long_latency.get("latency_reduction"),
        },
        "formal_evidence_eligible": {
            "passed": evidence_eligible,
            "criterion": "reviewed/frozen benchmark coverage with no terminal failures",
            "coverage_complete": coverage_complete,
            "expected_rows": expected_row_count,
            "observed_rows": len(rows),
            "duplicate_run_keys": duplicate_run_keys,
        },
    }
    gates["all_admission_gates"] = {
        "passed": evidence_eligible
        and f1_gate
        and context_stable
        and forbidden_zero
        and very_long_latency_gate,
        "latency_claim_authorized": evidence_eligible and very_long_latency_gate,
    }
    return {
        "strata": STRATA,
        "summaries": summaries,
        "comparisons": comparisons,
        "gates": gates,
        "terminal_failures": total_failures,
        "coverage_complete": coverage_complete,
    }


def main() -> None:
    args = parse_args()
    if args.repeats < 3:
        raise ValueError("Formal E3 experiments require --repeats >= 3")
    if args.context_stability_tolerance < 0:
        raise ValueError("--context-stability-tolerance must be non-negative")
    validation = validate_development(args.benchmark)
    if not args.allow_unreviewed_benchmark:
        require_frozen_benchmark(args.benchmark)

    agent_names = [name.strip() for name in args.agents.split(",") if name.strip()]
    unknown = sorted(set(agent_names) - set(AGENT_NAMES))
    if unknown:
        raise ValueError(f"unknown agents: {unknown}")
    if not {"B1", "Ours"} <= set(agent_names):
        raise ValueError("E3 admission requires both B1 and Ours")
    if args.output_dir.exists() and any(args.output_dir.iterdir()) and not args.resume:
        raise ValueError("output directory is not empty; use a new directory or exact --resume")

    cases = load_jsonl(args.benchmark)
    metadata_by_case = {case.case_id: case.metadata for case in cases}

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
    safe_config = {
        "protocol_version": args.protocol_version,
        "benchmark": str(args.benchmark.resolve()),
        "benchmark_sha256": sha256_file(args.benchmark),
        "benchmark_review_enforced": not args.allow_unreviewed_benchmark,
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
            args.embedding_model
            if args.embedding == "sentence-transformers"
            else "HashEmbeddingModel(dim=384)"
        ),
        "execution_policy": "query_interleaved_alternating",
        "trace_enabled": args.trace,
        "bootstrap_samples": args.bootstrap_samples,
        "bootstrap_seed": 202601,
        "answer_format_version": ANSWER_FORMAT_VERSION,
        "answer_scorer_versions": [LEGACY_ANSWER_SCORER_VERSION],
        "strata": STRATA,
        "context_stability_tolerance": args.context_stability_tolerance,
    }
    checkpoint = JsonlRunCheckpoint(
        args.output_dir / "run_checkpoint.jsonl",
        safe_config,
    )
    existing_rows = checkpoint.prepare(resume=args.resume, retry_failed=True)

    def append_enriched(row: dict) -> None:
        checkpoint.append(_enrich_row(row, metadata_by_case))

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
        on_row=append_enriched,
        continue_on_error=args.continue_on_error,
    )
    rows = [_enrich_row(row, metadata_by_case) for row in rows]
    analysis = build_e3_analysis(
        rows,
        bootstrap_samples=args.bootstrap_samples,
        context_stability_tolerance=args.context_stability_tolerance,
        benchmark_review_enforced=not args.allow_unreviewed_benchmark,
        expected_case_ids={case.case_id for case in cases},
        expected_agents=set(agent_names),
        expected_repeats=args.repeats,
    )

    generated_at = datetime.now(timezone.utc).isoformat()
    environment = collect_environment(ROOT, args.benchmark)
    payload = {
        "experiment": "E3",
        "protocol_version": args.protocol_version,
        "generated_at": generated_at,
        "benchmark_validation": validation,
        "configuration": safe_config,
        "environment": environment,
        "rows": rows,
        **analysis,
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "e3_long_context_scaling_development.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    write_formal_artifacts(
        args.output_dir,
        rows=rows,
        summary_payload={
            "protocol_version": args.protocol_version,
            "generated_at": generated_at,
            "experiments": {
                f"E3_{stratum}": analysis["summaries"][stratum]
                for stratum in STRATA
            },
            "comparisons": analysis["comparisons"],
            "gates": analysis["gates"],
        },
        manifest={
            "protocol_version": args.protocol_version,
            "generated_at": generated_at,
            "configuration": safe_config,
            "environment": environment,
            "benchmark_validation_sha256": validation["sha256"],
            "secrets_recorded": False,
        },
    )

    print(
        f"E3 cases={len(cases)}, rows={len(rows)}, agents={', '.join(agent_names)}, "
        f"repeats={args.repeats}"
    )
    for stratum in STRATA:
        f1 = analysis["comparisons"][f"{stratum}_answer_f1_B1_vs_Ours"]
        latency = analysis["comparisons"][f"{stratum}_e2e_latency_B1_vs_Ours"]
        ours = next(
            (item for item in analysis["summaries"][stratum] if item["agent"] == "Ours"),
            None,
        )
        print(
            f"  {stratum}: Ours F1={ours['avg_answer_f1'] if ours else 'NA'}, "
            f"delta_vs_B1={f1['mean_delta']}, CI95={f1['ci95']}, "
            f"context_tokens={ours['avg_context_tokens'] if ours else 'NA'}, "
            f"latency_reduction={latency.get('latency_reduction')}"
        )
    print(json.dumps(analysis["gates"], ensure_ascii=False, indent=2))
    print(f"Artifacts: {args.output_dir}")


if __name__ == "__main__":
    main()
