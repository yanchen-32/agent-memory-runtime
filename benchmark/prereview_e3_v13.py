"""Deterministic technical prereview for v1.3 E3 Development candidates."""

from __future__ import annotations

import argparse
import csv
from datetime import datetime
import json
from pathlib import Path

from .artifacts import sha256_file
from .generate_e3_v13 import BENCHMARK_VERSION, REVIEW_FIELDS, STRATA
from .loader import BenchmarkCase, load_jsonl
from .validate_e3_v13 import validate_development


PREREVIEW_VERSION = "v1.3-e3.1"
CHECK_FIELDS = REVIEW_FIELDS[3:-1]


def write_signoff_template(
    data_dir: str | Path,
    review_path: str | Path,
    *,
    deep_audit_path: str | Path | None = None,
) -> dict:
    root = Path(data_dir)
    review_file = Path(review_path)
    validation = validate_development(root / "development.jsonl")
    audit_file = Path(deep_audit_path) if deep_audit_path else None
    payload = {
        "benchmark_version": BENCHMARK_VERSION,
        "scope": "all_v13_e3_development_candidates",
        "reviewed_case_count": validation["cases"],
        "reviewed_scenario_family_count": validation["scenario_families"],
        "review_checklist_file": review_file.name,
        "review_checklist_sha256": sha256_file(review_file),
        "development_sha256": validation["sha256"],
        "deep_audit_file": audit_file.name if audit_file else None,
        "deep_audit_sha256": sha256_file(audit_file) if audit_file else None,
        "attestation": (
            "I reviewed the v1.3-E3 technical and deep audit for the bound "
            f"{validation['cases']}-case Development candidate and approve it "
            "for freezing."
        ),
        "reviewer": "",
        "decision": "pending_human_review",
    }
    (root / "review_signoff.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return payload


def _answer_is_evidenced(case: BenchmarkCase) -> bool:
    evidence = {
        str(turn["memory_id"]): str(turn["content"])
        for turn in case.conversation
    }
    return (
        len(case.expected_memory_ids) == 1
        and case.expected_answer in evidence[case.expected_memory_ids[0]]
    )


def _chronology_is_sound(case: BenchmarkCase) -> bool:
    points = [datetime.fromisoformat(str(turn["valid_from"])) for turn in case.conversation]
    query_time = datetime.fromisoformat(str(case.memory_query_time))
    return (
        all(left < right for left, right in zip(points, points[1:]))
        and query_time > points[-1]
    )


def prereview_candidates(data_dir: str | Path) -> dict:
    root = Path(data_dir)
    validation = validate_development(root / "development.jsonl")
    cases = load_jsonl(root / "development.jsonl")
    family_histories = {}
    for case in cases:
        family_histories.setdefault(str(case.metadata["scenario_family"]), {})[
            str(case.metadata["stratum"])
        ] = case.conversation

    prefix_checks = {}
    for family, by_stratum in family_histories.items():
        ordered = [by_stratum[stratum] for stratum in STRATA]
        prefix_checks[family] = all(
            upper[: len(lower)] == lower and len(upper) > len(lower)
            for lower, upper in zip(ordered, ordered[1:])
        )

    results = {}
    for case in cases:
        available = {str(turn["memory_id"]) for turn in case.conversation}
        answer_occurrences = [
            str(turn["memory_id"])
            for turn in case.conversation
            if case.expected_answer in str(turn["content"])
        ]
        target = int(case.metadata["target_b1_prompt_tokens"])
        measured = int(case.metadata["b1_prompt_tokens"])
        checks = {
            "question_clear": bool(case.query.strip()) and case.query.endswith("？"),
            "answer_correct": _answer_is_evidenced(case),
            "evidence_id_correct": set(case.expected_memory_ids) <= available,
            "forbidden_id_correct": set(case.forbidden_memory_ids) <= available
            and not (set(case.expected_memory_ids) & set(case.forbidden_memory_ids)),
            "chronology_correct": _chronology_is_sound(case),
            "prefix_nesting_correct": prefix_checks[str(case.metadata["scenario_family"])],
            "token_stratum_correct": abs(measured - target) <= target * 0.05,
            "answer_leakage_free": answer_occurrences == case.expected_memory_ids,
        }
        results[case.case_id] = checks
    failures = {
        case_id: [field for field, passed in checks.items() if not passed]
        for case_id, checks in results.items()
        if not all(checks.values())
    }
    if failures:
        raise ValueError(f"v1.3 E3 technical prereview failed: {failures}")

    review_path = root / "review_checklist.csv"
    with review_path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(
            stream,
            fieldnames=REVIEW_FIELDS,
            lineterminator="\n",
        )
        writer.writeheader()
        for case in cases:
            writer.writerow({
                "case_id": case.case_id,
                "scenario_family": case.metadata["scenario_family"],
                "stratum": case.metadata["stratum"],
                **{field: "yes" for field in CHECK_FIELDS},
                "notes": f"technical-prereview={PREREVIEW_VERSION}",
            })

    signoff = write_signoff_template(root, review_path)
    report = {
        "benchmark_version": BENCHMARK_VERSION,
        "prereview_version": PREREVIEW_VERSION,
        "status": "pending_human_review",
        "checked_cases": len(cases),
        "checked_scenario_families": validation["scenario_families"],
        "technical_failures": 0,
        "review_sha256": sha256_file(review_path),
        "development_sha256": validation["sha256"],
        "signoff_status": signoff["decision"],
    }
    report_path = root / "ai_prereview.md"
    report_path.write_text(
        "# Benchmark v1.3-E3 Candidate Technical Prereview\n\n"
        f"- prereview version: `{PREREVIEW_VERSION}`\n"
        f"- checked cases: `{len(cases)}`\n"
        f"- checked scenario families: `{validation['scenario_families']}`\n"
        "- technical failures: `0`\n"
        f"- review checklist SHA-256: `{report['review_sha256']}`\n"
        "- status: `pending_human_review`\n\n"
        "All case-level fields are automatic technical evidence. The human "
        "reviewer signs the bound package once in `review_signoff.json`.\n",
        encoding="utf-8",
    )
    manifest_path = root / "candidate_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest.update({
        "review_sha256": report["review_sha256"],
        "signoff_file": "review_signoff.json",
        "technical_prereview": {
            "file": report_path.name,
            "sha256": sha256_file(report_path),
            "version": PREREVIEW_VERSION,
            "checked_cases": len(cases),
            "technical_failures": 0,
        },
    })
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description="Technically prereview v1.3 E3 Development.")
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path(__file__).resolve().parent / "data" / "v1.3-e3",
    )
    args = parser.parse_args()
    print(json.dumps(prereview_candidates(args.data_dir), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
