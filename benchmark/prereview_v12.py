"""Deterministic technical prereview for Benchmark v1.2 scoring candidates."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

from .artifacts import sha256_file
from .generate_v12 import REVIEW_FIELDS, quantity_answer_spec
from .loader import BenchmarkCase, load_jsonl
from .validate_v12 import SPLITS, validate_candidate_splits


PREREVIEW_VERSION = "v1.2.0"
CHECK_FIELDS = REVIEW_FIELDS[1:-1]


def write_signoff_template(
    data_dir: str | Path,
    review_path: str | Path,
    *,
    deep_audit_path: str | Path | None = None,
) -> dict:
    root = Path(data_dir)
    review_file = Path(review_path)
    validation = validate_candidate_splits(root)
    audit_file = Path(deep_audit_path) if deep_audit_path else None
    payload = {
        "benchmark_version": "1.2-candidate",
        "scope": "all_development_test_holdout_scoring_candidates",
        "reviewed_case_count": validation["total_cases"],
        "review_checklist_file": review_file.name,
        "review_checklist_sha256": sha256_file(review_file),
        "split_sha256": {
            split: report["sha256"] for split, report in validation["splits"].items()
        },
        "deep_audit_file": audit_file.name if audit_file else None,
        "deep_audit_sha256": sha256_file(audit_file) if audit_file else None,
        "attestation": (
            "I reviewed the v1.2 scoring-migration audit and approve the bound "
            "480-case Development/Test/Holdout set for freezing."
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
    if case.expected_answer == "UNKNOWN":
        return not case.expected_memory_ids
    evidence_by_id = {
        str(turn["memory_id"]): str(turn["content"]) for turn in case.conversation
    }
    evidence = "\n".join(evidence_by_id[memory_id] for memory_id in case.expected_memory_ids)
    return bool(evidence) and all(part in evidence for part in case.expected_answer.split())


def _aliases_are_sound(case: BenchmarkCase) -> bool:
    if not case.answer_aliases:
        return True
    evidence_by_id = {
        str(turn["memory_id"]): str(turn["content"]) for turn in case.conversation
    }
    evidence = "\n".join(evidence_by_id[memory_id] for memory_id in case.expected_memory_ids)
    return all(
        all(part in evidence for part in alias.replace("和", " ").split())
        for alias in case.answer_aliases
    )


def prereview_candidates(data_dir: str | Path) -> dict:
    root = Path(data_dir)
    validation = validate_candidate_splits(root)
    cases = [
        case
        for split in SPLITS
        for case in load_jsonl(root / f"{split}.jsonl")
    ]
    results = {}
    for case in cases:
        available = {str(turn["memory_id"]) for turn in case.conversation}
        results[case.case_id] = {
            "question_clear": bool(case.query.strip()) and "？" in case.query,
            "answer_correct": _answer_is_evidenced(case),
            "evidence_ids_correct": set(case.expected_memory_ids) <= available,
            "forbidden_ids_correct": set(case.forbidden_memory_ids) <= available
            and not (set(case.expected_memory_ids) & set(case.forbidden_memory_ids)),
            "timestamps_correct": bool(case.query_time and case.memory_query_time),
            "aliases_checked": _aliases_are_sound(case),
            "answer_spec_checked": (
                case.answer_spec == quantity_answer_spec(case.expected_answer)
                if case.category == "budget"
                else case.answer_spec is None
            ),
        }
    failures = {
        case_id: [field for field, passed in checks.items() if not passed]
        for case_id, checks in results.items()
        if not all(checks.values())
    }
    if failures:
        raise ValueError(f"v1.2 technical prereview failed: {failures}")

    review_path = root / "review_checklist.csv"
    with review_path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=REVIEW_FIELDS)
        writer.writeheader()
        for case in cases:
            writer.writerow({
                "case_id": case.case_id,
                **{field: "yes" for field in CHECK_FIELDS},
                "notes": f"technical-prereview={PREREVIEW_VERSION}",
            })
    signoff = write_signoff_template(root, review_path)
    report = {
        "benchmark_version": "1.2-candidate",
        "prereview_version": PREREVIEW_VERSION,
        "status": "pending_human_review",
        "checked_cases": len(cases),
        "checked_budget_answer_specs": validation["budget_answer_specs"],
        "technical_failures": 0,
        "split_sha256": {
            split: item["sha256"] for split, item in validation["splits"].items()
        },
        "review_sha256": sha256_file(review_path),
        "signoff_status": signoff["decision"],
    }
    report_path = root / "ai_prereview.md"
    report_path.write_text(
        "# Benchmark v1.2 Candidate Technical Prereview\n\n"
        f"- prereview version: `{PREREVIEW_VERSION}`\n"
        f"- checked cases: `{len(cases)}`\n"
        f"- checked Budget answer specs: `{validation['budget_answer_specs']}`\n"
        "- technical failures: `0`\n"
        f"- review checklist SHA-256: `{report['review_sha256']}`\n"
        "- status: `pending_human_review`\n\n"
        "All case-level fields are automatic evidence. The reviewer signs the "
        "bound package once in `review_signoff.json`.\n",
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
    parser = argparse.ArgumentParser(description="Technically prereview Benchmark v1.2.")
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path(__file__).resolve().parent / "data/v1.2",
    )
    args = parser.parse_args()
    print(json.dumps(prereview_candidates(args.data_dir), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
