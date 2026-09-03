"""Deep structural audit for the v1.3 E3 Development candidate."""

from __future__ import annotations

import argparse
from collections import Counter
import csv
import json
from pathlib import Path

from .artifacts import sha256_file
from .generate_e3_v13 import BENCHMARK_VERSION, SCENARIOS_PER_STRATUM, STRATA
from .loader import load_jsonl
from .prereview_e3_v13 import CHECK_FIELDS, write_signoff_template
from .validate_e3_v13 import validate_development


AUDIT_VERSION = "v1.3-e3.1"


def _load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def audit_candidates(data_dir: str | Path) -> dict:
    root = Path(data_dir)
    development_path = root / "development.jsonl"
    review_path = root / "review_checklist.csv"
    signoff_path = root / "review_signoff.json"
    candidate_path = root / "candidate_manifest.json"
    validation = validate_development(development_path)
    cases = load_jsonl(development_path)
    failures: list[str] = []

    expected_ids = [case.case_id for case in cases]
    with review_path.open(encoding="utf-8", newline="") as stream:
        reviews = list(csv.DictReader(stream))
    observed_review_ids = [str(row.get("case_id", "")) for row in reviews]
    exact_review_coverage = Counter(observed_review_ids) == Counter(expected_ids)
    if not exact_review_coverage:
        failures.append("review checklist does not cover every case exactly once")
    technical_yes = all(
        row.get(field, "").strip().lower() == "yes"
        for row in reviews
        for field in CHECK_FIELDS
    )
    if not technical_yes:
        failures.append("technical prereview fields are incomplete")

    signoff = _load_json(signoff_path)
    signoff_pending = (
        not str(signoff.get("reviewer", "")).strip()
        and signoff.get("decision") == "pending_human_review"
        and signoff.get("scope") == "all_v13_e3_development_candidates"
        and signoff.get("reviewed_case_count") == len(cases)
        and signoff.get("reviewed_scenario_family_count")
        == SCENARIOS_PER_STRATUM
        and signoff.get("review_checklist_sha256") == sha256_file(review_path)
        and signoff.get("development_sha256") == validation["sha256"]
    )
    if not signoff_pending:
        failures.append("signoff must remain complete, bound, and pending during audit")

    candidate = _load_json(candidate_path)
    prereview = candidate.get("technical_prereview", {})
    prereview_path = root / str(prereview.get("file", ""))
    prereview_consistent = (
        prereview_path.is_file()
        and prereview.get("sha256") == sha256_file(prereview_path)
        and prereview.get("checked_cases") == len(cases)
        and prereview.get("technical_failures") == 0
    )
    candidate_consistent = (
        candidate.get("benchmark_version") == BENCHMARK_VERSION
        and candidate.get("status") == "pending_human_review"
        and candidate.get("case_count") == len(cases)
        and candidate.get("scenario_count") == SCENARIOS_PER_STRATUM
        and candidate.get("development_sha256") == validation["sha256"]
        and candidate.get("review_sha256") == sha256_file(review_path)
        and candidate.get("case_order_policy")
        == validation["case_order_policy"]
        and prereview_consistent
    )
    if not candidate_consistent:
        failures.append("candidate manifest is inconsistent with validated artifacts")

    forbidden_split_files = [
        name for name in ("test.jsonl", "holdout.jsonl") if (root / name).exists()
    ]
    if forbidden_split_files:
        failures.append(f"undeclared E3 split files exist: {forbidden_split_files}")

    family_rows = cases[:: len(STRATA)]
    predicate_distribution = Counter(
        str(case.metadata["predicate"]) for case in family_rows
    )
    query_template_distribution = Counter(
        str(case.metadata["query_template_index"]) for case in family_rows
    )
    target_position_distribution = Counter(
        str(case.metadata["target_position_band"]) for case in family_rows
    )
    all_distractor_kinds = sorted({
        str(kind)
        for case in family_rows
        for kind in case.metadata["distractor_kinds"]
    })

    return {
        "benchmark_version": BENCHMARK_VERSION,
        "audit_version": AUDIT_VERSION,
        "status": "passed_pending_human_review" if not failures else "failed",
        "valid": not failures,
        "case_count": len(cases),
        "scenario_family_count": validation["scenario_families"],
        "strata": validation["strata"],
        "token_ranges": validation["token_ranges"],
        "predicate_distribution": dict(sorted(predicate_distribution.items())),
        "query_template_distribution": dict(sorted(query_template_distribution.items())),
        "target_position_distribution": dict(sorted(target_position_distribution.items())),
        "distractor_kinds": all_distractor_kinds,
        "strict_prefix_nested": validation["prefix_nested"],
        "answer_leakage_free": validation["answer_leakage_free"],
        "technical_review_rows": len(reviews),
        "review_exact_coverage": exact_review_coverage,
        "technical_fields_all_yes": technical_yes,
        "human_signatures_present": int(bool(str(signoff.get("reviewer", "")).strip())),
        "candidate_manifest_consistent": candidate_consistent,
        "technical_prereview_consistent": prereview_consistent,
        "development_sha256": validation["sha256"],
        "review_sha256": sha256_file(review_path),
        "forbidden_split_files": forbidden_split_files,
        "failures": failures,
    }


def write_audit(data_dir: str | Path) -> dict:
    root = Path(data_dir)
    report = audit_candidates(root)
    if not report["valid"]:
        raise ValueError(f"v1.3 E3 deep audit failed: {report['failures']}")
    json_path = root / "deep_audit.json"
    json_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    markdown_path = root / "deep_audit.md"
    markdown_path.write_text(
        "# Benchmark v1.3-E3 Candidate Deep Audit\n\n"
        f"Status: **{report['status']}**.\n\n"
        f"- Development cases: {report['case_count']}\n"
        f"- Independent scenario families: {report['scenario_family_count']}\n"
        f"- Technical review rows: {report['technical_review_rows']}\n"
        f"- Strict prefix nesting: {report['strict_prefix_nested']}\n"
        f"- Answer leakage free: {report['answer_leakage_free']}\n"
        f"- Predicate distribution: `{report['predicate_distribution']}`\n"
        f"- Query-template distribution: `{report['query_template_distribution']}`\n"
        f"- Target-position distribution: `{report['target_position_distribution']}`\n"
        f"- Distractor kinds: `{report['distractor_kinds']}`\n"
        f"- Human signatures present during audit: {report['human_signatures_present']}\n\n"
        "All case-level technical checks passed. One package-level human signoff "
        "is still required before freezing.\n",
        encoding="utf-8",
    )
    signoff = write_signoff_template(
        root,
        root / "review_checklist.csv",
        deep_audit_path=json_path,
    )
    manifest_path = root / "candidate_manifest.json"
    manifest = _load_json(manifest_path)
    manifest["deep_audit"] = {
        "file": json_path.name,
        "report_file": markdown_path.name,
        "sha256": sha256_file(json_path),
        "report_sha256": sha256_file(markdown_path),
        "version": AUDIT_VERSION,
        "status": report["status"],
        "valid": report["valid"],
    }
    manifest["review_signoff"] = {
        "file": "review_signoff.json",
        "unsigned_template_sha256": sha256_file(root / "review_signoff.json"),
        "scope": signoff["scope"],
    }
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description="Deep-audit v1.3 E3 Development.")
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path(__file__).resolve().parent / "data" / "v1.3-e3",
    )
    args = parser.parse_args()
    print(json.dumps(write_audit(args.data_dir), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
