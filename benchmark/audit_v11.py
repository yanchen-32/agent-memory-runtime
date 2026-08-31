"""Deep, deterministic data-quality audit for Benchmark v1.1 candidates."""

from __future__ import annotations

import argparse
import csv
from collections import Counter, defaultdict
from datetime import datetime
import json
from pathlib import Path
import re

from .artifacts import sha256_file
from .freeze_v1 import CHECK_FIELDS
from .loader import BenchmarkCase, load_jsonl
from .prereview_v11 import write_signoff_template
from .validate_v11 import SPLIT_CASE_COUNTS, validate_candidate_splits


AUDIT_VERSION = "v1.1.0"
SPLITS = tuple(SPLIT_CASE_COUNTS)


def _query_shell(case: BenchmarkCase) -> str:
    value = case.query
    subject = str(case.metadata.get("subject") or "")
    if subject:
        value = value.replace(subject, "<SUBJECT>")
    if case.expected_answer != "UNKNOWN":
        value = value.replace(case.expected_answer, "<ANSWER>")
    value = re.sub(r"2026年\d+月\d+日", "<DATE>", value)
    return re.sub(r"\d+", "<N>", value)


def _answer_supported(case: BenchmarkCase) -> bool:
    if case.expected_answer == "UNKNOWN":
        return not case.expected_memory_ids
    by_id = {str(turn["memory_id"]): str(turn["content"]) for turn in case.conversation}
    evidence = "\n".join(by_id[memory_id] for memory_id in case.expected_memory_ids)
    return all(part in evidence for part in case.expected_answer.split())


def _alias_supported(case: BenchmarkCase) -> bool:
    if not case.answer_aliases:
        return True
    by_id = {str(turn["memory_id"]): str(turn["content"]) for turn in case.conversation}
    evidence = "\n".join(by_id[memory_id] for memory_id in case.expected_memory_ids)
    return all(
        all(part in evidence for part in alias.replace("和", " ").split())
        for alias in case.answer_aliases
    )


def _ambiguous_non_evidence(case: BenchmarkCase) -> bool:
    if case.expected_answer == "UNKNOWN":
        return False
    expected = set(case.expected_memory_ids)
    other = "\n".join(
        str(turn["content"])
        for turn in case.conversation
        if str(turn["memory_id"]) not in expected
    )
    return bool(other) and all(part in other for part in case.expected_answer.split())


def audit_candidates(data_dir: str | Path) -> dict:
    root = Path(data_dir)
    validation = validate_candidate_splits(root)
    cases_by_split = {
        split: load_jsonl(root / f"{split}.jsonl") for split in SPLITS
    }
    all_cases = [case for split in SPLITS for case in cases_by_split[split]]
    failures: list[dict[str, str]] = []

    def check(name: str, passed: bool, evidence: str, severity: str = "high") -> None:
        if not passed:
            failures.append({"check": name, "severity": severity, "evidence": evidence})

    case_ids = [case.case_id for case in all_cases]
    queries = [case.query for case in all_cases]
    check("case_id_uniqueness", len(case_ids) == len(set(case_ids)), "case IDs must be globally unique")
    check("query_exact_uniqueness", len(queries) == len(set(queries)), "questions must be globally unique")

    memory_id_cases: defaultdict[str, list[BenchmarkCase]] = defaultdict(list)
    conversation_cases: defaultdict[tuple[str, ...], list[BenchmarkCase]] = defaultdict(list)
    shells: defaultdict[tuple[str, str, str], list[str]] = defaultdict(list)
    unsupported: list[str] = []
    unsupported_aliases: list[str] = []
    ambiguous: list[str] = []
    answer_leakage: list[str] = []
    temporal_leakage: list[str] = []
    long_context_short: list[str] = []
    for split, cases in cases_by_split.items():
        for case in cases:
            for turn in case.conversation:
                memory_id_cases[str(turn["memory_id"])].append(case)
            conversation_cases[tuple(str(turn["content"]) for turn in case.conversation)].append(case)
            shells[(split, case.category, _query_shell(case))].append(case.case_id)
            if not _answer_supported(case):
                unsupported.append(case.case_id)
            if not _alias_supported(case):
                unsupported_aliases.append(case.case_id)
            if _ambiguous_non_evidence(case):
                ambiguous.append(case.case_id)
            if case.expected_answer != "UNKNOWN" and case.expected_answer in case.query:
                answer_leakage.append(case.case_id)
            by_id = {str(turn["memory_id"]): turn for turn in case.conversation}
            query_time = datetime.fromisoformat(str(case.memory_query_time))
            for memory_id in case.expected_memory_ids:
                if datetime.fromisoformat(str(by_id[memory_id]["valid_from"])) > query_time:
                    temporal_leakage.append(case.case_id)
            if case.category == "long_context":
                char_count = sum(len(str(turn["content"])) for turn in case.conversation)
                if len(case.conversation) < 64 or char_count < 1_500:
                    long_context_short.append(case.case_id)

    bad_memory_reuse = []
    for memory_id, cases in memory_id_cases.items():
        unique_cases = {case.case_id: case for case in cases}.values()
        case_list = list(unique_cases)
        if len(case_list) <= 1:
            continue
        pair_ids = {case.metadata.get("pair_id") for case in case_list}
        if len(case_list) != 2 or len(pair_ids) != 1 or None in pair_ids:
            bad_memory_reuse.append(memory_id)
    bad_conversation_reuse = []
    for contents, cases in conversation_cases.items():
        if len(cases) <= 1:
            continue
        pair_ids = {case.metadata.get("pair_id") for case in cases}
        if len(cases) != 2 or len(pair_ids) != 1 or None in pair_ids:
            bad_conversation_reuse.append(cases[0].case_id)

    check("memory_id_integrity", not bad_memory_reuse, f"unexpected reused IDs: {bad_memory_reuse[:10]}")
    check("conversation_uniqueness", not bad_conversation_reuse, f"unexpected duplicate conversations: {bad_conversation_reuse[:10]}")
    check("expected_answer_evidence", not unsupported, f"unsupported answers: {unsupported[:10]}")
    check("answer_alias_evidence", not unsupported_aliases, f"unsupported aliases: {unsupported_aliases[:10]}")
    check("non_evidence_answer_ambiguity", not ambiguous, f"answers also present in non-evidence: {ambiguous[:10]}")
    check("answer_not_in_question", not answer_leakage, f"answer leakage: {answer_leakage[:10]}")
    check("no_future_expected_evidence", not temporal_leakage, f"future evidence: {temporal_leakage[:10]}")
    check("long_context_depth", not long_context_short, f"short long-context cases: {long_context_short[:10]}")

    cross_split_query_overlap = {}
    cross_split_content_overlap = {}
    for index, left in enumerate(SPLITS):
        for right in SPLITS[index + 1 :]:
            label = f"{left}/{right}"
            left_queries = {case.query for case in cases_by_split[left]}
            right_queries = {case.query for case in cases_by_split[right]}
            left_content = {str(turn["content"]) for case in cases_by_split[left] for turn in case.conversation}
            right_content = {str(turn["content"]) for case in cases_by_split[right] for turn in case.conversation}
            cross_split_query_overlap[label] = len(left_queries & right_queries)
            cross_split_content_overlap[label] = len(left_content & right_content)
    check(
        "cross_split_exact_query_leakage",
        not any(cross_split_query_overlap.values()),
        str(cross_split_query_overlap),
    )
    check(
        "cross_split_exact_content_leakage",
        not any(cross_split_content_overlap.values()),
        str(cross_split_content_overlap),
    )
    max_shell_group = max(len(group) for group in shells.values())
    check("template_concentration", max_shell_group <= 4, f"maximum normalized query shell group={max_shell_group}", "medium")

    review_path = root / "review_checklist.csv"
    with review_path.open(encoding="utf-8", newline="") as stream:
        reviews = list(csv.DictReader(stream))
    review_ids = [row["case_id"] for row in reviews]
    technical_yes = all(row[field] == "yes" for row in reviews for field in CHECK_FIELDS)
    check("review_row_coverage", Counter(review_ids) == Counter(case_ids), "review rows must cover every case exactly once")
    check("technical_prereview", technical_yes, "all six prereview fields must be yes")
    signoff_path = root / "review_signoff.json"
    signoff = json.loads(signoff_path.read_text(encoding="utf-8"))
    signoff_pending = (
        not str(signoff.get("reviewer", "")).strip()
        and signoff.get("decision") == "pending_human_review"
        and signoff.get("review_checklist_sha256") == sha256_file(review_path)
        and signoff.get("reviewed_case_count") == len(all_cases)
    )
    check(
        "human_review_separation",
        signoff_pending,
        "one-time signoff must remain pending during technical audit",
        "critical",
    )

    long_cases = [case for case in all_cases if case.category == "long_context"]
    long_depths = [len(case.conversation) for case in long_cases]
    long_chars = [sum(len(str(turn["content"])) for turn in case.conversation) for case in long_cases]
    report = {
        "benchmark_version": "1.1-candidate",
        "audit_version": AUDIT_VERSION,
        "status": "passed_pending_human_review" if not failures else "failed",
        "valid": not failures,
        "case_count": len(all_cases),
        "case_grain": "one question over one conversation; governance scenarios have paired current/historical questions",
        "split_counts": {split: len(cases) for split, cases in cases_by_split.items()},
        "category_counts": dict(sorted(Counter(case.category for case in all_cases).items())),
        "governance_pairs": validation["governance_pairs"],
        "checks": {
            "structural_validation": True,
            "global_unique_case_ids": len(case_ids),
            "global_unique_queries": len(set(queries)),
            "cross_split_query_overlap": cross_split_query_overlap,
            "cross_split_content_overlap": cross_split_content_overlap,
            "normalized_query_shells": len(shells),
            "max_normalized_shell_group": max_shell_group,
            "unsupported_answers": len(unsupported),
            "unsupported_aliases": len(unsupported_aliases),
            "ambiguous_non_evidence_answers": len(ambiguous),
            "answers_exposed_in_questions": len(answer_leakage),
            "future_expected_evidence": len(temporal_leakage),
            "unexpected_memory_id_reuse": len(bad_memory_reuse),
            "unexpected_conversation_reuse": len(bad_conversation_reuse),
            "long_context_case_count": len(long_cases),
            "long_context_memory_depth_min": min(long_depths),
            "long_context_memory_depth_max": max(long_depths),
            "long_context_chars_min": min(long_chars),
            "long_context_chars_max": max(long_chars),
            "review_rows": len(reviews),
            "technical_fields_all_yes": technical_yes,
            "human_signatures_present": int(bool(str(signoff.get("reviewer", "")).strip())),
        },
        "preaudit_remediations": [
            {
                "severity": "high",
                "finding": "40 long-context cases contained only 12 short memories",
                "remediation": "expanded every case to 64 chronological memories and at least 1500 Chinese characters",
            },
            {
                "severity": "high",
                "finding": "10 semantic-recall distractors repeated the expected memory type",
                "remediation": "replaced the distractor with a distinct session-buffer fact",
            },
            {
                "severity": "medium",
                "finding": "normalized question-template groups reached 24 cases",
                "remediation": "introduced category-specific question variants; maximum group is now 4",
            },
        ],
        "failures": failures,
        "split_sha256": {
            split: sha256_file(root / f"{split}.jsonl") for split in SPLITS
        },
        "review_sha256": sha256_file(review_path),
        "limitations": [
            "Automatic checks cannot certify linguistic naturalness or domain realism; human review remains mandatory.",
            "Visible Test/Holdout candidates are not a secret blind benchmark after repository publication.",
            "External LongBench component notices remain a separate redistribution audit item.",
        ],
    }
    return report


def write_audit(data_dir: str | Path) -> dict:
    root = Path(data_dir)
    report = audit_candidates(root)
    json_path = root / "deep_audit.json"
    json_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    checks = report["checks"]
    markdown = (
        "# Benchmark v1.1 Deep Data-Quality Audit\n\n"
        f"Status: **{report['status']}**. The audit covers {report['case_count']} questions, "
        f"{report['governance_pairs']} governance pairs and all three candidate splits.\n\n"
        "## Evidence after remediation\n\n"
        f"- Unique case IDs / questions: {checks['global_unique_case_ids']} / {checks['global_unique_queries']}\n"
        f"- Unsupported answers / aliases: {checks['unsupported_answers']} / {checks['unsupported_aliases']}\n"
        f"- Ambiguous answers in non-evidence: {checks['ambiguous_non_evidence_answers']}\n"
        f"- Cross-split exact query/content overlap: {sum(checks['cross_split_query_overlap'].values())} / "
        f"{sum(checks['cross_split_content_overlap'].values())}\n"
        f"- Maximum normalized question-shell group: {checks['max_normalized_shell_group']}\n"
        f"- Long-context depth: {checks['long_context_memory_depth_min']}–{checks['long_context_memory_depth_max']} memories; "
        f"{checks['long_context_chars_min']}–{checks['long_context_chars_max']} Chinese characters\n"
        f"- Technical prereview rows: {checks['review_rows']}; human signatures: {checks['human_signatures_present']}\n\n"
        "## Remediated findings\n\n"
        + "".join(
            f"- **{item['severity']}** — {item['finding']}; {item['remediation']}.\n"
            for item in report["preaudit_remediations"]
        )
        + "\n## Remaining limitations\n\n"
        + "".join(f"- {item}\n" for item in report["limitations"])
        + "\nThe data is technically ready for human review, not yet authorized for freeze.\n"
    )
    (root / "deep_audit.md").write_text(markdown, encoding="utf-8")
    signoff = write_signoff_template(
        root,
        root / "review_checklist.csv",
        deep_audit_path=json_path,
    )
    manifest_path = root / "candidate_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["deep_audit"] = {
        "file": json_path.name,
        "sha256": sha256_file(json_path),
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
    parser = argparse.ArgumentParser(description="Deep-audit Benchmark v1.1 candidate data.")
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path(__file__).resolve().parent / "data" / "v1.1",
    )
    args = parser.parse_args()
    print(json.dumps(write_audit(args.data_dir), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
