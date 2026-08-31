# Benchmark v1.1 Deep Data-Quality Audit

Status: **passed_pending_human_review**. The audit covers 480 questions, 80 governance pairs and all three candidate splits.

## Evidence after remediation

- Unique case IDs / questions: 480 / 480
- Unsupported answers / aliases: 0 / 0
- Ambiguous answers in non-evidence: 0
- Cross-split exact query/content overlap: 0 / 0
- Maximum normalized question-shell group: 4
- Long-context depth: 64–64 memories; 1901–1903 Chinese characters
- Technical prereview rows: 480; human signatures: 0

## Remediated findings

- **high** — 40 long-context cases contained only 12 short memories; expanded every case to 64 chronological memories and at least 1500 Chinese characters.
- **high** — 10 semantic-recall distractors repeated the expected memory type; replaced the distractor with a distinct session-buffer fact.
- **medium** — normalized question-template groups reached 24 cases; introduced category-specific question variants; maximum group is now 4.

## Remaining limitations

- Automatic checks cannot certify linguistic naturalness or domain realism; human review remains mandatory.
- Visible Test/Holdout candidates are not a secret blind benchmark after repository publication.
- External LongBench component notices remain a separate redistribution audit item.

The data is technically ready for human review, not yet authorized for freeze.
