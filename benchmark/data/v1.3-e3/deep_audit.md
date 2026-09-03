# Benchmark v1.3-E3 Candidate Deep Audit

Status: **passed_pending_human_review**.

- Development cases: 192
- Independent scenario families: 48
- Technical review rows: 192
- Strict prefix nesting: True
- Answer leakage free: True
- Predicate distribution: `{'截止日期': 8, '数据库': 8, '架构': 8, '答辩日期': 8, '部署平台': 8, '项目名称': 8}`
- Query-template distribution: `{'0': 6, '1': 6, '2': 6, '3': 6, '4': 6, '5': 6, '6': 6, '7': 6}`
- Target-position distribution: `{'back': 16, 'front': 16, 'middle': 16}`
- Distractor kinds: `['lexical_decoy', 'same_predicate_other_subject', 'same_subject_other_predicate', 'unrelated']`
- Human signatures present during audit: 0

All case-level technical checks passed. One package-level human signoff is still required before freezing.
