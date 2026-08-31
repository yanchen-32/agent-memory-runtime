# Milestone 06: Runner Hardening and Benchmark v1.0 Candidate

Date: 2026-08-31

## Completed engineering scope

- Reuse one Sentence Transformer model instance across cases and agents.
- Use case-local embedding caches, batch B2 document insertion, and batch-prime
  Ours before sequential governance writes.
- Retry transient OpenAI-compatible HTTP/network errors with configurable
  exponential backoff; do not retry terminal authentication errors.
- Preserve terminal failures as safe structured rows, report success/failure
  counts, and invalidate formal comparisons when failures remain.
- Append every completed `(case_id, agent, repeat)` to a configuration-bound
  JSONL checkpoint and support exact-configuration resume without duplicates.
- Require a human-approved hash manifest for formal E1/E2 runs; retain an
  explicit Development-only override for non-formal pilots.
- Add 36 AMR-CN Benchmark v1.0 candidate cases: 12 each for Development, Test,
  and Holdout, with scenario-family-disjoint splits and full E1/E2 category
  coverage.
- Add six paired governance scenarios. Each has current and historical queries,
  identical extractor-visible subject/predicate keys, explicit validity times,
  expected evidence IDs, and forbidden stale/future IDs.
- Add structural split validation, review checklist, review guide, and a freeze
  command that refuses incomplete human review.

## Verified

- 50 automated tests pass.
- All 36 candidate cases pass schema and split validation.
- Offline Ours execution completed all 36 cases without runtime failures;
  governance queries had no forbidden-version retrieval, budget cases stayed
  within budget, and multi-hop cases retrieved the complete evidence chain.
- A 6-case Development run with real local `bge-small-zh-v1.5` loaded model
  weights once. First-case setup was about 8.3 seconds; later case setup was
  about 24–35 milliseconds. Applicable retrieval cases reached Recall@5=1 in
  this engineering smoke. RuleBasedClient answer accuracy is not a model result.
- A 72-row, three-repeat B1/Ours checkpoint resumed without adding duplicate
  rows and emitted the full formal artifact bundle.

## Pending human gate

`development.jsonl`, `test.jsonl`, and `holdout.jsonl` remain candidates. The
36-row `review_checklist.csv` has not been signed by a human reviewer, so no
`frozen_manifest.json` exists and formal results are not allowed. The freeze
command will record final hashes only after every review field is approved.

E5 Consolidation continues to use its separate v0.1 benchmark and requires a
separate reviewed v1 data family before formal E5 claims.
