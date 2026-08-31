# Benchmark v1.0 Data Sources

This directory stores manifests and review material, not large upstream data.

## AMR-CN governance candidate splits

The checked-in `development.jsonl`, `test.jsonl`, and `holdout.jsonl` files are
candidate v1.0 E1/E2 splits. They contain 36 cases, including six paired
Update/Conflict scenarios. Every governance pair includes a current-state query and a
historical query, with:

- the same extractor-visible `subject + predicate` for both versions;
- increasing timezone-aware `valid_from` timestamps;
- the expected version in `expected_memory_ids`;
- the stale or future version in `forbidden_memory_ids`;
- a point-in-time `memory_query_time`;
- scenario families disjoint across Development, Test, and Holdout.

They are deliberately marked `pending_human_review` and cannot support formal
claims yet. Run structural validation with:

```bash
python -m benchmark.validate_splits benchmark/data/v1.0
```

Review every row in `review_checklist.csv`. Each check must be `yes`, `reviewer`
must identify the human reviewer, and `decision` must be `approved`. Only then
may the frozen manifest be generated:

```bash
python -m benchmark.freeze_v1 \
  --data-dir benchmark/data/v1.0 \
  --reviews benchmark/data/v1.0/review_checklist.csv
```

The freeze command refuses incomplete review and writes `frozen_manifest.json`
only after all cases pass. `candidate_manifest.json` records the current
pre-review hashes and explicitly sets `formal_results_allowed` to false.

Each split also covers fact recall, semantic recall, long context, noise,
abstention, budget, multi-hop and forgetting. E5 Consolidation remains a
separate benchmark family and must undergo its own versioned review before E5
formal claims; it is intentionally not mixed into the E1/E2 runner.

## External benchmark A: LongMemEval-S Cleaned

- Upstream: `xiaowu0162/longmemeval-cleaned`
- File: `longmemeval_s_cleaned.json`
- License: MIT
- Upstream file SHA256:
  `d6f21ea9d60a0d56f34a05b609c79c88a451d2ae03597821ea3d5a9678c3a442`
- Pinned upstream revision:
  `98d7416c24c778c2fee6e6f3006e7a073259d48f`
- Local ignored path:
  `.datasets/longmemeval-cleaned/longmemeval_s_cleaned.json`
- Role: frozen external test; never tune project parameters on its results.

The adapter preserves upstream questions, answers, evidence session IDs and
session order. Each upstream session becomes one AMR memory record. Because the
source timestamps contain no timezone, the adapter deterministically assigns UTC
and records that assumption in every case's metadata.

An input audit found 13 cases with one repeated non-answer session ID each.
No duplicated ID intersects `answer_session_ids`. The adapter therefore gives
only those repeated filler occurrences stable `__occN` suffixes and preserves
all official evidence IDs unchanged. A duplicated answer-session ID is treated
as an error requiring manual adjudication.

Convert five smoke cases:

```bash
python -m benchmark.adapters.longmemeval \
  --input .datasets/longmemeval-cleaned/longmemeval_s_cleaned.json \
  --output .datasets/converted/longmemeval_s_smoke5.jsonl \
  --limit 5

python -m benchmark.validate_v1 \
  .datasets/converted/longmemeval_s_smoke5.jsonl
```

Remove `--limit 5`, write to `longmemeval_s_v1.jsonl`, and run the same
validator for the frozen 500-case external test. Its expected converted SHA256
is recorded in `source_manifest.json`.

Converting or translating the source invalidates direct comparability with the
official end-to-end score unless the official evaluation protocol is retained.

## Project-specific AMR-CN

Development/Test/Holdout remain project-owned Chinese governance cases. Split
by scenario family rather than individual rows to prevent template leakage.
