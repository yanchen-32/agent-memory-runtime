# Benchmark v1.0 Data Sources

This directory stores manifests and review material, not large upstream data.

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
