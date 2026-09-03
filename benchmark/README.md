# Benchmark

## Versions

- benchmark_v0.1.jsonl is the frozen 16-case baseline dataset.
- benchmark_v0.2.jsonl contains the v0.1 cases plus 12 long-horizon cases.
- v0.2 adds Temporal, Budget, Multi-hop and Forgetting categories.
- `data/v1.3-e3/` is the 192-case, Development-only long-context scaling
  candidate. It uses 48 matched scenarios at approximately 1K/4K/16K/32K
  unconstrained B1 prompt tokens and remains pending human review.

Each case contains a conversation, query, expected answer and expected memory IDs.
The loader also supports frozen `answer_aliases`, `forbidden_memory_ids` and
metadata. Budget cases define token_budget. Forgetting cases define
forget_memory_ids and optional memory_metadata so lifecycle behavior can be reproduced.

## Unified evaluation

Run all five systems with one evaluator:

~~~bash
python experiments/run_all.py
~~~

The smoke runner writes:

- results/unified_results_v0.2.json: one row per agent, case and repeat.
- results/unified_summary_v0.2.json: aggregated accuracy, token and latency metrics.

B0 and B1 do not retrieve memories, so their Recall and MRR fields are null. This keeps retrieval metrics comparable without treating non-retrieval as a failed retrieval.

Formal E1/E2 runs additionally write `manifest.json`, `raw_rows.jsonl`,
`summary.json`, `summary.csv`, `environment.txt`, `run.log` and `figures/`.
Answer F1 uses deterministic NFKC normalization and English-word/Chinese-character
token overlap. B1 and Ours are compared by paired Case bootstrap after reducing
repeats with the per-Case median.
