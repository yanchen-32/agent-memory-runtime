# Benchmark

## Versions

- benchmark_v0.1.jsonl is the frozen 16-case baseline dataset.
- benchmark_v0.2.jsonl contains the v0.1 cases plus 12 long-horizon cases.
- v0.2 adds Temporal, Budget, Multi-hop and Forgetting categories.

Each case contains a conversation, query, expected answer and expected memory IDs. Budget cases additionally define token_budget. Forgetting cases define forget_memory_ids and optional memory_metadata so lifecycle behavior can be reproduced.

## Unified evaluation

Run all four systems with one evaluator:

~~~bash
python experiments/run_all.py
~~~

The runner writes:

- results/unified_results_v0.2.json: one row per agent, case and repeat.
- results/unified_summary_v0.2.json: aggregated accuracy, token and latency metrics.

B0 and B1 do not retrieve memories, so their Recall and MRR fields are null. This keeps retrieval metrics comparable without treating non-retrieval as a failed retrieval.
