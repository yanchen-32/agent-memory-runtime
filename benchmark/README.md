# Benchmark v0.1

`benchmark_v0.1.jsonl` is the first frozen benchmark seed set.

Categories:
- fact_recall
- semantic_recall
- temporal
- update
- conflict
- long_context
- noise
- abstention

Important: B2 Vector Memory is expected to perform poorly on some temporal/conflict cases. That is intentional. Those failures become evidence for adding Temporal Versioning and Conflict Governance later.

The benchmark currently contains 16 compact deterministic cases. v0.2 can scale each category to 20-50 cases without changing the schema.
