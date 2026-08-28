# Milestone 04 — Temporal Query, Context Budget and E1/E2

## Implemented

- Point-in-time version filtering through valid_from and valid_to.
- Current-state reads when query_time is omitted.
- Historical-state reads when query_time is supplied.
- Controlled write timestamps for reproducible version-chain tests.
- ContextBudgetManager combining relevance, importance, diversity, redundancy and token efficiency.
- Budget cases run once without a budget and once with the case budget.
- Per-case budget-before/after prompt tokens, context tokens, accuracy delta and latency fields.
- Formal E1 and E2 runner with three repeats by default.
- Summary statistics include accuracy mean/std and latency mean/std/P50/P95.

## Commands

Install dependencies:

~~~bash
pip install -r requirements.txt
~~~

Run unit tests:

~~~bash
pytest -q
~~~

Run the unified benchmark:

~~~bash
python experiments/run_all.py
~~~

Run formal E1/E2 experiments:

~~~bash
python experiments/run_e1_e2.py
~~~

The default E1/E2 run uses three repeats. Use a real LLM and embedding model for reportable results:

~~~bash
python experiments/run_e1_e2.py \
  --client openai \
  --model your-model-name \
  --base-url http://localhost:8000/v1 \
  --embedding sentence-transformers \
  --embedding-model BAAI/bge-small-zh-v1.5 \
  --repeats 3
~~~

## Output files

- e1_long_term_accuracy_v0.2.json
- e2_update_conflict_v0.2.json

Each file stores the selected rows and aggregated summaries. Results are ignored by Git and are not included as evidence until they are produced and reviewed.

## Scope boundary

Temporal query is now executable for explicit version timestamps. Query intent parsing, richer temporal expressions, adaptive retrieval weights, semantic consolidation and Kunpeng profiling remain subsequent work.
