# Agent Memory Runtime

Competition-oriented Agent Memory Runtime repository for long-horizon Agent memory research and Kunpeng deployment.

## Current milestones

- Milestone 01: B0 No-Memory Agent, B2 Vector Memory Baseline, Benchmark v0.1.
- Milestone 02: Memory Runtime V1 — first-stage memory modules implemented and integrated.
- Milestone 03: B1 Full-History Baseline and unified Benchmark v0.2 runner.
- Milestone 04: B3 Hybrid Baseline, Temporal/Budget experiments and Consolidation V1.
- Milestone 05: Formal Evaluation v1.1, Adaptive Consolidation V1 and Memory Observatory backend.

## Quick start

~~~bash
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\\Scripts\\activate
pip install -r requirements.txt
pytest -q

python experiments/run_b0_no_memory.py
python experiments/run_b2_vector_memory.py
python experiments/run_all.py
python experiments/run_e1_e2.py --repeats 3 --allow-unreviewed-benchmark
~~~

The unified runner evaluates B0, B1, B2 and Ours with the same answer, retrieval, token and latency fields:

~~~text
results/unified_results_v0.2.json
results/unified_summary_v0.2.json
~~~

Use repeated offline runs for variance estimation:

~~~bash
python experiments/run_all.py --repeats 3
~~~

Use an OpenAI-compatible LLM endpoint:

~~~bash
python experiments/run_all.py \\
  --client openai \\
  --model your-model-name \\
  --base-url http://localhost:8000/v1 \\
  --embedding sentence-transformers \\
  --embedding-model .models/bge-small-zh-v1.5 \\
  --thinking disabled \\
  --repeats 3
~~~

Run one converted benchmark case before spending API budget on a full run:

~~~bash
python experiments/run_all.py \
  --benchmark .datasets/converted/longmemeval_s_v1.jsonl \
  --case-id <QUESTION_ID> \
  --agents Ours \
  --client openai \
  --model deepseek-v4-flash \
  --base-url https://api.deepseek.com \
  --embedding sentence-transformers \
  --embedding-model .models/bge-small-zh-v1.5 \
  --output-prefix longmemeval_single_ours
~~~

The command reads the API key from `LLM_API_KEY`; it does not store the key in
the result files. Add `B2,B3` only after the one-call smoke test succeeds.

Long runs write an append-only checkpoint after every `(case, agent, repeat)`.
Use the exact same configuration and add `--resume` after an interruption.
Transient HTTP failures are retried with exponential backoff; terminal failures
are retained as `status=failed` rows without storing response bodies or secrets.
The Sentence Transformer weights are loaded once per process, while each case
uses an isolated embedding cache and batched document encoding.

Benchmark v1.0 validation and its completed human-review freeze are documented
in `benchmark/data/v1.0/README.md`. Formal `run_e1_e2.py` runs require the
matching signed `frozen_manifest.json`; use `--allow-unreviewed-benchmark` only
for Development pilots whose outputs cannot support formal claims.

The default implementation is offline and deterministic. RuleBasedClient and
HashEmbeddingModel are smoke-test tools only. Formal results must use the same
real LLM and Embedding snapshot for every comparable agent. API keys must be
provided through environment variables and are never written to result manifests.

## Repository layout

~~~text
agent/                 Agent interfaces and B0/B1/B2/Ours adapters
memory/
  schema/              MemoryRecord and runtime enums/results
  extraction/          Memory Extraction V1
  classification/      Memory Classification V1
  storage/             MemoryStore + in-memory/SQLite adapters
  retrieval/           Vector, BM25 and Hybrid/RRF retrieval
  scoring/             Importance and Recency scoring
  governance/          Dedup, Conflict Detection, Versioned Update
  lifecycle/           Forgetting/Archive and Compression
  consolidation/       Fixed and Adaptive Episodic -> Semantic policies
  service/             Integrated Write/Read pipelines
  observability.py     Unified mechanistic trace events
  runtime.py           MemoryRuntimeV1 facade
  vector_store.py      Frozen B2 vector baseline store
benchmark/             Data, Answer F1, paired statistics and artifact manifests
experiments/           Baseline and unified experiment scripts
tests/                 Regression tests
docs/                  Milestone and architecture notes
~~~

## V1 data flow

Write pipeline:

~~~text
Conversation / Tool Result
  -> Memory Extraction
  -> Memory Classification
  -> Importance Score
  -> Deduplication
  -> Conflict Detection
  -> ADD / SUPERSEDE
  -> Memory Storage
~~~

Read pipeline:

~~~text
Query
  -> Vector Retrieval + BM25
  -> RRF Hybrid Retrieval
  -> Active-status validity filter
  -> Importance / Utility / Recency / Entity Rerank
  -> Compression
  -> Memory Context
~~~

B1 deliberately bypasses this memory pipeline and sends the full conversation history to the model. B2 remains a minimal vector baseline. Ours uses MemoryRuntimeV1.

## Baseline integrity

- B0 No Memory: the agent receives only the current query.
- B1 Full History: the agent receives the complete conversation history.
- B2 Vector Memory: embedding + cosine vector search + Top-K prompt injection.
- B3 Hybrid Memory: Vector + BM25 + RRF only.
- Ours: complete runtime with governance, reranking, budget, consolidation and lifecycle.

memory/vector_store.py remains deliberately minimal so B2 and later ablations remain comparable.

## Metric notes

- B0 and B1 have no retrieval operation; Recall and MRR are recorded as null.
- B2, B3 and Ours report Recall@1/5/10 and MRR against benchmark memory IDs.
- Answer F1 is the primary answer-quality metric. Raw Exact Match,
  Normalized Match and Answer Accuracy remain separate auxiliary metrics.
- prompt_tokens is a deterministic estimate unless a model tokenizer is configured.
- End-to-end latency is split into memory, context build, LLM and post stages;
  setup latency is reported separately.
- Generated result files are not committed as experimental evidence.

## Temporal query and context budget

MemoryRuntimeV1.read accepts query_time. Omitting query_time reads the current active state; supplying query_time reads versions valid at that point using the interval [valid_from, valid_to).

The runtime also exposes ContextBudgetManager through select_context. Selection combines retrieval relevance, memory importance, diversity, redundancy and token efficiency. The budget applies to the complete prompt represented by prefix, selected context and suffix.

Run an offline, non-formal infrastructure smoke with three repeats:

~~~bash
python experiments/run_e1_e2.py --allow-unreviewed-benchmark
~~~

The signed `benchmark/data/v1.0/frozen_manifest.json` is checked in. Run the
frozen Test split without the override:

~~~bash
python experiments/run_e1_e2.py \
  --benchmark benchmark/data/v1.0/test.jsonl \
  --repeats 3
~~~

The E1 output covers long-term recall, budget, multi-hop and forgetting. The E2
output covers update, conflict and temporal query accuracy. The runner alternates
agent order by Query and writes a protocol manifest, raw JSONL, summaries and
B1-vs-Ours paired Bootstrap comparisons.

Run Fixed vs Adaptive Consolidation:

~~~bash
python experiments/run_e5_consolidation.py --repeats 3
~~~

Full mechanistic tracing is off by default. Enable it for Demo evidence with
`--trace`. Measure trace-on/off overhead separately:

~~~bash
python experiments/run_trace_overhead.py --repeats 30
~~~

The frozen target and protocol are in `docs/final_target_v2.0.md` and
`docs/experiment_protocol_v1.1.md`.
