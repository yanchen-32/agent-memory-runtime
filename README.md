# Agent Memory Runtime

Competition-oriented Agent Memory Runtime repository for long-horizon Agent memory research and Kunpeng deployment.

## Current milestones

- Milestone 01: B0 No-Memory Agent, B2 Vector Memory Baseline, Benchmark v0.1.
- Milestone 02: Memory Runtime V1 — first-stage memory modules implemented and integrated.
- Milestone 03: B1 Full-History Baseline and unified Benchmark v0.2 runner.

## Quick start

~~~bash
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\\Scripts\\activate
pip install -r requirements.txt
pytest -q

python experiments/run_b0_no_memory.py
python experiments/run_b2_vector_memory.py
python experiments/run_all.py
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
  --embedding-model BAAI/bge-small-zh-v1.5 \\
  --repeats 3
~~~

The default implementation is offline and deterministic. Sentence Transformers is optional and should use an ARM64-compatible embedding model for Kunpeng experiments.

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
  service/             Integrated Write/Read pipelines
  runtime.py           MemoryRuntimeV1 facade
  vector_store.py      Frozen B2 vector baseline store
benchmark/             Benchmark loader, metrics, v0.1 and v0.2 cases
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
- Ours: Memory Runtime V1 write/read pipeline with governance and lifecycle hooks.

memory/vector_store.py remains deliberately minimal so B2 and later ablations remain comparable.

## Metric notes

- B0 and B1 have no retrieval operation; Recall and MRR are recorded as null.
- B2 and Ours report Recall@1/5/10 and MRR against benchmark memory IDs.
- prompt_tokens is a deterministic estimate unless a model tokenizer is configured.
- latency_ms measures query-time answer latency; setup latency is reported separately.
- Generated result files are not committed as experimental evidence.
