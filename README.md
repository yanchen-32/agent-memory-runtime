# Agent Memory Runtime

Competition-oriented Agent Memory Runtime repository for long-horizon Agent memory research and Kunpeng deployment.

## Current milestones

- Milestone 01: B0 No-Memory Agent, B2 Vector Memory Baseline, Benchmark v0.1.
- Milestone 02: Memory Runtime V1 — all 13 first-stage memory modules implemented and integrated.

## Quick start

```bash
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -r requirements.txt
pytest -q
python experiments/run_b0_no_memory.py
python experiments/run_b2_vector_memory.py
```

The default implementation is offline and deterministic. For a production-like vector baseline/runtime, install `sentence-transformers` separately and use `SentenceTransformerEmbedder` with an ARM64-compatible embedding model.

## Repository layout

```text
agent/                 Agent interfaces and frozen B0/B2 agents
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
benchmark/             Benchmark loader, metrics and v0.1 cases
experiments/           Baseline experiment scripts
tests/                 Regression tests
docs/                  Milestone and architecture notes
```

## V1 data flow

Write pipeline:

```text
Conversation / Tool Result
  -> Memory Extraction
  -> Memory Classification
  -> Importance Score
  -> Deduplication
  -> Conflict Detection
  -> ADD / SUPERSEDE
  -> Memory Storage
```

Read pipeline:

```text
Query
  -> Vector Retrieval + BM25
  -> RRF Hybrid Retrieval
  -> Active-status validity filter
  -> Importance / Utility / Recency / Entity Rerank
  -> Compression
  -> Memory Context
```

Lifecycle V1 computes memory strength from importance, utility, recency and access frequency, then archives low-value memories instead of destructively deleting them.

## Baseline integrity

- **B0 No Memory**: the agent receives only the current query.
- **B2 Vector Memory**: embedding + cosine vector search + Top-K prompt injection.

`memory/vector_store.py` remains deliberately minimal. It does not contain BM25, versioning, conflict governance, lifecycle or reranking. New capabilities live in separate Memory Runtime modules so B2/B3/Proposed and later ablations remain comparable.

## Using Memory Runtime V1

```python
from memory import HashEmbeddingModel, MemoryRuntimeV1

runtime = MemoryRuntimeV1(embedder=HashEmbeddingModel(dim=384))

runtime.write(
    [{"role": "user", "content": "Agent Memory项目的数据库是openGauss。"}],
    user_id="demo-user",
)

result = runtime.read("项目数据库是什么？", top_k=3, user_id="demo-user")
print(result.context)
```

All production-sensitive components are dependency-injectable. `LLMMemoryExtractor` and `LLMMemoryClassifier` accept generic JSON generator callables, and `SentenceTransformerEmbedder` can replace the hash embedder without changing the runtime interfaces.

## Connecting a real LLM

The repository includes a minimal OpenAI-compatible client for Agent baselines. It works with endpoints exposing `/v1/chat/completions`.

```bash
export LLM_BASE_URL=http://localhost:8000/v1
export LLM_API_KEY=EMPTY
export LLM_MODEL=your-model-name
```

For reportable experiments, keep the same LLM, embedding model, generation parameters and benchmark split across compared methods.
