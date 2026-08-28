# Agent Memory Runtime

Competition-oriented Agent Memory Runtime starter repository.

Current frozen milestone:
- B0 No-Memory Agent
- B2 Vector Memory Baseline
- Benchmark v0.1

## Quick start

```bash
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -r requirements.txt
pytest -q
python experiments/run_b0_no_memory.py
python experiments/run_b2_vector_memory.py
```

The default implementation is fully offline and deterministic. For a stronger vector baseline, install `sentence-transformers` and switch to `SentenceTransformerEmbedder`.

## Repository layout

```text
agent/          Agent interfaces and B0/B2 agents
memory/         Vector memory store and embedding backends
benchmark/      Benchmark loader, metrics and v0.1 cases
experiments/    Runnable baseline scripts
results/        Generated experiment outputs (gitignored)
tests/          Minimal regression tests
docs/           Frozen milestone notes
```

## Baseline definitions

- **B0 No Memory**: the agent receives only the current query.
- **B2 Vector Memory**: historical memory is embedded, cosine-ranked, and Top-K results are packed into the prompt.

B2 deliberately does **not** include BM25, temporal validity, conflict handling, consolidation, lifecycle or budget-aware packing. Those belong to later stages and must remain separable for ablation.

## Connecting a real LLM

The repository also includes a minimal OpenAI-compatible client. It works with local or remote endpoints exposing `/v1/chat/completions`.

```bash
export LLM_BASE_URL=http://localhost:8000/v1
export LLM_API_KEY=EMPTY
export LLM_MODEL=your-model-name
```

For reportable results, use the same LLM and generation parameters across B0/B1/B2/B3/Proposed.
