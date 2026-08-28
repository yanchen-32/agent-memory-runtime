# Milestone 03 — B1 and Unified Benchmark v0.2

## Implemented

- B1 Full-History Agent: sends the complete conversation history without memory retrieval.
- Runtime benchmark adapter: evaluates the proposed Memory Runtime as Ours.
- Unified runner: evaluates B0, B1, B2 and Ours through one entry point.
- Metrics: exact-answer accuracy, Recall@1/5/10, MRR, prompt token estimate, context token estimate, setup latency and query latency.
- Repeated runs: supported with --repeats.
- Client switch: offline deterministic RuleBasedClient or OpenAI-compatible endpoint.
- Embedding switch: deterministic HashEmbeddingModel or optional SentenceTransformerEmbedder.
- Benchmark v0.2: 28 cases covering the original 8 categories plus Temporal, Budget, Multi-hop and Forgetting.

## Run

Offline smoke benchmark:

~~~bash
python experiments/run_all.py
~~~

Repeated offline run:

~~~bash
python experiments/run_all.py --repeats 3
~~~

Real LLM and embedding example:

~~~bash
python experiments/run_all.py \
  --client openai \
  --model your-model-name \
  --base-url http://localhost:8000/v1 \
  --embedding sentence-transformers \
  --embedding-model BAAI/bge-small-zh-v1.5 \
  --repeats 3
~~~

## Metric interpretation

- B0 and B1 do not perform retrieval, so Recall and MRR are recorded as null.
- B2 and Ours report retrieval metrics against benchmark memory IDs.
- prompt_tokens is a deterministic estimate unless a model tokenizer is added.
- latency_ms measures query-time answer latency; setup_latency_ms is reported separately.
- No experimental result is included in source control. Results are generated locally under results/.

## Known limitations

Temporal historical querying, adaptive retrieval weights, semantic consolidation, full context-budget optimization and Kunpeng profiling remain later milestones. The new benchmark cases expose these research questions; their presence does not imply that the corresponding capabilities are already solved.
