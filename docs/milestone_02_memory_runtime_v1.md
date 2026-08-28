# Milestone 02 — Memory Runtime V1

This milestone implements the first executable version of all 13 planned memory modules while keeping the frozen B2 VectorMemoryStore baseline untouched.

## Implemented V1 modules

1. Memory Extraction — deterministic rule extractor + pluggable structured LLM extractor.
2. Memory Classification — Working/Episodic/Semantic rule classifier + pluggable LLM classifier.
3. Memory Storage — abstract MemoryStore, in-memory backend, SQLite backend.
4. Vector Retrieval — MemoryRecord semantic retrieval using the existing embedding abstraction.
5. Hybrid Retrieval — Vector + BM25 + Reciprocal Rank Fusion.
6. Importance Score — transparent multi-feature score.
7. Recency Score — exponential time decay.
8. Reranker — explainable weighted feature reranker.
9. Deduplication — exact and embedding-based duplicate detection with conflict handoff.
10. Memory Update — version-chain supersede operation.
11. Conflict Detection — deterministic subject/predicate/object value conflict detection.
12. Forgetting — strength-based archive lifecycle; no destructive deletion.
13. Compression — single-memory compression with optional summarizer hook.

## Integrated pipelines

Write:

`Extraction -> Classification -> Importance -> Dedup -> Conflict -> Add/Supersede -> Storage`

Read:

`Vector + BM25 -> RRF -> active-status hard filter -> Weighted Rerank -> Compression -> Context`

`MemoryRuntimeV1` wires the V1 components for development and tests. Production components remain dependency-injectable.

## Deliberate V1 boundaries

- B2 remains unchanged and contains no hybrid retrieval or governance features.
- LLM SDKs are not hard dependencies. Structured LLM extraction/classification are adapter-style hooks.
- Sentence Transformers remains optional; HashEmbeddingModel keeps CI/offline execution deterministic.
- Full Temporal QA, conflict taxonomies beyond value conflict, cross-encoder reranking, semantic consolidation, context-budget control, openGauss, and Kunpeng profiling are later milestones.
