# Milestone 04: B3 Hybrid Memory and Consolidation

## Scope

This milestone adds two independently testable capabilities to the frozen
Agent Memory Runtime architecture:

1. **B3 Hybrid Memory baseline**: Vector Retrieval + BM25 + Reciprocal Rank
   Fusion (RRF). B3 does not use Runtime V1 reranking, temporal governance,
   conflict resolution, forgetting, or consolidation.
2. **Episodic → Semantic Consolidation V1**: deterministic grouping of active
   episodic records, semantic record generation, source traceability, and
   idempotent re-consolidation.

## Consolidation contract

- Groups with fewer than two episodic records are skipped.
- Extracted `(subject, predicate, object)` triples are grouped by the complete
  triple. Different object values therefore remain separate and are not
  silently merged across a known conflict.
- Episodic records are preserved.
- Generated semantic records contain `source_ids`, `source_count`, and the
  consolidation engine version in metadata.
- Re-running consolidation updates the existing semantic record instead of
  creating duplicates.
- The current summary is deterministic and uses an existing source fact; no
  LLM-generated fact is treated as ground truth.

## Verification

```bash
python3 -m compileall -q agent memory benchmark
python3 experiments/run_all.py \
  --benchmark benchmark/data/consolidation_v0.1.jsonl \
  --agents B3,Ours
```

The repository test suite includes focused B3 and consolidation tests. If the
environment does not provide `pytest`, run the compile and benchmark smoke
commands above and record the missing test dependency rather than claiming a
full test pass.
