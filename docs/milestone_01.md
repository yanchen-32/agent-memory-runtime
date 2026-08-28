# Milestone 01 — Baselines and Benchmark v0.1

Status:
- [x] Git project established
- [x] B0 No-Memory Agent
- [x] B2 Vector Memory Baseline
- [x] Benchmark v0.1

Scope guard:
- B0 receives only the current query.
- B2 implements embedding + vector cosine search + Top-K context injection.
- Temporal versioning, conflict governance, BM25/RRF, consolidation, lifecycle and context budget are intentionally absent.

Acceptance:
1. `pytest -q` passes.
2. `python experiments/run_b0_no_memory.py` produces `results/b0_no_memory_v0.1.json`.
3. `python experiments/run_b2_vector_memory.py` produces `results/b2_vector_memory_v0.1.json` and prints Recall/MRR metrics.
4. All 8 frozen benchmark categories are represented.
