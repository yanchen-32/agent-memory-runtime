# Milestone 05: Formal Evaluation, Adaptive Consolidation and Observatory

Date: 2026-08-31

## Completed engineering scope

- Frozen final target v2.0 and formal protocol v1.1 without overwriting the
  historical v1.0 protocol.
- Added raw Exact Match, Normalized Match, Answer Accuracy and token-overlap
  Answer F1 with frozen benchmark aliases.
- Added Query-interleaved agent execution, per-stage E2E timing, per-Case repeat
  medians, B1-vs-Ours paired bootstrap and protocol evidence artifacts.
- Added configurable non-thinking OpenAI-compatible requests without persisting
  API keys.
- Added explainable Adaptive Consolidation trigger and granularity decisions,
  conflict-first blocking, policy version, compression fields and source lineage.
- Added a Fixed vs Adaptive E5 runner.
- Added trace-off-by-default Retrieval/Rerank, Temporal Filter, Context Budget and
  Consolidation events plus a trace overhead runner.
- Corrected database update extraction so `is / uses / changes to` share one
  governed fact key, and corrected Recall@K to divide by all relevant memories.

## Verified locally

- openEuler 22.03 LTS, aarch64, Kunpeng-920, 4 CPU, one NUMA node.
- `agent-memory-real`: Python 3.10.21, sentence-transformers 6.0.0,
  torch 2.13.0, transformers 5.16.1.
- `BAAI/bge-small-zh-v1.5` downloaded as safetensors and successfully encoded
  two Chinese sentences to `(2, 512)` on CPU.
- Offline E1/E2, E5 and trace-overhead smoke paths execute and preserve raw rows.
- These offline RuleBased/HashEmbedding values are not formal competition results.

## External blockers / pending validation

- DeepSeek smoke requires `LLM_API_KEY`, `LLM_MODEL=deepseek-v4-flash` and
  `LLM_BASE_URL=https://api.deepseek.com` in the shell environment.
- The current host has one NUMA node and cannot validate multi-NUMA claims.
- `numactl` is not installed.
- Benchmark v1.0 Development/Test/Holdout and real-model E1–E7 results remain pending.
