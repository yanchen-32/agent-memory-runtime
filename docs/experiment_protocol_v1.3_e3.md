# Agent Memory Runtime E3 long-context scaling protocol v1.3

Status: Development candidate protocol

Scope: E3 only

Created: 2026-09-03

Parent protocol: `docs/experiment_protocol_v1.1.md`

## Objective

E3 tests whether Ours preserves answer quality while keeping retrieved context
bounded as full conversation history grows, and whether its Very Long
end-to-end latency is at least 50% lower than B1 Full History.

This protocol does not modify the closed Benchmark v1.2 Development/Test
results. It creates a new Development-only candidate and does not create or run
a Test or Holdout split.

## Controlled design

There are 24 scenario families. Every family has the same question, current
target fact, expected answer, expected memory ID, and stale forbidden memory ID
at four history sizes. Larger variants are strict prefix extensions: only
chronologically later distractor memories are appended.

| Stratum | Target unconstrained B1 prompt tokens | Cases |
| --- | ---: | ---: |
| Short | 1,000 | 24 |
| Medium | 4,000 | 24 |
| Long | 16,000 | 24 |
| Very Long | 32,000 | 24 |

The generator constructs the actual B1 prompt and measures it with the
project's deterministic token estimator. A candidate is valid within +/-5% of
its target. Provider-reported API prompt tokens are retained separately and do
not redefine split membership after execution.

Cases use scenario-major stratum-interleaved order. Agents remain
query-interleaved with alternating order, as in protocol v1.1. The primary
comparison is B1 versus Ours; B0/B2/B3 are diagnostic controls.

## Metrics and comparison grain

Every query is run at least three times. Repeats are reduced by per-case median
before comparisons. Each stratum reports independently:

- Answer F1 and paired case bootstrap CI for Ours minus B1;
- prompt and context tokens;
- memory, context-build, LLM, post-processing, and end-to-end latency;
- paired end-to-end latency reduction versus B1;
- Recall@K, MRR, complete-chain hit, and forbidden retrieval.

Setup/ingestion time is recorded separately and excluded from query E2E under
the parent protocol. Remote endpoint latency is part of E2E and must not be
reported as a local CPU optimization result.

## Predeclared admission gates

1. `CI95(delta Answer F1 Ours-B1).lower >= 0` in every stratum.
2. Ours' context is stable: the relative spread between the maximum and
   minimum stratum-level means of per-case median Context Tokens is at most 10%.
3. Ours has zero forbidden retrievals and zero terminal failures.
4. The paired Very Long E2E latency reduction against B1 is at least 50%.
5. Formal evidence requires complete unique run coverage and a reviewed,
   frozen benchmark hash. Candidate pilots are never formal evidence.

All five gates must pass for E3 admission. In particular, the original 50%
latency claim is authorized only by Gate 4 on Very Long under formally eligible
evidence. An overall average or improvement in Short/Medium/Long cannot replace
that result.

## Validity constraints

- Test/Holdout data must not be derived from v1.2 Test outputs.
- Prompt, model, embedding, Top-K, generation parameters, retry policy, and
  scoring contract must be identical across agents in a formal run.
- Failed requests remain recorded and invalidate the corresponding formal
  comparison; they cannot be deleted or selectively rerun.
- Stratum membership, aliases, targets, and thresholds freeze before formal
  execution and cannot be changed after observing results.
- If the provider rejects the 32K input or changes tokenization materially, the
  run is a failed/incompatible configuration, not evidence for a smaller
  post-hoc Very Long definition.

## Candidate workflow

```bash
python -m benchmark.generate_e3_v13
python -m benchmark.validate_e3_v13
```

After technical review and human signoff, freeze the candidate in a separate
checkpoint. Then run `experiments/run_e3_scaling.py` in a new output directory
with real DeepSeek and BGE. Until that freeze exists, only explicitly labelled
`--allow-unreviewed-benchmark` pilots are permitted.
