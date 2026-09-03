# E3 v1.3 Development-only ablation protocol

Status: predeclared diagnostic protocol

The frozen v1.3-E3 result remains unchanged and failed its original admission
gate. This protocol may use Development only. Its outputs cannot authorize an
E3 performance claim or replace the frozen result.

## Cumulative single-variable chain

| Variant | Difference from preceding variant |
| --- | --- |
| A0 | frozen Ours behavior, within-run control |
| A1 | render version > 1 as a self-contained `CURRENT_FACT` |
| A2 | add `QUERY_TIME` and `VALID_FROM/VALID_TO` to every retrieved memory |
| A3 | retain only exact query `subject + predicate` matches |
| A4 | keep A3 context and retrieval unchanged; replace only the identity title with B2's title |

The five variants are query-interleaved in alternating order. Each case is run
three times. Comparisons reduce repeats by the per-case median and bootstrap
paired scenario cases within every history stratum.

A3 and A4 must have identical retrieved IDs and byte-identical context for
every case/repeat, while their complete Prompt hashes must differ. Failure of
this invariant invalidates the header-only control.

Terminal failures stay in the checkpoint. Initial execution uses a new output
directory without `--resume`; exact resume is only for an interrupted run and
does not delete failed rows.

## Interpretation

- A0 -> A1 estimates the effect of self-contained current-fact rendering.
- A1 -> A2 estimates the effect of explicit version-time semantics.
- A2 -> A3 estimates the effect of exact SPO filtering.
- A3 -> A4 tests whether the agent identity title alone changes answers.

No Test or Holdout split may be inspected or generated during this process.
After choosing a method, lock its configuration before generating the
pre-registered v1.4 full-factorial candidate.

## Execution

Run all five variants together so adjacent comparisons share the same endpoint
load conditions. The directory must be new and the initial command must omit
`--resume`:

```bash
python experiments/run_e3_ablation.py \
  --output-dir results/development/e3_v13_ablation_a0_a4_deepseek_bge_20260903 \
  --variants A0,A1,A2,A3,A4 \
  --repeats 3 \
  --client openai \
  --model deepseek-v4-flash \
  --base-url https://api.deepseek.com \
  --timeout 180 \
  --max-retries 4 \
  --retry-backoff-seconds 2 \
  --max-tokens 64 \
  --thinking disabled \
  --embedding sentence-transformers \
  --embedding-model .models/bge-small-zh-v1.5 \
  --top-k 5 \
  --bootstrap-samples 10000
```

This executes 2,880 Development diagnostic calls. The resulting report remains
ineligible for the original E3 claim even if every ablation improves.
