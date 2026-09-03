# v1.2 validates temporal governance and answer quality, but not the latency target

Status: Final Development + Test report  
Protocol: Benchmark v1.2 / experiment protocol v1.1  
Scoring: `legacy-answer-v1` plus frozen Budget scorer `quantity-semantic-v1`  
Holdout: Not executed

## Executive conclusion

The frozen Benchmark v1.2 Development and single authorized Test runs support
the central memory-governance result. Ours answered all Test E2 cases correctly,
answered and retrieved all historical-query cases correctly, and retrieved no
forbidden memory on Update/Conflict/Temporal cases. B2 and B3 retained high
answer accuracy but retrieved a forbidden version on every applicable Test run,
so their answers do not demonstrate version governance.

Ours also improved Test E1 answer F1 over B1 Full History by 0.2292 at the
case-median grain; the paired 95% bootstrap interval was [0.1302, 0.3333]. The
typed Budget result was 24/24 semantically correct and within the 80-token
ceiling. These results support answer quality, temporal correctness, and bounded
context construction under this benchmark and model configuration.

The experiment does **not** establish a 50% end-to-end latency reduction. Ours
was 3.55% slower than B1 on paired Test E1 latency and 1.20% faster on Test E2.
The run also does not complete the separate E3 scaling, E5 adaptive
consolidation, E7 Kunpeng optimization, or evaluator-private Holdout evidence
required by the broader project target.

## Evidence scope and integrity

Benchmark v1.2 contains 480 reviewed cases: 288 Development, 96 Test, and 96
Holdout. Development and Test contain the same eleven category families and
were evaluated; Holdout was not passed to the experiment runner and is excluded
from every result below.

| Split | Cases | Repeats | Agents | Expected rows | Observed rows | Terminal failures |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Development | 288 | 3 | 5 | 4,320 | 4,320 | 0 |
| Test | 96 | 3 | 5 | 1,440 | 1,440 | 0 |
| Holdout | 96 | — | — | — | — | Not run |

Both completed runs used DeepSeek `deepseek-v4-flash`, thinking disabled,
temperature 0, at most 64 output tokens, local `bge-small-zh-v1.5`, Top-K 5,
and query-interleaved alternating agent order. All five agents used the same
answer-format and time-interpretation instructions. B2/B3 intentionally did
not receive Ours' temporal version filtering.

The Test run was executed exactly once in a new directory without `--resume`.
It recorded commit `9080b03ef4bee36555bccc6ddda1e79235a4556f` and the frozen
Test SHA-256
`d20b522e90af47ce61e457cfaca2521438285df9d477ce16d97e89d0a6b9e9ea`.
One row (`v11_test_semantic_006`, Ours, repeat 2) needed the frozen API retry
policy and then succeeded on attempt 2; no row was selectively rerun. Manifests
record that no secrets were stored.

The Development manifest records the earlier base commit `c03023a`, because the
v1.2 implementation was still an uncommitted worktree at launch. Its benchmark
SHA-256 nevertheless matches the frozen Development file, and the exact v1.2
implementation, scorer, data, and freeze metadata were subsequently checkpointed
at `9080b03`. This is a provenance limitation, not a missing result row.

## Primary results

### Test

Accuracy is the frozen per-run answer-accuracy field. Latencies are end-to-end
P50 values and include remote LLM time.

| Agent | E1 accuracy | E2 accuracy | E1 P50 | E2 P50 |
| --- | ---: | ---: | ---: | ---: |
| B0 | 25.00% | 0.00% | 653.1 ms | 611.2 ms |
| B1 | 74.48% | 100.00% | 640.8 ms | 663.2 ms |
| B2 | 88.54% | 96.88% | 682.3 ms | 667.7 ms |
| B3 | 89.06% | 97.92% | 750.4 ms | 726.7 ms |
| Ours | **99.48%** | **100.00%** | 661.6 ms | 674.8 ms |

At the predeclared case-median comparison grain:

- E1 Answer F1: B1 0.7656, Ours 0.9948, delta +0.2292, paired bootstrap
  95% interval [0.1302, 0.3333], 64 paired cases.
- E2 Answer F1: B1 1.0000, Ours 1.0000, delta 0, 32 paired cases.
- E1 paired latency reduction: -3.55% (Ours slower), target not met.
- E2 paired latency reduction: +1.20%, target not met.

### Development consistency check

| Agent | E1 accuracy | E2 accuracy | E1 P50 | E2 P50 |
| --- | ---: | ---: | ---: | ---: |
| B0 | 25.00% | 0.00% | 654.2 ms | 645.5 ms |
| B1 | 74.31% | 100.00% | 676.0 ms | 659.4 ms |
| B2 | 87.15% | 90.63% | 674.4 ms | 624.7 ms |
| B3 | 86.46% | 96.53% | 752.1 ms | 695.5 ms |
| Ours | **97.74%** | **100.00%** | 670.4 ms | 626.3 ms |

Development and Test agree on the direction of the E1 quality result and on
Ours' perfect E2 answer accuracy. Development's paired E1 F1 delta was +0.2002
with 95% interval [0.1458, 0.2558]. It did not show a latency advantage either.

## Budget scoring and result

v1.2 changes only the scoring contract for 40 Budget cases. The frozen typed
`answer_spec` separates:

1. semantic quantity correctness;
2. shortest-format compliance; and
3. joint task success = semantic correctness and token-budget compliance.

This avoids treating answers such as `5` and `5条` as semantically different,
while still reporting strict format behavior. It rejects approximations,
ranges, verbose explanations, and accidental substring matches.

| Split / Agent | Semantic correct | Strict / format compliant | Within budget | Task success | Max prompt |
| --- | ---: | ---: | ---: | ---: | ---: |
| Development / B1 | 0/72 | 0/72 | 72/72 | 0/72 | 51 |
| Development / B2 | 58/72 | 43/72 | 72/72 | 58/72 | 79 |
| Development / B3 | 60/72 | 39/72 | 72/72 | 60/72 | 78 |
| Development / Ours | **63/72** | 42/72 | 72/72 | **63/72** | 78 |
| Test / B1 | 0/24 | 0/24 | 24/24 | 0/24 | 51 |
| Test / B2 | 24/24 | 22/24 | 24/24 | 24/24 | 80 |
| Test / B3 | 24/24 | 22/24 | 24/24 | 24/24 | 80 |
| Test / Ours | **24/24** | 20/24 | 24/24 | **24/24** | 80 |

Ours ranked the expected Budget memory first in every Development and Test run.
Its nine Development semantic failures were three stable `检索数量上限` cases,
each repeated three times, whose outputs were `UNKNOWN`. Test had no Budget
failure. The eight-case Test Budget slice distinguishes Ours from B1, whose
evidence was removed by full-history truncation, but it does not distinguish
Ours from B2/B3 on semantic answer success. The [100, 100] paired bootstrap
interval against B1 reflects eight identical paired differences and must not be
read as an absolute population confidence interval.

## Temporal governance and retrieval

| Gate | Development | Test | Decision |
| --- | ---: | ---: | --- |
| Ours historical answers | 144/144 | 48/48 | Pass |
| Ours historical retrieval | 144/144 | 48/48 | Pass |
| Ours forbidden retrieval, Update/Conflict/Temporal | 0/288 | 0/96 | Pass |
| Ours Recall@1 / Recall@5 / MRR | 95% / 100% / 100% | 95% / 100% / 100% | Pass |
| Ours complete-chain hit | 100% | 100% | Pass |
| B2/B3 forbidden retrieval | Every defined governance run | Every defined governance run | Expected failure |

Recall@1 is 95%, not 100%, because Multi-hop cases have two relevant memories
and one retrieved item can yield Recall@1 = 0.5. Recall@5, MRR, and complete-chain
hit show that the full required evidence was nevertheless present.

## Supported claims and open requirements

| Claim or project requirement | v1.2 evidence status |
| --- | --- |
| E1 answer quality versus B1 | Supported on Development and Test by paired F1 |
| E2 current/historical temporal correctness | Supported |
| Forbidden/stale version suppression | Supported for Ours; B2/B3 fail the governance gate |
| Correct typed Budget answers within 80 tokens | Supported; Test 24/24 run-level successes |
| 50% end-to-end latency reduction | **Not supported** |
| Ours versus B2/B3 Budget differentiation | **Not supported on Test answer success** |
| E3 long-context scaling by frozen size strata | Not completed by this run |
| E5 fixed versus adaptive consolidation | Not completed by this run |
| E7 NUMA/NEON controlled performance study | Not completed by this run |
| Evaluator-private blind Holdout | Not executed; repository Holdout is visible |

## Residual errors and validity limits

- Ours made one incorrect Test E1 run: `v11_test_multihop_003`, repeat 3,
  predicted `Debian + LoongArch` for frozen `Debian LoongArch`. Both relevant
  memories were retrieved. Multi-hop remains on `legacy-answer-v1`; changing
  aliases or scoring after seeing Test would invalidate v1.2.
- The LLM is accessed through a remote endpoint. End-to-end latency therefore
  combines runtime work with network and service variability; it is not a CPU
  microbenchmark.
- Three repeats satisfy the formal minimum but provide less stability than the
  protocol's suggested five repeats.
- Test contains only eight Budget cases. Perfect run-level accuracy is strong
  evidence for these frozen cases, not proof of broad population performance.
- The source-derived benchmark and repository-visible splits are appropriate
  for controlled internal evaluation, but only an evaluator-controlled private
  set can support a genuinely blind final claim.

## Final disposition

Benchmark v1.2 Development and Test are closed evidence. They must not be
rerun, resumed, rescored, or modified for tuning. Future work on `数量上限`
prompt behavior, semantic Multi-hop scoring, latency, or stronger baseline
separation belongs to a new benchmark/protocol version trained only on a new
Development split. The current Holdout remains unexecuted pending separate,
preferably evaluator-controlled authorization.

## Reproducibility sources

- Frozen data and review lineage: `benchmark/data/v1.2/frozen_manifest.json`
- Scoring contract: `docs/answer_scoring_protocol_v1.2.md`
- Development acceptance: `docs/benchmark_v12_development_acceptance.md`
- Test acceptance: `docs/benchmark_v12_test_acceptance.md`
- Raw Development evidence: `results/formal/e1_e2_v12_development_deepseek_bge_r3/`
- Raw Test evidence: `results/formal/e1_e2_v12_test_deepseek_bge_r3/`
- Exact archive inventory and verification: `docs/benchmark_v12_results_archive.md`
