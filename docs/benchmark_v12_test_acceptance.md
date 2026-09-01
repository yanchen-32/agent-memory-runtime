# Benchmark v1.2 Test Acceptance

## Decision

The single authorized frozen Benchmark v1.2 Test run completed successfully.
Ours achieved 100% Budget Semantic Accuracy and Budget Task Success, 100% E2
answer accuracy, correct answers and retrieval on all historical questions,
and zero forbidden retrieval. No Test-derived tuning, rerun, resume, data edit,
prompt edit, scorer edit, or parameter change was performed.

The Test result supports the Development conclusion that typed Budget semantics
remove unit-format bias while preserving the token constraint. Holdout remains
unexecuted and reserved for evaluator-controlled final evaluation.

## Execution integrity

| Check | Evidence | Result |
| --- | ---: | --- |
| Expected rows | 1,440 / 1,440 | Pass |
| Unique `(case_id, agent, repeat)` keys | 1,440 / 1,440 | Pass |
| Per-agent coverage | 288 rows each | Pass |
| Terminal failures | 0 | Pass |
| Rows requiring an API retry | 1, then succeeded | Pass |
| Run status | `completed` | Pass |
| Git commit | `9080b03ef4bee36555bccc6ddda1e79235a4556f` | Match |
| Frozen Test SHA-256 | `d20b522e90af47ce61e457cfaca2521438285df9d477ce16d97e89d0a6b9e9ea` | Match |
| Secrets recorded | `false` | Pass |

The only retried row was `v11_test_semantic_006`, Ours, repeat 2. It succeeded
on attempt 2 and no terminal failure or selective rerun occurred. The run used
the same frozen protocol and configuration as Development: DeepSeek
`deepseek-v4-flash`, thinking disabled, maximum 64 output tokens, local BGE
small Chinese v1.5, Top-K 5, three repeats, and query-interleaved alternating
agent order. The new output directory was
`results/formal/e1_e2_v12_test_deepseek_bge_r3`; `--resume` was not used.

## Budget result

| Agent | Semantic correct | Strict / format compliant | Within budget | Budget Task Success | Max prompt |
| --- | ---: | ---: | ---: | ---: | ---: |
| B0 | 0 / 24 (0.00%) | 0 / 24 (0.00%) | 24 / 24 | 0 / 24 (0.00%) | 49 |
| B1 | 0 / 24 (0.00%) | 0 / 24 (0.00%) | 24 / 24 | 0 / 24 (0.00%) | 51 |
| B2 | 24 / 24 (100.00%) | 22 / 24 (91.67%) | 24 / 24 | 24 / 24 (100.00%) | 80 |
| B3 | 24 / 24 (100.00%) | 22 / 24 (91.67%) | 24 / 24 | 24 / 24 (100.00%) | 80 |
| Ours | 24 / 24 (100.00%) | 20 / 24 (83.33%) | 24 / 24 | 24 / 24 (100.00%) | 80 |

Ours retrieved the expected Budget memory at rank 1 in all 24 runs and made no
`UNKNOWN` prediction. At the formal case-median grain, Ours and both retrieval
baselines succeeded on all eight Budget cases; B1 succeeded on none because
the 80-token truncation removed its full-history evidence. Thus Test confirms
the semantic/format separation and memory-vs-full-history distinction, but its
eight Budget cases do not distinguish Ours from B2/B3 on answer success.

The predeclared paired comparison against B1 gives a +100 percentage-point
Budget Semantic Accuracy and Task Success delta. The bootstrap interval is
[100, 100] points because all eight paired differences equal one; this is not
an absolute population confidence interval and the small Test category must be
reported with its 8-case denominator.

## Aggregate E1 and E2

| Agent | E1 accuracy | E2 accuracy | E1 p50 | E2 p50 |
| --- | ---: | ---: | ---: | ---: |
| B0 | 25.00% | 0.00% | 653.1 ms | 611.2 ms |
| B1 | 74.48% | 100.00% | 640.8 ms | 663.2 ms |
| B2 | 88.54% | 96.88% | 682.3 ms | 667.7 ms |
| B3 | 89.06% | 97.92% | 750.4 ms | 726.7 ms |
| Ours | 99.48% | 100.00% | 661.6 ms | 674.8 ms |

At the per-case median grain, Ours' E1 answer F1 is 0.9948 versus 0.7656 for
B1, a +0.2292 delta with 95% paired bootstrap interval [0.1302, 0.3333]. E2 is
tied at 1.0. Ours is 3.55% slower than B1 on paired E1 latency and 1.20% faster
on E2; neither result meets the historical 50% latency-reduction target.

## Temporal governance

| Gate | Evidence | Result |
| --- | ---: | --- |
| Ours historical answers | 48 / 48 | Pass |
| Ours historical retrieval | 48 / 48 | Pass |
| Ours forbidden retrieval on Update/Conflict/Temporal | 0 / 96 | Pass |
| Ours Recall@1 / Recall@5 / MRR on defined retrieval rows | 95% / 100% / 100% | Pass |
| Ours complete-chain hit | 100% | Pass |

B2 and B3 retrieved a forbidden version in all 96 governance runs, as expected
from their intentionally version-unfiltered design. Their high E2 answer
accuracy is not evidence of memory-governance compliance.

## Residual error and interpretation

Ours has one incorrect run among 192 E1 runs. On
`v11_test_multihop_003`, repeat 3, it predicted `Debian + LoongArch` for the
legacy target `Debian LoongArch`. Both relevant memories were retrieved and
the response is semantically correct, but Multi-hop remains governed by
`legacy-answer-v1`. This observation is recorded as a limitation only; no
post-Test alias, scorer, prompt, or method change is permitted.

Test therefore supports the primary claims:

1. Ours maintains temporal answer correctness and prevents stale/forbidden
   retrieval.
2. Ours answers typed Budget quantities correctly under the 80-token ceiling.
3. Ours improves E1 answer quality over B1, but does not demonstrate the
   proposed 50% latency reduction.

Holdout must remain unopened by the experiment runner until a separate final
authorization. Any future Multi-hop scoring design belongs to a new protocol
version and cannot change these v1.2 Test results.
