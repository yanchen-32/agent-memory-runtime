# Benchmark v1.1 Development Acceptance

## Technical summary

The frozen Benchmark v1.1 Development run is complete and passes the two
pre-declared gates for considering Test: Ours answered all 144 historical runs
correctly with correct historical retrieval, and retrieved zero forbidden
memories across all 288 governance runs. The run contains all 4,320 expected
`case × agent × repeat` rows, has no duplicate run keys, no terminal failures,
and no API retries.

Ours achieved the highest E1 run-level accuracy (93.40%) and tied B1 at 100%
on E2. The paired, per-case median E1 answer-F1 improvement over B1 was 19.38
percentage points (95% bootstrap CI: 13.76–25.17 points). This accuracy result
does not establish a latency advantage: Ours was 2.85% slower than B1 on E1
and 5.11% slower on E2 under the paired median comparison, so the protocol's
50% latency-reduction target was not met.

The Development evidence authorizes a Test decision under the declared
governance gates, but it also identifies Budget and Multi-hop as the remaining
Ours error concentrations. Test and Holdout were not executed in this run.

## Ours passes the temporal-governance gates

| Gate | Evidence | Result |
| --- | ---: | --- |
| Historical answer correctness | 144 / 144 runs; 48 cases × 3 repeats | Pass |
| Historical retrieval correctness | 144 / 144 runs | Pass |
| Forbidden retrieval | 0 forbidden memories in 288 governance runs | Pass |
| Ours Recall@5 | 100% across 720 defined retrieval rows | Pass |
| Ours MRR | 100% across 720 defined retrieval rows | Pass |
| Ours complete-chain hit | 100% across 720 defined retrieval rows | Pass |

The 144-run historical result is the v1.1-scaled equivalent of the earlier
`6/6` Development requirement. B2 and B3 each retrieved a forbidden version in
all 288 governance runs because those baselines intentionally do not apply
version filtering. Ours retrieved none.

## Ours leads aggregate E1 and ties B1 on E2

Each accuracy below uses all successful repeated runs in its experiment: E1
contains 192 cases × 3 repeats per agent; E2 contains 96 cases × 3 repeats per
agent.

| Agent | E1 accuracy | E2 accuracy | E1 p50 latency | E2 p50 latency |
| --- | ---: | ---: | ---: | ---: |
| B0 | 25.00% | 0.00% | 512.6 ms | 526.2 ms |
| B1 | 74.13% | 100.00% | 534.5 ms | 528.8 ms |
| B2 | 84.90% | 90.28% | 541.7 ms | 566.3 ms |
| B3 | 82.29% | 96.53% | 600.1 ms | 598.9 ms |
| Ours | 93.40% | 100.00% | 558.5 ms | 543.6 ms |

The inferential comparison uses one median answer-F1 per case before paired
bootstrap resampling. On E1, B1 averaged 0.7703 and Ours 0.9640, for a +0.1938
delta with a 95% CI of [0.1376, 0.2517]. On E2 both methods averaged 1.0000,
so the delta and interval were exactly zero. These comparisons are valid
because the run has no terminal failures.

## Budget compliance passes, but Budget accuracy remains the largest weakness

All 288 comparable Budget rows (B1/B2/B3/Ours, 24 cases × 3 repeats) respected
the 80-token ceiling. The maximum post-budget prompt was 79 tokens. Compliance
therefore passes, but accuracy after truncation varies materially:

| Agent | Budget rows within limit | Maximum prompt | Accuracy |
| --- | ---: | ---: | ---: |
| B1 | 72 / 72 | 51 | 0.00% |
| B2 | 72 / 72 | 79 | 62.50% |
| B3 | 72 / 72 | 78 | 52.78% |
| Ours | 72 / 72 | 78 | 56.94% |

Ours was incorrect on 11 of 24 distinct Budget cases in at least one repeat.
Its other repeated errors were concentrated in three Multi-hop cases, where
run-level accuracy was 90.28%. The Ours prediction string varied across repeats
for three of 288 cases. This is a medium-confidence Development finding: it
supports freezing the current Test configuration if avoiding post-Development
overfitting is the priority, but it also identifies the exact areas for a later
method revision after the frozen Test decision.

### Why the frozen Ours score is only 56.94%

The strict score is `41 / 72 = 56.94%`, but retrieval failure is not the main
cause. The expected memory ranked first in all 72 Ours Budget runs. The 31
strictly incorrect rows decompose exactly into:

| Driver | Rows | Share of strict errors | Interpretation |
| --- | ---: | ---: | --- |
| Correct number plus Chinese measure word, such as `5条` or `5条。` | 22 | 70.97% | Semantically correct but rejected because the frozen target contains only `5` and no unit alias |
| `UNKNOWN` | 9 | 29.03% | Genuine answer failure despite the target memory being ranked first |

Under a post-hoc unit-tolerant sensitivity check that accepts `N`, `N条`, and
`N条。` as equivalent, Ours scores `63 / 72 = 87.50%`. This is diagnostic only,
not a replacement for the frozen official score. The same missing-unit-alias
effect appears in B2 (14 rows) and B3 (22 rows), so it is a benchmark/scoring
interaction rather than an Ours-only defect.

The result is strongly concentrated by question wording:

| Normalized question form | Strict accuracy | Unit-tolerant accuracy | Main behavior |
| --- | ---: | ---: | --- |
| `检索 Top-K 是多少？` | 12 / 12 | 12 / 12 | Bare number |
| `请给出设定的 Top-K？` | 12 / 12 | 12 / 12 | Bare number |
| `每次取回多少条候选记忆？` | 2 / 12 | 12 / 12 | Usually adds `条` |
| `Top-K 参数为何？` | 12 / 12 | 12 / 12 | Bare number |
| `固定检索几条结果？` | 0 / 12 | 12 / 12 | Always adds `条` |
| `检索数量上限？` | 3 / 12 | 3 / 12 | Nine `UNKNOWN` responses |

The wording pattern explains the entire 56.94% result: three templates produce
36/36 correct rows, the two measure-word templates contribute only 2/24 under
strict matching, and the `数量上限` template contributes 3/12.

Budgeting still has a measurable effect, but it does not remove the target
evidence. Before enforcing 80 tokens, strict Ours accuracy was 52/72 (72.22%);
after enforcement it was 41/72 (56.94%), a net decline of 11 rows or 15.28
percentage points. Forty rows were correct both times, 19 were incorrect both
times, 12 changed from correct to incorrect, and one changed from incorrect to
correct. Average prompt size fell from 88.67 to 76.26 tokens, while average
context fell from 43.00 to 30.60 tokens. Because the before/after values come
from separate DeepSeek calls and temperature zero is not guaranteed to be
deterministic, this comparison establishes prompt sensitivity, not that context
truncation alone caused every changed answer.

## Scope, data, and metric definitions

- Benchmark grain: one question over one synthetic conversation.
- Development population: 288 frozen cases, including 48 historical questions
  and 48 paired current-state questions.
- Execution grain: one case, one agent, one repeat.
- Agents: B0, B1, B2, B3, and Ours.
- Repeats: three, query-interleaved with alternating method order.
- LLM: `deepseek-v4-flash`, thinking disabled, maximum 64 output tokens.
- Embedding: local `.models/bge-small-zh-v1.5`, reused across cases.
- Retrieval: Top-K 5.
- E1 categories: fact recall, semantic recall, long context, noise,
  abstention, budget, multi-hop, and forgetting.
- E2 categories: update, conflict, and temporal.
- Accuracy: normalized benchmark answer match on each successful run.
- Formal paired comparison: per-case median across repeats, then 10,000 paired
  bootstrap samples with seed 202601.

## Integrity and execution checks

| Check | Evidence | Status |
| --- | ---: | --- |
| Expected execution rows | 4,320 / 4,320 | Pass |
| Unique `(case_id, agent, repeat)` keys | 4,320 / 4,320 | Pass |
| Per-agent coverage | 864 rows for every agent | Pass |
| Per-repeat coverage | 1,440 rows for every repeat | Pass |
| Terminal failures | 0 | Pass |
| API attempts | All 4,320 rows completed on attempt 1 | Pass |
| Secrets recorded | `false` | Pass |
| Frozen review enforced | `true` | Pass |
| Benchmark SHA-256 | `c32d229b3253b2e2a7a2167a9c4e061f8ad9f637779d3c1fb798952439a45ea8` | Match |

The run completed at 2026-09-01 14:33 China Standard Time. Exact tables are
used instead of charts because the acceptance decision depends on a compact
set of fixed gate values and five-agent lookups; a chart would not add material
interpretive value.

## Methodology

The formal runner first verified the frozen benchmark manifest and all bound
review artifacts. It then executed cases in query-interleaved alternating agent
order, wrote an append-only checkpoint after every row, and emitted raw rows,
experiment summaries, paired comparisons, environment metadata, and a terminal
run log. BGE weights were loaded once for the process; case-local caches
prevented cross-case state leakage. The acceptance checks reconciled raw-row
counts against `288 cases × 5 agents × 3 repeats`, verified composite-key
uniqueness, segmented results by category and agent, and independently checked
the historical, forbidden-retrieval, and Budget predicates.

## Limitations and robustness

- This is a synthetic, visible Development split and cannot provide a blind
  generalization estimate.
- Only one LLM snapshot, one embedding snapshot, one Top-K setting, and three
  repeats were evaluated.
- DeepSeek output showed minor nondeterminism despite temperature 0: Ours had
  three cases with different normalized predictions across repeats.
- The frozen Budget score conflates shortest-format adherence with semantic
  correctness. A response such as `5条` violates the requested bare shortest
  format but is a natural Chinese answer to `多少条`; whether it should count is
  a protocol-design choice, not a retrieval question.
- B1 matches Ours on E2 answers because full history plus the shared temporal
  explanation is sufficient on these questions. The differentiating evidence
  is Ours' zero forbidden retrieval, not E2 answer accuracy alone.
- Latency is endpoint- and load-dependent. The paired results reject the stated
  50% reduction target in this run but should not be generalized to other
  serving deployments.
- The prescribed portable HTML report could not be packaged because Node.js and
  npm are unavailable in the current runtime. This Markdown report is the
  durable technical handoff; raw JSON/JSONL artifacts remain the authoritative
  numerical source.

## Recommended next steps

1. Treat the pre-declared Development governance gate as passed and preserve
   the exact current configuration before any Test call.
2. Record a Git checkpoint containing this acceptance report; keep raw result
   artifacts outside Git if required by the repository's result policy.
3. Apply the decision in `answer_scoring_protocol_v1.2.md`: keep 56.94% as the
   frozen v1.1 result, create reviewed typed answer contracts for v1.2, and
   rerun Development before treating semantic scoring as official.
4. Obtain separate authorization before sending the frozen Test split to an
   external API. Run Test once in a new output directory without `--resume`.
5. Do not tune from Test. Reserve Holdout for the final evaluator-controlled
   run.
6. After the frozen Test decision, investigate short-answer normalization for
   Multi-hop and evidence allocation under the 80-token Budget as a separate
   method version rather than silently changing the evaluated configuration.

## Further questions

- Whether the current 56.94% Budget accuracy is acceptable for the competition
  objective or should become a v1.2 method-development target.
- Whether the Test run should use the protocol minimum of three repeats or the
  five-repeat main-result setting from the earlier experiment protocol.
- Whether a private evaluator-controlled Holdout location is available before
  final reporting.

Authoritative artifacts are in
`results/formal/e1_e2_v11_development_deepseek_bge_r3/`: `raw_rows.jsonl`,
`summary.json`, `manifest.json`, and `run.log`.
