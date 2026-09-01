# Benchmark v1.2 Development Acceptance

## Decision

The frozen Benchmark v1.2 Development run completed successfully and passes
the predeclared temporal-governance gates. Ours achieved 87.50% Budget Semantic
Accuracy and 87.50% Budget Task Success while satisfying the 80-token ceiling
on every Budget run. This replaces format-sensitive strict accuracy as the
primary Budget interpretation for v1.2; it does not rewrite the official v1.1
result.

The Development evidence is sufficient to promote the frozen v1.2 protocol to
a separately authorized Test run. Test and Holdout were not executed here.

## Execution integrity

| Check | Evidence | Result |
| --- | ---: | --- |
| Expected rows | 4,320 / 4,320 | Pass |
| Unique `(case_id, agent, repeat)` keys | 4,320 / 4,320 | Pass |
| Per-agent coverage | 864 rows each | Pass |
| Terminal failures | 0 | Pass |
| API retries | 0 | Pass |
| Run status | `completed` | Pass |
| Frozen benchmark enforcement | `true` | Pass |
| Development SHA-256 | `46d856387fec4d77ffc99841407f73afe6dc820503e5b361bf262349e91775c6` | Match |
| Secrets recorded | `false` | Pass |

The run used DeepSeek `deepseek-v4-flash`, thinking disabled, maximum 64 output
tokens, local BGE small Chinese v1.5, Top-K 5, three repeats, and
query-interleaved alternating agent order. Results are stored under
`results/formal/e1_e2_v12_development_deepseek_bge_r3`.

## Budget v1.2 result

All agents respected the 80-token constraint in all 72 Budget runs. Typed
semantic correctness, shortest-format compliance, and joint task success are
reported separately:

| Agent | Semantic correct | Strict / format compliant | Within budget | Budget Task Success | Max prompt |
| --- | ---: | ---: | ---: | ---: | ---: |
| B0 | 0 / 72 (0.00%) | 0 / 72 (0.00%) | 72 / 72 | 0 / 72 (0.00%) | 49 |
| B1 | 0 / 72 (0.00%) | 0 / 72 (0.00%) | 72 / 72 | 0 / 72 (0.00%) | 51 |
| B2 | 58 / 72 (80.56%) | 43 / 72 (59.72%) | 72 / 72 | 58 / 72 (80.56%) | 79 |
| B3 | 60 / 72 (83.33%) | 39 / 72 (54.17%) | 72 / 72 | 60 / 72 (83.33%) | 78 |
| Ours | 63 / 72 (87.50%) | 42 / 72 (58.33%) | 72 / 72 | 63 / 72 (87.50%) | 78 |

For Ours, the expected memory ranked first in all 72 runs. The remaining nine
semantic failures are exactly three cases repeated three times:
`v11_dev_budget_006`, `v11_dev_budget_012`, and `v11_dev_budget_018`. All use
the question form `请查找…的检索数量上限？` and all nine predictions are
`UNKNOWN`. These are genuine generation/evidence-use failures under the typed
contract, not unit-format errors.

At the formal per-case median grain, Ours succeeded on 21/24 Budget cases
(87.50%) versus 0/24 for B1. The paired bootstrap delta is +87.50 percentage
points with a 95% interval of [75.00, 100.00] points. The same result holds for
Budget Task Success because every compared run met the token ceiling.

The v1.1 Ours result remains officially 56.94% under its frozen strict scorer.
The independent v1.2 rerun reproduces the earlier semantic sensitivity result
of 87.50%, now under a predeclared, reviewed, and frozen scoring contract.

## Aggregate E1 and E2

| Agent | E1 accuracy | E2 accuracy | E1 p50 | E2 p50 |
| --- | ---: | ---: | ---: | ---: |
| B0 | 25.00% | 0.00% | 654.2 ms | 645.5 ms |
| B1 | 74.31% | 100.00% | 676.0 ms | 659.4 ms |
| B2 | 87.15% | 90.63% | 674.4 ms | 624.7 ms |
| B3 | 86.46% | 96.53% | 752.1 ms | 695.5 ms |
| Ours | 97.74% | 100.00% | 670.4 ms | 626.3 ms |

The formal E1 paired answer-F1 comparison aggregates repeats by case median.
Ours scored 0.9705 versus 0.7703 for B1, a +0.2002 delta with 95% bootstrap
interval [0.1458, 0.2558]. E2 was tied at 1.0. Ours was 0.96% slower than B1
on paired E1 end-to-end latency and 1.53% slower on E2, so the historical 50%
latency-reduction target remains unmet.

## Temporal and forbidden-retrieval gates

| Gate | Evidence | Result |
| --- | ---: | --- |
| Ours historical answers | 144 / 144 | Pass |
| Ours historical retrieval | 144 / 144 | Pass |
| Ours forbidden retrieval on Update/Conflict/Temporal | 0 / 288 rows | Pass |
| Ours forbidden retrieval including Forgetting | 0 / 360 rows | Pass |
| Ours defined Recall@1 / Recall@5 / MRR | 95% / 100% / 100% | Pass |
| Ours complete-chain hit | 100% | Pass |

B2 and B3 intentionally remain without version filtering and retrieved a
forbidden version on every defined governance run. Their answer accuracy is
therefore not evidence of memory-governance compliance.

## Remaining errors and limitations

- Ours' only Budget semantic failures are the three stable `数量上限` cases.
  Retrieval is correct, so the next method investigation should target prompt
  interpretation or evidence allocation rather than BGE recall.
- Ours has four run-level Multi-hop errors. Three are the semantically natural
  `openEuler + ARM64` against frozen `openEuler ARM64`; the remaining verbose
  response occurs once. Multi-hop still uses `legacy-answer-v1`, so this is a
  separate future scoring-contract question and must not be patched into v1.2
  after seeing these outputs.
- Latency remains endpoint- and load-dependent and does not show a method
  advantage in this run.
- The run manifest records base Git commit `c03023a`; the v1.2 implementation
  was an uncommitted worktree at launch. Exact frozen benchmark and scoring-tool
  hashes are recorded in `frozen_manifest.json`, but a clean Git checkpoint is
  required before any Test run.
- Test/Holdout candidates are repository-visible and are not equivalent to an
  evaluator-secret blind set.

## Promotion requirements

Before Test, create a clean Git checkpoint of the exact v1.2 scorer, runner,
frozen data, and this acceptance record. Test must use the same frozen hashes,
model parameters, prompt version, and scorer version; it must run once in a
new output directory without `--resume`. No prompt, alias, answer-spec, or
method tuning may be derived from Test output. Holdout remains reserved for an
evaluator-controlled final run.
