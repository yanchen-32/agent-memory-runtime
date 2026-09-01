# Answer Scoring Protocol v1.2

Status: scoring design frozen; Benchmark v1.2 data not yet reviewed or frozen.

## Decision

Budget questions measure whether a system can recover the correct fact while
obeying a context ceiling. Semantic correctness and presentation style are
therefore separate outcomes. A natural Chinese measure word such as `条` must
not turn a correct scalar value into a factual error, while verbose, ambiguous,
or approximate responses must not pass merely because they contain the target
number.

This is a material change to the primary metric, so it belongs to experiment
protocol v1.2. It does not alter Benchmark v1.1, its frozen hash, or its official
56.94% Ours Budget accuracy. The v1.1 unit-tolerant 87.50% result remains a
post-hoc sensitivity analysis until a reviewed v1.2 Development run is made.

## Versioned answer contract

Every v1.2 scalar-quantity case must declare the accepted semantics before any
model output is observed:

```json
{
  "expected_answer": "5",
  "answer_spec": {
    "type": "quantity",
    "canonical_value": "5",
    "value_aliases": [],
    "units": ["条"],
    "unit_policy": "optional",
    "output_format": "bare_value",
    "scorer_version": "quantity-semantic-v1"
  }
}
```

`canonical_value`, `value_aliases`, `units`, and `unit_policy` are
benchmark-owned annotations. They may not be inferred from predictions or
expanded after viewing Development, Test, or Holdout outputs. Chinese numerals,
alternative units, ranges, and approximations are invalid unless explicitly
declared in the case before freezing.

Cases without `answer_spec` retain `legacy-answer-v1`. This compatibility rule
allows exact reproduction of v1.0 and v1.1; a formal v1.2 split must not mix
undeclared scoring semantics within its Budget category.

## Deterministic quantity semantics

The `quantity-semantic-v1` scorer applies Unicode NFKC normalization, ignores
whitespace and terminal `.?!。！？`, and permits only the existing frozen answer
prefixes such as `答案是` or `回答：`. It then compares the entire remaining
response with a predeclared `value` or `value + unit` form.

For the example above:

| Prediction | Semantic correct | Format compliant | Reason |
| --- | ---: | ---: | --- |
| `5` | Yes | Yes | Canonical shortest value |
| `5。` | Yes | Yes | Terminal punctuation only |
| `5条` | Yes | No | Correct value and declared optional unit |
| `答案是：5条。` | Yes | No | Correct semantics, non-minimal presentation |
| `大约5条` | No | No | Approximation was not declared |
| `5或6` | No | No | Ambiguous range/disjunction |
| `Top-K为5` | No | No | Extra proposition rather than the answer form |
| `答案是5条，因为配置如此` | No | No | Explanation is outside the answer contract |
| `UNKNOWN` | No | No | Unsupported-answer response |

Full-response matching is intentional. Substring matching, LLM-as-judge, and
prediction-derived aliases are prohibited for this scalar task.

## Metrics

For each post-budget run, record all of the following rather than collapsing
them into an arbitrarily weighted score:

- **Budget Semantic Accuracy**: mean of the typed semantic-correct indicator.
- **Budget Constraint Satisfaction**: mean of `prompt_tokens <= token_budget`.
- **Budget Task Success**: mean of `semantic_correct AND budget_satisfied`.
  This is the primary Budget outcome.
- **Answer Format Compliance**: mean of the shortest-output-format indicator.
  This is a secondary quality metric, not factual accuracy.
- **Strict Answer Accuracy**: legacy normalized answer-clause match, retained
  only for reproducibility and sensitivity comparison.
- **Answer F1**: existing lexical token-overlap F1, retained as a diagnostic;
  it is not the primary metric for scalar quantities.
- **Expected-memory hit and forbidden retrieval**: retrieval diagnostics kept
  separate from generation correctness.
- **Token reduction and semantic accuracy delta**: before/after Budget effects,
  reported separately. Because these require two model calls, they measure
  prompt sensitivity and may include endpoint nondeterminism.

The report must include numerators, denominators, case count, repeat count,
scorer version, and benchmark hash. A system cannot compensate for exceeding
the token ceiling with a correct answer: that run has Semantic Accuracy 1 but
Budget Task Success 0.

## Repeats and uncertainty

Run-level rates are descriptive. Formal method comparisons first aggregate
each `case × agent` across repeats with the median (equivalent to majority for
binary outcomes with an odd repeat count), then use paired case bootstrap with
10,000 resamples and the protocol seed to report the difference and 95%
confidence interval. Cases, not repeated calls, are the independent sampling
units.

Category-level results must accompany the aggregate result. Failures are
classified as token-limit failure, retrieval/evidence failure, semantic answer
failure, or format-only violation. A format-only violation must never be
reported as a semantic failure.

## Promotion rule

Before Test, create a deterministic Benchmark v1.2 candidate, add the typed
contract to every Budget case, validate it, complete the existing AI
pre-review plus one human sign-off, and freeze new hashes. Rerun all Development
methods in a new output directory with no v1.1 checkpoint resume. Only this new
run may publish `quantity-semantic-v1` as an official result. Test and Holdout
remain untouched until that Development artifact passes integrity checks and
the already declared temporal/forbidden-retrieval gates.
