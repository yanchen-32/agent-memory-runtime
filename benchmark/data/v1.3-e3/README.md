# Benchmark v1.3-E3 Development Candidate

Status: `pending_human_review`

The predeclared measurement and admission rules are in
`docs/experiment_protocol_v1.3_e3.md`.

This candidate isolates long-history scale. It contains 48 matched scenario
families and four prefix-nested variants per family, for 192 Development cases:

| Stratum | Target unconstrained B1 prompt tokens | Cases |
| --- | ---: | ---: |
| Short | 1,000 | 48 |
| Medium | 4,000 | 48 |
| Long | 16,000 | 48 |
| Very Long | 32,000 | 48 |

The independent families jointly cover six predicates, eight question
templates, front/middle/back target positions, and four distractor types.

Within a family, the question, current target fact, answer, expected memory ID,
and stale forbidden memory ID are identical. A larger stratum is a strict
prefix extension of the smaller history: it only appends chronologically later
distractor memories. The expected answer occurs only in the target memory.
Cases use scenario-major ordering, so every four adjacent cases cover all four
strata; this reduces confounding between context scale and endpoint load drift.

Prompt size is measured by constructing the actual unconstrained B1 prompt and
using the project's deterministic prompt-token counter. It is not inferred
from character count. Provider-reported API prompt tokens remain a separate
runtime measurement.

Regenerate and validate:

```bash
python -m benchmark.generate_e3_v13
python -m benchmark.validate_e3_v13
python -m benchmark.prereview_e3_v13
python -m benchmark.audit_e3_v13
```

The last two commands complete the case-level technical review and recreate a
single hash-bound `review_signoff.json`. The human reviewer fills `reviewer`
and changes `decision` to `approved` once, then runs:

```bash
python -m benchmark.freeze_e3_v13
```

The generated files are Development candidates only. No Test or Holdout split
is generated, and candidate results are not formal evidence. After technical
and human review, a separate freeze checkpoint is required before a formal
DeepSeek/BGE run.

An explicitly labelled candidate pilot may be run with:

```bash
python experiments/run_e3_scaling.py \
  --allow-unreviewed-benchmark \
  --output-dir results/pilot/e3_v13_development_rule \
  --agents B0,B1,B2,B3,Ours \
  --repeats 3
```

The formal runner reports each stratum separately. E3 admission requires all
of the following:

- the paired Answer F1 bootstrap lower bound for Ours minus B1 is non-negative
  in every stratum;
- the relative spread of Ours' stratum-level case-median context-token means is
  at most 10%;
- Ours has zero forbidden retrievals and zero terminal failures; and
- paired Very Long end-to-end latency reduction versus B1 reaches 50%.

Only the last gate authorizes the original 50% latency claim.
