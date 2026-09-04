# E3 v1.4 Development candidate

This directory contains the generated Development candidate and its bound
technical-review package. Test and Holdout data deliberately do not exist yet.

The design is the complete `6 predicates x 8 query templates x 3 target
positions` factorial: 144 independent scenario families and 576 planned cases
after applying the four prefix-nested history strata.

The query dimension includes canonical predicates, schema-level predicate
aliases, and nonliteral relation phrases. Every versioned agent prompt receives
`QUERY_TIME`, `VALID_FROM`, and `VALID_TO`. The four histories in a family are
strict prefix extensions and are stratified by the resulting B1 prompt tokens.

Regenerate the deterministic design with:

```bash
python -m benchmark.generate_e3_v14
python -m benchmark.validate_e3_v14
python -m benchmark.review_e3_v14 prereview
python -m benchmark.review_e3_v14 audit
```

After reviewing the reports, the human reviewer fills only `reviewer` and sets
`decision` to `approved` in `review_signoff.json`, then freezes once with:

```bash
python -m benchmark.freeze_e3_v14
```

Do not generate or inspect Test until the frozen Development run passes every
predeclared admission gate.
