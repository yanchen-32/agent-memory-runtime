# E3 v1.4 orthogonal design (not a benchmark candidate)

This directory contains only a pre-registered scenario design. It deliberately
contains no questions, answers, Development JSONL, Test, or Holdout data.

The design is the complete `6 predicates x 8 query templates x 3 target
positions` factorial: 144 independent scenario families and 576 planned cases
after applying the four prefix-nested history strata.

Semantic case generation is deferred until the v1.3 Development-only ablation
selects and locks a method. This prevents the successor data from becoming an
additional tuning set before its construction rules are fixed.

Regenerate the deterministic design with:

```bash
python -m benchmark.design_e3_v14
```
