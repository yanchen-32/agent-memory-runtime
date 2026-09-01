# Benchmark v1.2 Scoring Migration

Benchmark v1.2 is a controlled scoring-only migration from frozen v1.1. The
480 questions, conversations, expected answers, evidence IDs, forbidden IDs,
timestamps, split membership, and case IDs remain unchanged. Only versioned
lineage metadata and typed `answer_spec` declarations on the 40 Budget cases
may differ.

The scoring contract is `quantity-semantic-v1`, defined in
`docs/answer_scoring_protocol_v1.2.md`. It separates semantic correctness from
shortest-answer format compliance and defines Budget Task Success as semantic
correctness under the token ceiling.

Regenerate and audit deterministically:

```bash
python -m benchmark.generate_v12
python -m benchmark.prereview_v12
python -m benchmark.audit_v12
python -m benchmark.validate_v12 benchmark/data/v1.2
```

The 480 checklist rows are automatic technical evidence. After inspecting
`deep_audit.md`, the human reviewer signs exactly once by changing only these
two values in `review_signoff.json`:

```json
"reviewer": "your-name",
"decision": "approved"
```

Then freeze:

```bash
python -m benchmark.freeze_v12
```

Formal Development must use a new output directory and must not use `--resume`.
Test and Holdout remain untouched until the Development result is accepted.
