# Benchmark v1.1 Expansion Plan

`v1.0` remains frozen and is used only as a small regression suite.  `v1.1`
is a new, independently reviewed benchmark; no v1.0 file is amended.

## Composition target

The first full v1.1 candidate release has 480 project-owned AMR-CN cases:

| split | cases | governance pairs | pair cases |
| --- | ---: | ---: | ---: |
| Development | 288 | 48 | 96 |
| Test | 96 | 16 | 32 |
| Holdout | 96 | 16 | 32 |

The remaining 320 cases cover fact recall, semantic recall, multi-hop,
abstention, noise, budget, forgetting, and long-context recall.  The target is
40 cases per capability across all three splits.  A governance pair always has
one current-state and one historical-state question over the same two memory
versions.

The checked-in candidate files now meet this target.  The 480-case count is
deliberately counted in *independent cases* rather than
repeats: repeated calls measure serving variability but do not enlarge the
benchmark sample.

## Source policy

There are two distinct evaluation tracks.  Their scores must never be pooled.

1. **AMR-CN governance v1.1 (project-owned).**  Authors create independent
   Chinese scenarios from a capability specification.  LongMemEval may inform
   category balance and difficulty, but upstream conversations, questions,
   answers, entity names, and evidence text must not be translated or
   paraphrased into this track.
2. **External immutable tracks.**  Upstream examples are retained byte-for-byte
   through a lossless adapter, with their source ID, revision, hash, licence and
   evaluation protocol.  They are reported separately and never used for
   parameter selection after the split is designated Test or Holdout.

This distinction lets the project demonstrate Chinese version-governance
behavior without falsely describing transformed third-party annotations as
original labels.

## Required provenance for every AMR-CN case

Each `metadata` object must contain:

```json
{
  "benchmark_version": "1.1-candidate",
  "source_type": "project_owned",
  "derivation_type": "independent_authoring_from_capability_spec",
  "scenario_family": "unique-family-id",
  "split": "development",
  "author": "author-id",
  "review_status": "pending_human_review"
}
```

For Update/Conflict pairs, additionally record `pair_id`, `query_mode`
(`current` or `historical`), `subject`, `predicate`, and version timestamps.
The two memory records must share `subject + predicate`, have strictly
increasing `valid_from`, and identify the non-answering version in
`forbidden_memory_ids`.

## Split and leakage rules

- Split by `scenario_family`, not individual question.  A family, entities,
  facts, or template parameters may occur in exactly one split.
- Allocate a family before authoring its questions; Test and Holdout remain
  unread during Development work.
- Use a deterministic seed manifest to allocate families and preserve it in
  Git before generation.
- Run structure validation, automatic duplicate/similarity audit, then human
  review.  One completed package-level signature authorizes `freeze_v11`; its
  bound hashes cover all 480 technical-review rows and all three splits.

## External tracks

- **LongMemEval-S Cleaned:** primary long-term-memory external evaluation.
  The repository already has a lossless adapter and a pinned 500-question
  source record in `benchmark/data/v1.0/source_manifest.json`.  Reuse that
  exact revision for the initial external track rather than making a new
  translated copy.
- **LongBench v1 Chinese tasks:** downloaded E3 long-context stress track.  The
  five official Chinese task files contain 1,000 rows and retain their original
  metric and provenance; see `source_manifest.json` for the pin and component
  notice requirement.
- **RULER:** downloaded synthetic E3 length-stress generator.  It does not test
  version governance and cannot substitute for AMR-CN pairs; generated data is
  versioned by configuration, sequence length, tokenizer and seed.
- **LoCoMo:** excluded from this release.  Its CC BY-NC 4.0 data terms are not
  appropriate for a potentially commercial competition deliverable.

`source_manifest.json` records this admission decision.  A source is not
downloaded or included until its exact upstream revision, file hash, and data
licence have been added to that manifest.

After downloading the sources to the ignored `.datasets/` directory, verify
their exact local state with:

```bash
python -m benchmark.verify_external_v11 --workspace .
```

Regenerate, prereview and deep-audit the candidate splits with:

```bash
python -m benchmark.generate_v11 --output-dir benchmark/data/v1.1 --seed 20260831
python -m benchmark.prereview_v11 --data-dir benchmark/data/v1.1
python -m benchmark.audit_v11 --data-dir benchmark/data/v1.1
python -m benchmark.validate_v11 benchmark/data/v1.1
```

After the technical checklist and deep-audit report have been inspected, the
human reviewer signs the complete 480-case package **once**.  Open
`review_signoff.json` and change only these two values:

```json
"reviewer": "your-name",
"decision": "approved"
```

Do not edit 480 rows in `review_checklist.csv`; those rows are the automatic
case-level evidence.  The one signature is cryptographically bound to that
checklist, all three split hashes, the deep-audit hash and the 480-case scope.
Then freeze all three splits with:

```bash
python -m benchmark.freeze_v11 --data-dir benchmark/data/v1.1
```

The freeze command rejects a missing one-time signature, failed technical
fields, changed checklist/split hashes, a stale deep audit, and
candidate-manifest mismatches.

## Candidate confidentiality

The checked-in Test and Holdout files are **candidates**, not sealed scores:
their content is visible to every repository reader.  Do not select prompts,
parameters, or architecture from them.  Before a competition-facing Holdout is
claimed, move its final content to a private evaluator-controlled location and
publish only its frozen hash and evaluation artifacts after the run.
