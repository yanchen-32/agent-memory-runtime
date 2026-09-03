# Benchmark v1.2 results archive

The final Development + Test evidence bundle was created without rerunning or
rescoring either experiment.

## Artifact

- Local path:
  `results/archive/agent-memory-runtime_benchmark-v1.2_dev-test_20260903.tar.gz`
- Size: 1,695,206 bytes
- SHA-256:
  `d0395ddc3cb70d13eaf4f6eeb2a6acce2bf39289048015f07b0e0180aba2fca5`
- Evidence files covered by the internal checksum manifest: 37
- Holdout execution/results: absent; Holdout was not run
- Secrets: absent according to both run manifests and the pre-package scan

The bundle contains the final report, Development/Test acceptance reports,
frozen scoring and experiment protocols, v1.2 review/freeze lineage,
Development and Test benchmark files, and every machine-readable file produced
by both formal runs. `holdout.jsonl` is deliberately not packaged.

The archive itself is below the Git-ignored `results/` tree. Git stores this
record and `docs/benchmark_v12_results_checksums.sha256`, but not the binary
bundle. Preserve the `.tar.gz` and its adjacent `.sha256` file in the project's
durable artifact storage or attach them to a release before removing local
results.

## Verification

Verify the compressed artifact:

```bash
cd results/archive
sha256sum --check agent-memory-runtime_benchmark-v1.2_dev-test_20260903.tar.gz.sha256
```

Then extract it into a new directory and verify all 37 evidence files:

```bash
mkdir -p /tmp/benchmark-v12-verify
tar -xzf results/archive/agent-memory-runtime_benchmark-v1.2_dev-test_20260903.tar.gz \
  -C /tmp/benchmark-v12-verify
cd /tmp/benchmark-v12-verify/benchmark_v1.2_dev_test_final
sha256sum --check SHA256SUMS
```

The repository copy of the per-file list is
`docs/benchmark_v12_results_checksums.sha256`. The internal copy is authoritative
for verification after extraction; both were identical at packaging time.

## Provenance anchors

- Implementation/frozen-data checkpoint:
  `9080b03ef4bee36555bccc6ddda1e79235a4556f`
- Test acceptance checkpoint: `dfdc638`
- Development benchmark SHA-256:
  `46d856387fec4d77ffc99841407f73afe6dc820503e5b361bf262349e91775c6`
- Test benchmark SHA-256:
  `d20b522e90af47ce61e457cfaca2521438285df9d477ce16d97e89d0a6b9e9ea`
- Development/Test raw rows: 4,320 / 1,440
- Terminal failures: 0

The Development dirty-worktree provenance limitation and the single successful
Test API retry are documented in `docs/benchmark_v12_final_report.md` rather
than hidden by the archive process.
