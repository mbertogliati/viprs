# PR #60 Benchmark Audit

Date: 2026-06-21
Branch audited: `workspace-split`
Merged commit: `0b12ed74293863fd2354a29e647c42f5614b4f7c`

## Verdict

PR #60 did not receive a valid Criterion comparison against a cached `main` baseline during
review. Both architecture jobs restored `target/criterion` with the expected
`criterion-baseline-*-main` cache keys, but both cache restores missed and the workflow fell
back to `make bench-ci`, which is a smoke run.

The `viprs vs libvips` E2E jobs did run successfully on both architectures, but their current
gate treats ratios from `1.05` to `5.0` as warnings only. Therefore performance gaps were
reported, not failed.

## Evidence

Final PR run:

- Run: `27888583912`
- URL: `https://github.com/mbertogliati/viprs/actions/runs/27888583912`
- Head SHA: `8b43a43b58c76b13f72fa6ccb41b9d59417d23b5`
- Result: failed because coverage failed, not because benchmark jobs failed.

Criterion benchmark jobs:

- `Benchmarks (x86_64)`: job `82529946166`
  - Restored cache key: `criterion-baseline-x86_64-main`
  - Result: `Cache not found for input keys: criterion-baseline-x86_64-main`
  - Executed path: `No baseline - running smoke test (first run)` -> `make bench-ci`
- `Benchmarks (arm64)`: job `82529952912`
  - Restored cache key: `criterion-baseline-arm64-main`
  - Result: `Cache not found for input keys: criterion-baseline-arm64-main`
  - Executed path: `No baseline - running smoke test (first run)` -> `make bench-ci`

`viprs vs libvips` jobs:

- `Performance: viprs vs libvips (x86_64)`: job `82529946030`
  - E2E suite completed successfully.
  - Uploaded artifact ID: `7770353639`
  - Ratio summary: `7` operations in warning zone `(1.05, 5.0]`, `0` failures `> 5.0`.
- `Performance: viprs vs libvips (arm64)`: job `82529954126`
  - E2E suite completed successfully.
  - Uploaded artifact ID: `7770348975`
  - Ratio summary: `6` operations in warning zone `(1.05, 5.0]`, `0` failures `> 5.0`.

## Missed Regression Risk

Criterion did not protect PR #60 against regressions relative to pre-merge `main`, because no
baseline existed in cache. The benchmark job name stayed green because `make bench-ci` only
verifies that benchmark binaries compile and execute for the filtered `512` scenarios.

`bench-vs-libvips` did protect against catastrophic slowdowns only. It did not enforce the
documented target ratio of `<= 1.05`; warnings were emitted but did not fail the job because
`.github/scripts/check-ratios.sh` fails only at ratio `> 5.0`.

## Recommended Follow-ups

1. Generate the Criterion baseline from the PR base SHA inside the PR job before comparing the PR
   head, so correctness never depends on a cache hit.
2. Keep the `main` baseline cache only as an optimization for push/scheduled runs, not as the
   source of truth for PR regression checks.
3. Change `check-ratios.sh` to fail at `> 1.05`, or keep the warning policy but rename the gate
   so it does not imply no regressions.
4. Remove `continue-on-error: true` from `bench-vs-libvips` once known gaps are tracked as issues.
5. Publish benchmark artifacts and summaries in PR comments so warnings cannot be missed in green
   job lists.

## Fix Applied In This Branch

The `Benchmarks` job now handles pull requests differently from `main` pushes:

1. Fetch and checkout `${{ github.event.pull_request.base.sha }}`.
2. Run `make bench-baseline` to save the `main` baseline locally in `target/criterion`.
3. Fetch and checkout `${{ github.event.pull_request.head.sha }}`.
4. Run `make bench-compare`.

This means a PR benchmark run cannot silently downgrade to smoke mode because of a missing
`criterion-baseline-*-main` cache.
