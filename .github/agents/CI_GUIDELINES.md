# GitHub Actions — CI Guidelines for PRs

Principles: modular checks, shared cache, explicit dependencies, minimum duration, and clear visibility in the PR.

---

## 1. File structure

```
.github/
├── workflows/
│   ├── ci.yml                  # main PR workflow — single entry point
│   └── _reusable-<name>.yml    # reusable workflows (on: workflow_call)
├── actions/
│   └── <name>/
│       └── action.yml          # composite actions (setup, cache, tooling)
└── scripts/
    └── <name>.sh               # bash extracted from inline YAML
```

**Rules:**
- One CI workflow per PR. Multiple workflows on the same PR require multiple required checks — hard to maintain and prone to race conditions.
- Prefix reusable workflows with `_` to distinguish them from trigger workflows.
- Never write more than 5 lines of bash inline in the YAML: move it to `scripts/`.
- Clippy must run with **all** supported CI features enabled. Running clippy with partial features hides errors in feature-gated code.

---

## 2. Internal CI workflow structure

All CI jobs must follow this layered scheme:

```
[setup] → [lint] ──┐
                   ├──→ [test] ──┐
[setup] → [build] ─┘             ├──→ [status-gate]
                                 │
[setup] → [security] ────────────┘
```

```yaml
name: CI

on:
  pull_request:
    branches: [main, develop]
  merge_group:           # required if you use merge queue

concurrency:
  group: ci-${{ github.ref }}
  cancel-in-progress: true   # cancel previous runs on the same PR

permissions:
  contents: read             # minimum permissions — always declare explicitly

jobs:

  # ── Layer 1: independent, run in parallel ───────────────────────────────

  lint:
    name: "Lint"
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: ./.github/actions/setup-toolchain   # composite action with cache
      - run: ./scripts/lint.sh

  build:
    name: "Build"
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: ./.github/actions/setup-toolchain
      - run: ./scripts/build.sh

  security:
    name: "Security scan"
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - run: ./scripts/security-scan.sh

  # ── Layer 2: depend on lint + build ─────────────────────────────────────

  test:
    name: "Tests"
    needs: [lint, build]       # does not run if lint or build failed
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: ./.github/actions/setup-toolchain
      - run: ./scripts/test.sh

  # ── Layer 3: single gate — the only required check in Branch Rules ───────

  status-gate:
    name: "CI passed"
    needs: [lint, build, security, test]
    if: always()               # always runs, even if others failed
    runs-on: ubuntu-latest
    steps:
      - name: Check results
        run: |
          results='${{ toJSON(needs.*.result) }}'
          if echo "$results" | grep -qE '"failure"|"cancelled"'; then
            echo "❌ One or more checks failed."
            exit 1
          fi
          echo "✅ All checks passed."
```

---

## 3. Cache — principles

### 3.1 Two distinct problems

Caching in CI solves two separate problems that require different mechanisms:

- **Dependency download** — fetching packages from the registry (crates.io, npm, etc). Solved with `actions/cache` on the local registry directories. The cache key is the lockfile hash: if dependencies didn't change, the cache is always hit regardless of source changes.
- **Dependency compilation** — building third-party crates/packages from source. Solved either by including the build output directory (`target/`, `node_modules/.cache`, etc.) in the cache, or by passing a pre-built artifact between jobs. These are different strategies with different tradeoffs — see §3.3.

Never conflate the two. A registry cache hit means no downloads; it does not mean no compilation.

### 3.2 Cache declaration

Cache must be declared in a **shared composite action**, not duplicated across each job. This ensures all jobs use the same key scheme and the same paths.

```yaml
# .github/actions/setup-toolchain/action.yml
name: Setup toolchain
description: Install dependencies with shared cache across jobs

runs:
  using: composite
  steps:
    - name: Install toolchain
      uses: dtolnay/rust-toolchain@stable
      with:
        components: rustfmt, clippy
      shell: bash

    - name: Cache registry
      uses: actions/cache@v4
      with:
        path: |
          ~/.cargo/registry/index
          ~/.cargo/registry/cache
          ~/.cargo/git/db
        key: ${{ runner.os }}-cargo-registry-${{ hashFiles('**/Cargo.lock') }}
        restore-keys: |
          ${{ runner.os }}-cargo-registry-
      shell: bash
```

**Key rules:**
- The cache `key` uses only the lockfile hash — never the source hash. Source changes must not invalidate the dependency cache.
- `restore-keys` provides a partial fallback: if the exact key misses (e.g. a new dependency was added), the closest previous cache is restored and only the delta is downloaded.
- Separate registry cache from build output cache — they have different invalidation patterns and different sizes.
- Never cache outputs that depend on your own source code in the shared toolchain action. That belongs in job-level cache or artifacts.

### 3.3 Avoiding redundant compilation across jobs

Each job runs on a clean ephemeral runner. Even with a registry cache hit, every job recompiles dependencies from source unless the compiled output is explicitly shared. There are two strategies:

**Strategy A — Cache the build output directory**

Include the build output directory in the cache with a key scoped only to the lockfile. When the lockfile hasn't changed, the compiled dependency artifacts are restored and the build tool only recompiles your own code.

```yaml
- name: Cache build output
  uses: actions/cache@v4
  with:
    path: target/
    key: ${{ runner.os }}-cargo-target-${{ hashFiles('**/Cargo.lock') }}
    restore-keys: |
      ${{ runner.os }}-cargo-target-
```

Appropriate when the build output directory is small enough that cache upload/restore is faster than recompiling dependencies. Measure before committing to this — for large compiled outputs (> ~1 GB), the transfer cost can exceed the compilation cost.

**Strategy B — Dedicated `deps` job with artifact passing**

Add a first-layer job whose sole responsibility is compiling dependencies and uploading the build output as an artifact. Downstream jobs download the artifact instead of recompiling.

```yaml
deps:
  name: "Compile dependencies"
  runs-on: ubuntu-latest
  steps:
    - uses: actions/checkout@v4
    - uses: ./.github/actions/setup-toolchain
    - name: Compile
      run: cargo build --release
    - name: Compress and upload build output
      run: tar -czf target.tar.gz target/
    - uses: actions/upload-artifact@v4
      with:
        name: cargo-target-${{ github.sha }}
        path: target.tar.gz
        compression-level: 0   # already compressed
        retention-days: 1

lint:
  needs: [deps]
  steps:
    - uses: actions/download-artifact@v4
      with:
        name: cargo-target-${{ github.sha }}
    - run: tar -xzf target.tar.gz
    - run: cargo clippy -- -D warnings
```

Appropriate when dependency compilation is the dominant cost and the compressed artifact transfers faster than recompiling. Compressing before upload typically reduces artifact size by 50–70% for compiled outputs.

**Choosing between strategies:**

| Dependency compile time | Compressed artifact size | Recommended |
|---|---|---|
| < 1 min | any | Registry cache only — no added complexity |
| 1–3 min | < 500 MB | Strategy A — build output cache |
| > 3 min | any | Strategy B — dedicated deps job + artifact |
| > 3 min | > 1 GB | Strategy B + distributed cache (sccache + S3/GCS) |

Measure actual job durations before optimizing. The right strategy depends on the specific ratio of compile time to transfer time in your pipeline.

---

## 4. Visibility in the PR

### 4.1 Job and step names

Every job and step must be readable in the PR UI without additional context.

```yaml
# ✅ Good
jobs:
  test:
    name: "Tests — unit + integration (ubuntu)"
    steps:
      - name: "Compile test binary"
      - name: "Run unit tests"
      - name: "Run integration tests"
      - name: "Upload coverage report"

# ❌ Avoid
jobs:
  test:
    steps:
      - run: cargo test
```

### 4.2 Readable error output

Every script that can fail must print context before exiting:

```bash
# scripts/test.sh
set -euo pipefail

echo "▶ Running tests..."
cargo test 2>&1 | tee test-output.txt
exit_code=${PIPESTATUS[0]}

if [ $exit_code -ne 0 ]; then
  echo ""
  echo "❌ Tests failed. Summary:"
  grep -E "^(FAILED|error)" test-output.txt | head -20
  exit $exit_code
fi

echo "✅ Tests passed."
```

### 4.3 Diagnostic artifacts

Always upload artifacts on failure so you can diagnose without rerunning:

```yaml
- name: Upload logs on failure
  if: failure()
  uses: actions/upload-artifact@v4
  with:
    name: test-logs-${{ github.run_id }}
    path: |
      test-output.txt
      coverage/
    retention-days: 7
```

### 4.4 Job summaries

Add a summary so it appears in the workflow run's "Summary" tab:

```yaml
- name: Generate summary
  if: always()
  run: |
    echo "## Test results" >> $GITHUB_STEP_SUMMARY
    echo "" >> $GITHUB_STEP_SUMMARY
    echo "| Suite | Result |" >> $GITHUB_STEP_SUMMARY
    echo "|-------|--------|" >> $GITHUB_STEP_SUMMARY
    echo "| Unit  | ${{ steps.unit.outcome }} |" >> $GITHUB_STEP_SUMMARY
    echo "| Integration | ${{ steps.integration.outcome }} |" >> $GITHUB_STEP_SUMMARY
```

---

## 5. Individual check re-runs

To allow granular re-runs without rerunning the full workflow:

### 5.1 Split long suites into separate jobs

If tests take more than 3 minutes, split them into distinct jobs with correct `needs`:

```yaml
test-unit:
  name: "Tests — unit"
  needs: [lint]
  ...

test-integration:
  name: "Tests — integration"
  needs: [build]
  ...

test-e2e:
  name: "Tests — e2e"
  needs: [test-unit, test-integration]
  ...
```

If e2e fails, you can re-run just that job from the UI without re-running lint/build.

### 5.2 Re-run from the UI

In the PR checks tab, GitHub allows:
- **Re-run failed jobs** — only the failed jobs, reusing the cache.
- **Re-run all jobs** — full workflow.

For a single-job re-run to be useful, the job must be self-contained with its own setup and cache — which is why the shared composite action matters.

---

## 6. Job dependencies — reference

| Situation | Pattern |
|---|---|
| Job B needs something A compiled | `needs: [A]` |
| Job B can run if A was skipped but not if it failed | `needs: [A]` + `if: needs.A.result != 'failure'` |
| Gate job that evaluates all others | `needs: [A, B, C]` + `if: always()` |
| Independent jobs that can parallelize | no `needs` — run in parallel by default |

```yaml
# Example: test runs if lint passed or was skipped, but not if it failed
test:
  needs: [lint]
  if: needs.lint.result != 'failure'
```

---

## 7. Required checks in Branch Rules

**One rule only:** add only the `status-gate` job (or whatever you name the gate) as the required check in Branch Protection / Rulesets.

```
Settings → Branches → Branch protection rules
→ Require status checks to pass before merging
→ Add: "CI passed"   ← the name of the status-gate job
```

**Why not add every job individually:**
- Renaming a job requires manually updating the rules.
- If you forget to update, the job fails silently without blocking the merge.
- Skipped jobs report as success — the gate with `if: always()` handles this correctly.

---

## 8. Non-negotiable security baseline

```yaml
# In every workflow
permissions:
  contents: read          # default — never omit

# For workflows that deploy
permissions:
  contents: read
  id-token: write         # OIDC — instead of storing cloud credentials in secrets
```

- Pin external actions to a commit SHA, not a tag: `actions/checkout@11bd71901bbe5b1630ceea73d27597364c9af683`
- Never print secrets with `echo`. Use `::add-mask::` if you need to log a sensitive dynamic value.

---

## 11. Push discipline — zero tolerance for speculative CI

**It is COMPLETELY PROHIBITED to push a CI fix without being 100% certain the entire pipeline will pass.**

Before pushing any change that touches CI (workflows, Makefile targets, test files, toolchain config):

1. **Reproduce the exact CI command locally** — run the same `make` target, with the same features and flags that CI uses. If the target runs in a container, reproduce the environment locally (e.g., `docker run`).
2. **If you cannot reproduce locally** (platform-specific, e.g., x86_64 Linux vs arm64 macOS), document WHY and explain what makes you confident it will pass.
3. **One push = one green CI run.** If CI fails after your push, your methodology was wrong. Fix the methodology, not just the symptom.

Rationale: each failed CI run wastes 10-15 minutes of compute, blocks other PRs, and erodes trust in the pipeline. Trial-and-error pushing is not engineering — it's guessing.

---

## Reproducing warnings/errors locally across architectures

The workspace has `warnings = "deny"` in `[workspace.lints.rust]`, so **any warning is a
compile error**. Warnings can hide behind feature flags and architecture-specific `#[cfg]`
blocks. You MUST check all combinations before pushing.

### Quick reference: what to run

```bash
# ── 1. Library lint (ARM/local) ──
make fmt && make clippy          # this is the minimum gate

# ── 2. Test compilation with -Dwarnings (catches dead_code in test modules) ──
RUSTFLAGS="-Dwarnings" cargo test --workspace --exclude xtask --lib --all-features --no-run

# ── 3. Benchmark compilation ──
cargo bench --no-run

# ── 4. Cross-architecture (x86_64 via Docker) ──
make cross-sync
# Clippy (lib):
docker exec -w /src viprs-cross-x86_64 bash -c \
  'RUSTFLAGS="-Dwarnings" cargo clippy --workspace --exclude xtask --lib --features "jpeg,png,webp,gif,tiff,avif,fft,tracing,icc,heif,jxl,openslide,dcraw" -- -D clippy::perf -D clippy::unwrap_used -D clippy::expect_used'

# Test compilation (catches arch-specific dead code):
docker exec -w /src viprs-cross-x86_64 bash -c \
  'RUSTFLAGS="-Dwarnings" cargo test --workspace --exclude xtask --lib --features "jpeg,png,webp,gif,tiff,avif,fft,tracing,icc,heif,jxl,openslide,dcraw,_integration,lock_instrumentation,simd-pulp,mmap,rayon" --no-run'

# xtask build (what CI's `make xtask` does on x86):
docker exec -w /src viprs-cross-x86_64 bash -c \
  'RUSTFLAGS="-Ctarget-cpu=native" cargo build --release -p xtask --features count-alloc'
```

### Why warnings hide per-architecture

Many functions in `viprs-ops-spatial` dispatch to NEON (aarch64) or AVX2 (x86_64+avx2)
implementations. Test helpers that validate scalar-vs-SIMD parity are gated with:

```rust
#[cfg(any(target_arch = "aarch64", all(target_arch = "x86_64", target_feature = "avx2")))]
```

On CI's `ubuntu-latest` (x86_64 **without** AVX2 by default), these blocks don't compile,
leaving their helper functions as dead code → error under `deny(warnings)`.

Similarly, constants/functions used only by `_root_test_support`-gated tests appear dead
when compiled without that feature.

### Key principle

**If a function is only called from a `#[cfg(...)]` block, the function itself must carry
the same `#[cfg(...)]` attribute.** Otherwise it's dead code on the excluded architectures.
- Third-party secrets: scoped to the corresponding environment (dev/staging/prod), not the global repo.