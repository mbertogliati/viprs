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

Cache must be declared in a **shared composite action**, not duplicated across each job.

```yaml
# .github/actions/setup-toolchain/action.yml
name: Setup toolchain
description: Install dependencies with shared cache across jobs

runs:
  using: composite
  steps:
    - name: Cache
      uses: actions/cache@v4
      with:
        path: |
          ~/.cargo/registry
          ~/.cargo/git
          target/
        key: ${{ runner.os }}-cargo-${{ hashFiles('**/Cargo.lock') }}
        restore-keys: |
          ${{ runner.os }}-cargo-

    - name: Install toolchain
      uses: dtolnay/rust-toolchain@stable
      with:
        components: rustfmt, clippy
      shell: bash
```

**Cache rules:**
- The `key` must always include the lockfile hash (`Cargo.lock`, `package-lock.json`, etc). If the lockfile hasn't changed, the cache is reused.
- `restore-keys` as a partial fallback — avoids compiling from scratch on minor changes.
- Separate dependency cache from build artifact cache if they have different invalidation patterns.
- Never cache outputs that depend on source code — only dependencies and toolchains.

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
- Third-party secrets: scoped to the corresponding environment (dev/staging/prod), not the global repo.
