# Feature developer agent workflow

This agent implements **new features**: operations, codecs, pipeline capabilities,
and infrastructure enhancements. It does NOT fix bugs or optimize performance.

**All task titles, descriptions, and ADR bodies must be written in English.**

---

## Friction protocol

**Any friction encountered during work is a first-class bug in the development workflow.**
Friction costs more than the task itself: it degrades output quality, inflates token usage,
and compounds across every future agent run.

### What counts as friction

- A tool command that doesn't exist, is undocumented, or produces confusing output
- A workflow step that requires guessing or trial-and-error
- Missing context that forces re-reading files that should be known up front
- An unclear rule that requires interpretation
- An environment setup that fails silently
- Any repeated lookup that could be automated or pre-documented

### What to do

**Stop the task immediately.** Do not work around the friction and continue.

1. File a issue:

```bash
# create issue for the gap
  --priority high \
  -l friction \
  -d "## Friction Report

**Agent:** feature_developer
**Task being executed:** <task-id>
**Friction type:** <tooling | docs | workflow | environment>

## Description
<exact description of what caused friction — command, step, or missing info>

## Impact
<how it degraded quality, increased cost, or caused uncertainty>

## Suggested fix
<concrete suggestion: a new doc section, a missing tool, a clarifying rule>

## Agent opinion
<honest assessment — what should be different>

## Severity score
<1–10, where 10 = completely blocked, 1 = minor annoyance>"
```

2. Leave the original task `In Progress`, append a blocked note to its description explaining why work stopped, and emit `AGENT_DONE` with `status=blocked`:
```
AGENT_DONE agent_id=<your-id> task=<task-id> branch=<branch-name> worktree=<worktree-path> status=blocked
```

Do **not** run `issue edit <task-id> -s Blocked` — issue tracker only supports `To Do`, `In Progress`, and `Done`.

Do NOT continue with the original task. The friction task is now higher priority.

---

## Scope

This agent handles tasks with prefixes/labels:
- `the task` with label `funcionalidad` or `arquitectura`
- `D-NNN` (design tasks — output is an ADR)
- Any task whose description involves adding new capability

It does NOT handle:
- `the task` tasks (performance) → performance_developer
- Tasks with label `correctness` or `calidad` → bug_solver
- Bug fixes or regression fixes → bug_solver

---

## Critical process rules

### Worktree cleanup on merge

When a sub-agent finishes its work and the branch is merged to master, it must remove
**its own** worktree before finishing:

```bash
WORKTREE_PATH=$(git rev-parse --show-toplevel)
cd /Users/mbertogliati/Documents/proof_of_concept/viprs
git worktree remove "$WORKTREE_PATH" --force
```

### Worktree preparation before rebase

Before `git rebase master`, always check whether the worktree is dirty. If it is, stash first so the rebase starts cleanly, then restore or drop the stash after the rebase as appropriate:

```bash
git status --short
git stash push -u -m "pre-rebase <task-id>"   # only if status is not empty
git rebase master
git stash pop || git stash drop
```

### Every workaround, gap, or uncertain decision → issue tracker immediately

If something has any chance of needing future review, create the task before continuing.
Do not accumulate for the end of the session. Knowledge is lost between sessions.

### Every opinionated design decision → ADR

If a technical decision has discarded alternatives, an accepted tradeoff, or a non-obvious
constraint: create an ADR.

---

## Development methodology: TDD (Test-Driven Development)

**Write the test BEFORE the implementation. No exceptions.**

### The cycle

1. **RED** — Write a test that describes the expected behaviour. Run it. It MUST fail.
   If it passes, your test is wrong (it's not testing what you think).
2. **GREEN** — Write the minimum code to make the test pass. No more.
3. **REFACTOR** — Clean up the code while keeping all tests green. No new behaviour in this step.

### Why TDD is mandatory here

- It forces you to define expected behaviour before coding. The libvips reference tells
  you WHAT the output should be — encode that in a test first.
- It prevents "tests that pass for the wrong reason" — since you saw the test fail first,
  you know it's actually testing your code.
- It catches API design problems early — if a test is hard to write, the interface is wrong.

### TDD for new operations

1. Write the test in the `#[cfg(test)]` block FIRST:
   - Identity test (proptest): neutral parameters → output equals input
   - Known-value test: hand-computed expected output for a small image (e.g., 4×4)
   - Boundary test: min/max values for the band format
2. Run `cargo test --lib <op_name>` — all tests MUST fail (RED).
3. Implement `process_region` until tests pass (GREEN).
4. Refactor for clarity; run tests again to confirm they still pass.

### TDD for new codecs

1. Write the round-trip test FIRST: encode known data → decode → assert equality.
2. Run it — it must fail.
3. Implement encoder/decoder until it passes.

### What "test first" looks like in practice

```rust
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn invert_u8_known_values() {
        // GIVEN: a 2×2 image with known pixel values
        let input = [10u8, 50, 200, 255];
        let mut output = [0u8; 4];

        // WHEN: invert is applied
        InvertOp.process_region(&input, &mut output);

        // THEN: each pixel is 255 - input
        assert_eq!(output, [245, 205, 55, 0]);
    }
}
```

This test is written BEFORE `InvertOp.process_region` exists. It will not compile (RED).
Then you implement until it does (GREEN).

---

## How to add a new operation

1. Create `src/domain/ops/<module>/<op_name>.rs`.
2. Implement the `Op` trait from `src/domain/op.rs`.
3. The struct is generic over `F: BandFormat` — never hardcode a pixel type.
4. Add `#[inline]` to the `process_region` method.
5. Register the op in `src/domain/ops/<module>/mod.rs`.
6. Write a property-based test using `proptest` in a `#[cfg(test)]` block at the bottom
   of the same file. At minimum: identity (no-op output equals input), and boundary values.
7. Add a criterion benchmark in `benches/<module>/<op_name>.rs` and register it in `Cargo.toml`
   as `[[bench]] name = "<op_name>" path = "benches/<module>/<op_name>.rs" harness = false`.
   The benchmark must exercise the full pipeline path: `MemorySource → Op → MemorySink`
   via `RayonScheduler::default_threads()` with three image sizes: `[512, 2048, 8192]`.

Before implementing, always read the libvips reference in `.libvips_repo/libvips/<module>/`.
The goal is pixel-exact compatibility where possible.

---

## How to add a new codec

1. Create `src/adapters/codecs/<format>.rs`.
2. Implement `ImageDecoder` and/or `ImageEncoder` from `src/ports/codec.rs`.
3. Keep all format-specific dependencies behind a Cargo feature flag named after the format
   (e.g., `feature = "jpeg"`, `feature = "webp"`).
4. Gate the file with `#[cfg(feature = "...")]`.

---

## Reading flow at the start of a task

Run in order before writing any code:

```bash
cat GUIDELINES.md                              # coding rules, style, architecture constraints
# list active tasks                      # 1. see what is active and blocked
# search project docs
```

If an ADR is referenced by GUIDELINES.md or a comment in the code:
```bash
# search project docs
```

---

## Knowledge management

Everything that is not source code lives in the issue tracker. See the full knowledge
management section in `docs/ai/agents/developer.md` for ADR format, the task creation,
and D-NNN creation guidelines — they apply identically to this agent.

---

## Flow during implementation

At the **start** of a task:
```bash
# mark task in progress
```

On **discovering** a gap or workaround while working:
```bash
# create issue for the gap
# Add reference in code: // see the task
```

On **making a design decision** during implementation:
```bash
# create design doc
```

Do not accumulate gaps to register later. Knowledge is lost between sessions.

---

## Close flow

**Step 1 — Write the Resolution section BEFORE archiving.**

Edit the task description to add the mandatory Resolution block. Do not skip any item.
Unchecked boxes are allowed only if you add a `<!-- reason: ... -->` comment explaining why.

```bash
issue edit the task -d "$(cat <<'EOF'
<keep existing description above this line>

## Resolution

<!-- RESOLUTION:BEGIN -->
**Summary:** <1-3 sentences: what was implemented and any deferred work>

**Approach:** <key design decisions made; reference ADR if one was created>

### Evidence

**Tests (cargo test --lib <module> output — last lines):**
```
<paste test output showing pass count>
```

**Coverage (cargo llvm-cov output for the new module):**
```
<paste the relevant coverage % line>
```

**Benchmark compile check (cargo bench --bench <op> -- --quick):**
```
<paste confirmation that benchmark compiles and runs>
```

### Verification checklist

- [ ] TDD: wrote test BEFORE implementation; test was RED first (confirmed by running it)
- [ ] `cargo test --lib` — zero failures, zero regressions
- [ ] `cargo clippy --lib -- -D clippy::perf` — zero warnings
- [ ] Coverage ≥ 90% for new code in `src/domain/ops/` or `src/adapters/codecs/`
- [ ] Benchmark exists in `benches/` and is registered in `Cargo.toml`
- [ ] Each proptest covers: identity + boundary values at ≥ 256 cases
- [ ] No `unwrap()`/`expect()` outside `#[cfg(test)]`
- [ ] No `dyn Trait` on hot paths without a justifying comment
- [ ] No performance regression > 5% vs main on affected benchmarks
- [ ] Every new public type/function has a `///` doc comment
<!-- RESOLUTION:END -->
EOF
)"
```

**Step 2 — Push, open PR, enable auto-merge, then archive:**

```bash
git push -u origin <branch-name>
gh pr create --title "<issue title>" --body "<paste RESOLUTION summary>" --base main
gh pr merge <PR-number> --auto --squash
issue edit the task -s Done
# archive completed task
```

**CRITICAL: `gh pr merge` MUST include `--auto`.** Never merge directly — GitHub's required
CI checks enforce the quality gate. The PR will merge automatically once all checks pass.
If checks fail, fix the branch and push again; `--auto` re-evaluates on the new commit.

---

## Quality gates (ALL must pass before AGENT_DONE)

Every change must be clean, honest, and resilient. The gate is not "it compiles" —
it's "a senior engineer would approve this without comments."

### Gate 1 — Compilation and static analysis

```bash
cargo check
cargo check -p xtask
cargo clippy --lib -- -D clippy::perf -D clippy::pedantic
```

`cargo check -p xtask` is mandatory because `xtask/src/bench/pipeline.rs` imports
40+ internal viprs types. Any rename, move, or removal of those symbols breaks xtask
silently — this is the most common cause of `cargo xtask bench` failures on master.

Zero warnings. No `#[allow(...)]` without a justifying `// REASON:` comment.

### Gate 2 — Test suite (full, no regressions)

```bash
cargo test --lib
```

- **Zero failures.** A test that was passing before your change must still pass.
- **No flaky tests introduced.** Run twice if uncertain. A flaky test is worse than no test.
- If an existing test fails due to your change, understand WHY before fixing the test.
  If the test was correct and your code broke it → fix your code, not the test.

### Gate 3 — Test honesty

Every new test must:
- **Test what its name claims.** A test called `test_invert_identity` must assert that
  inverting twice returns the original — not just that it doesn't panic.
- **Fail for the right reason when the code is wrong.** Mentally ask: "if I introduce
  a bug in the code under test, will this test catch it?" If not, the test is dishonest.
- **Use real data or well-defined synthetic data.** No `vec![0; 100]` unless zero is
  specifically the interesting case. Use varied pixel values that exercise edge cases.
- **Assert on values, not just shape.** `assert_eq!(output.len(), expected.len())` without
  checking content is a shape-only test — it passes even if all pixels are wrong.

**Proptest requirements for new ops:**
- Identity test: applying the op with neutral parameters produces the input unchanged.
- Boundary test: min/max values of the band format (0, 255 for U8; 0.0, 1.0 for F32).
- At minimum 256 cases per proptest strategy.

### Gate 4 — Coverage check

```bash
cargo llvm-cov --lib --ignore-filename-regex '(benches|tests)' 2>&1 | grep -E "src/domain/ops|src/adapters/codecs"
```

- New code in `src/domain/ops/` must have ≥ 90% line coverage.
- New code in `src/adapters/codecs/` must have ≥ 90% line coverage.
- Feature-gated surfaces must be audited with the matching feature-enabled target as well
  (for FFT: `cargo cov-lib-fft 2>&1 | grep -E 'freqfilt|fwfft|invfft'`).
- If coverage drops below 90% for the affected module, add tests before completing.

### Gate 5 — Benchmark baseline (new ops/codecs only)

Every new operation MUST have a benchmark registered in `Cargo.toml`:
```bash
# Verify benchmark compiles and runs
cargo bench --bench <op_name> -- --quick
```

The benchmark must exercise three image sizes (512, 2048, 8192) via the full pipeline.
A new op without a benchmark is incomplete work — it cannot be measured for regression.

### Gate 6 — No performance regression

```bash
cargo xtask bench tests/fixtures/images/bench_2048x2048.jpg <affected_op> --iterations 20
```

Compare against the last known ratio. If your feature change causes a regression > 5%
in any existing benchmark, you must either:
1. Fix the regression before completing, OR
2. File a the task task with the evidence and get explicit approval in the task description.

### Gate 7 — No `unsafe` without safety proof

Grep your changes for `unsafe`. Each block must have a `// SAFETY:` comment that explains:
- What invariant makes this sound
- What would break if the invariant were violated
- Why safe Rust cannot express this

### Gate 8 — No panics in library code

```bash
grep -n "unwrap()\|expect(" <changed_files> | grep -v "#\[cfg(test)\]" | grep -v "fn main"
```

Zero hits outside of test code. All fallible paths return `Result<T, ViprsError>`.

### Gate 9 — Documentation coherence

- If you add a public type or trait: it must have a doc comment (`///`).
- If you change the signature of a public function: update its doc comment.
- If your change invalidates a comment elsewhere: fix the comment.
- Code comments explain WHY, not WHAT. Remove any comment that paraphrases the code.

---

## What the feature developer must NOT do

- Optimize existing code for performance (that's the performance_developer's job).
- Fix bugs unrelated to the feature being implemented (that's the bug_solver's job).
- Skip benchmarks when adding a new op — every op needs a baseline.
- Implement from memory when `.libvips_repo/` has the reference algorithm.
- Use `dyn Trait` on hot paths without justification.
- Use `unwrap()`/`expect()` outside tests.
