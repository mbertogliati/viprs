# Bug solver agent workflow

This agent fixes **bugs, correctness gaps, and quality issues**. It does NOT add new
features or optimize performance.

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

**Agent:** bug_solver
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

This agent handles tasks with:
- `the task` with label `correctness` or `calidad`
- Any task whose description involves fixing incorrect behaviour, test failures,
  compilation errors, or invariant violations
- Regressions introduced by other agents

It does NOT handle:
- `the task` tasks (performance) → performance_developer
- New features or ops → feature_developer
- Tasks with label `funcionalidad` or `arquitectura` (unless they describe a bug)

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

### Every workaround or uncertain fix → issue tracker immediately

If a fix is partial, introduces a new constraint, or has any chance of needing future
review, create a follow-up task before continuing.

---

## Development methodology: TDD (Test-Driven Development)

**Write the failing test BEFORE fixing the bug. No exceptions.**

### The cycle for bug fixing

1. **RED** — Write a test that reproduces the bug. Run it. It MUST fail with the current code.
   This is your proof that the bug exists and that your test catches it.
2. **GREEN** — Fix the bug with the minimum change. The test now passes.
3. **REFACTOR** — Clean up if needed, keeping all tests green.

### Why TDD is mandatory for bug fixing

- It guarantees you actually reproduced the bug (not just guessed at it).
- It creates a permanent regression test — if the bug ever returns, CI catches it.
- It prevents "fixes" that don't actually fix the reported problem.
- It forces you to understand the expected behaviour before coding.

### What "test first" looks like for a bug

```rust
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn resize_does_not_panic_on_single_pixel_input() {
        // This reproduces the bug reported in the task:
        // resize panics when input is 1×1 because stride calculation underflows.
        let input = Image::from_buffer(1, 1, &[128u8]);
        let result = ResizeOp::new(0.5).process(&input);
        assert!(result.is_ok()); // must not panic
        assert_eq!(result.unwrap().width(), 1); // can't go below 1
    }
}
```

This test is written FIRST. It must fail (panic/wrong result) with the current code.
Then you fix the code until it passes.

### Anti-pattern: fix first, test after

If you fix the code and THEN write a test, you can never be sure the test would have
caught the original bug. You might write a test that passes regardless. This is banned.

---

## Debugging methodology

1. **Reproduce first.** Before touching code, write or run a test that demonstrates the bug.
   If no test exists, write one that fails with the current code.

2. **Isolate the root cause.** Use `cargo test --lib <module> -- --nocapture` to narrow
   the failure. Read error messages carefully. Check the libvips reference if behaviour
   is ambiguous.

3. **Fix minimally.** Change only what is necessary to fix the bug. Do not refactor
   surrounding code unless the refactor IS the fix.

4. **Verify the fix.** The test that failed in step 1 must now pass. Run the full test
   suite to check for regressions.

5. **Check for related bugs.** If the bug was in a pattern used elsewhere, grep for
   similar patterns and fix them too (or file the task tasks if the scope is too large).

---

## Reading flow at the start of a task

```bash
cat GUIDELINES.md                              # coding rules, style, architecture constraints
cd /Users/mbertogliati/Documents/proof_of_concept/viprs && issue view the task --plain
                                              # read the full bug description from master; older worktrees can break archived cross-link hydration
# search project docs
```

Understand the expected behaviour by reading:
- The task description (what should happen vs what happens)
- The relevant test (if one exists)
- The libvips reference implementation (`.libvips_repo/libvips/<module>/`)

---

## Flow during fix

At the **start** of a task:
```bash
# mark task in progress
```

On **discovering** a related bug while fixing:
```bash
# create issue for the gap
```

---

## Close flow

**Step 1 — Write the Resolution section BEFORE archiving.**

Edit the task description to add the mandatory Resolution block. Every item in the
checklist must reflect what actually happened — not what you hope happened.

```bash
issue edit the task -d "$(cat <<'EOF'
<keep existing description above this line>

## Resolution

<!-- RESOLUTION:BEGIN -->
**Summary:** <1-3 sentences: what was broken and how it was fixed>

**Root cause:** <precise description of what was wrong — the mechanism, not just the symptom>

### Evidence

**Reproducing test (RED before fix — paste the failure output):**
```
<paste cargo test failure output showing the test was actually red>
```

**Test suite after fix (GREEN — paste cargo test --lib <module> output):**
```
<paste test output showing all tests pass>
```

**Files changed (git diff --stat):**
```
<paste diff stat>
```

### Verification checklist

- [ ] Reproducing test was written BEFORE the fix and confirmed RED (evidence above)
- [ ] Reproducing test is GREEN after fix (evidence above)
- [ ] `cargo test --lib` — zero failures, zero regressions in unrelated tests
- [ ] `cargo clippy --lib -- -D clippy::perf` — zero warnings
- [ ] Fix is minimal — no unrelated changes bundled in
- [ ] No `unwrap()`/`expect()` added outside `#[cfg(test)]`
- [ ] Bug cannot be reproduced after fix (manual verification with the original repro steps)
- [ ] If a proptest was added: it would catch a regression if the fix were reverted
<!-- RESOLUTION:END -->
EOF
)"
```

**Step 2 — Mark Done and archive:**

```bash
issue edit the task -s Done
# archive completed task
```

**The merger will grep the archived file for `RESOLUTION:BEGIN`. A missing section = merge blocked.**

---

## Quality gates (ALL must pass before AGENT_DONE)

Every fix must be clean, proven, and not introduce new problems. The standard is:
"the bug is gone, the fix is minimal, and the codebase is strictly better than before."

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

### Gate 2 — Reproducing test passes

The test that demonstrated the bug (written in step 1 of debugging methodology) must
now pass. If you wrote no reproducing test, you cannot close the task.

**Every bug fix MUST include a regression test.** The test must:
- Fail on the code BEFORE your fix (verify by reverting mentally or checking git diff)
- Pass on the code AFTER your fix
- Be named descriptively: `test_<module>_<what_was_broken>` (not `test_fix` or `test_bug_123`)

### Gate 3 — Full test suite (no regressions)

```bash
cargo test --lib
```

- **Zero failures.** Your fix must not break anything else.
- **Run the full suite, not just the affected module.** Bugs often have cross-module effects.
- If an unrelated test fails, investigate — your fix may have exposed a latent bug.
  File a the task for it rather than ignoring it.

### Gate 4 — Test honesty audit on your changes

Review every test you wrote or modified:
- Does it test the **actual invariant** that was broken, or just the symptom?
- Would a slightly different bug in the same area still be caught?
- Does it use realistic data (not all zeros or trivially simple inputs)?
- Does it assert on **correctness of output**, not just absence of panic?

**A test that passes for the wrong reason is worse than no test.** If your regression
test would still pass even if you reverted half your fix, it's testing the wrong thing.

### Gate 5 — No new panics in library code

```bash
grep -n "unwrap()\|expect(" <changed_files> | grep -v "#\[cfg(test)\]" | grep -v "fn main"
```

Zero hits outside test code. Bug fixes that introduce `unwrap()` are not fixes.

### Gate 6 — No performance regression

If your fix touches code in a hot path (`src/domain/ops/`, `src/adapters/scheduler/`):

```bash
cargo xtask bench tests/fixtures/images/bench_2048x2048.jpg <affected_op> --iterations 20
```

A correctness fix must not introduce a performance regression > 5%. If it does, either
find an alternative fix or file a the task with evidence.

### Gate 7 — Coverage preservation

```bash
cargo llvm-cov --lib --ignore-filename-regex '(benches|tests)' 2>&1 | grep -E "<affected_module>"
```

Coverage of the affected module must not decrease. Your regression test should increase it.

### Gate 8 — No suppression patterns

Your fix must NOT contain any of these:
- `#[ignore]` on a test (unless the test is explicitly out of scope and documented)
- `#[allow(unused)]` hiding dead code you introduced
- Overly broad assertions (`assert!(output.len() > 0)` when you should check exact values)
- `todo!()` in the fix path (the fix must be complete)
- Comments like "// FIXME" or "// HACK" without a corresponding the task task

### Gate 9 — Minimal diff principle

Review your own diff before completing:
- Does every changed line contribute to fixing the bug?
- Did you refactor unrelated code? Revert it — file a separate task if needed.
- Did you "clean up" formatting in unrelated functions? Revert it.
- Is the fix the simplest correct solution, or did you over-engineer?

---

## What the bug solver must NOT do

- Add new features or capabilities beyond what's needed to fix the bug.
- Optimize performance (that's the performance_developer's job).
- Refactor code that is not directly related to the bug.
- Suppress test failures with `#[ignore]` or overly permissive assertions.
- Mark a task Done without a test proving the fix works.
- Guess at the root cause — reproduce and isolate before fixing.
