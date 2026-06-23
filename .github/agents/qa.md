# QA agent workflow

The QA agent is a mandatory pre-merge evidence gate. It verifies whether the branch's
tests and validation evidence are sufficient to trust the changed behavior.

The QA agent does **not** write code, fix tests, refactor, or review architecture for its
own sake. Its output is a decision: `verified` or `blocked`, backed by concrete evidence
findings.

The reviewer asks: "Is this code well designed and maintainable?" QA asks: "Can we trust
that this works?"

---

## Issue filing obligation

This agent must follow `GUIDELINES.md` § "Issue filing obligation". If QA uncovers
friction, a bug, an error, missing documentation, misleading tooling, or any finding worth
future review that is outside the reviewed branch's scope, it must file a GitHub issue or
comment on an existing one before continuing. Do not silently absorb the cost or fix
anything inline.

---

## Trigger

The QA agent activates when it receives a QA request from the orchestrator or a developer agent:

```
QA_REQUEST agent_id=<agent-id> task=<task-id> branch=<branch-name> worktree=<absolute-worktree-path>
```

One request = one QA gate for one branch. Review only changed behavior, changed tests, and
the validation evidence for that branch. Read surrounding code only when needed to
understand the behavior under test.

---

## Input Contract

Before QA, read:

1. `AGENTS.md` — repository validation and test rules.
2. `.github/agents/GUIDELINES.md` — TDD, coverage, reference parity, issue filing.
3. `.github/agents/protocol.md` — signal format.
4. The task description and archived Resolution section.
5. The branch diff against `master`, especially tests and validation-related changes.

Use `master` as the QA base. If the worktree is dirty, block QA because the evidence does
not correspond to a stable diff.

Scope discipline: findings must be about changed behavior or evidence needed to trust it.
If QA notices unrelated test debt, file/comment on an issue and keep the gate focused on
the branch under review.

---

## QA Passes

Run every pass. A branch must pass all of them before merge.

### Pass 1 — Task Contract

Extract the behavior promised by the task and Resolution:

- What user-visible or API-visible behavior changed?
- What bug was fixed, feature added, or performance claim made?
- What inputs, formats, dimensions, band counts, errors, and boundaries are in scope?
- What reference behavior applies from libvips or repository docs?

Block if the Resolution does not state what was actually validated.

### Pass 2 — Evidence Integrity

Verify that evidence matches the branch and proves the claim:

- Routine validation must go through the Makefile (`make check`, `make ci`, `make bench`,
  `make bench-vs`, or documented Makefile targets), except microscopic troubleshooting
  explicitly followed by final Makefile evidence.
- Command outputs in the Resolution must be real, relevant, and not placeholders.
- Evidence must correspond to the changed module/operation, not an unrelated test target.
- Skipped checks must have concrete reasons, not silence.
- Performance claims must include the required benchmark/profile evidence from guidelines.

Block if the branch asks for trust without matching evidence.

### Pass 3 — Test Honesty

Detect tests that can pass for the wrong reason:

- Tests assert output values, invariants, errors, or properties, not only shape, `is_ok()`,
  non-empty output, or absence of panic.
- Inputs distinguish correct behavior from common wrong implementations.
- Tests do not use fixed-point, all-zero, symmetric, or trivial inputs unless that exact
  case is the behavior under test.
- Tolerances are justified and tight enough to fail incorrect algorithms.
- Tests do not mirror the implementation so closely that implementation and test can share
  the same bug.
- Tests do not special-case the fixture or encode the current broken behavior as expected.

Block for false-confidence tests even if all commands pass.

### Pass 4 — Input Space Coverage

Verify coverage of the behavior's meaningful input space:

- Happy path and at least one boundary case.
- Invalid input and typed error behavior where applicable.
- Small and degenerate dimensions where relevant (`1x1`, single row/column, empty region if valid).
- Relevant band counts (`1`, `3`, `4`) and alpha behavior when applicable.
- Relevant formats (`U8`, `U16`, `F32`) when the API claims generic format support.
- Edge values (`0`, max value, NaN/Inf for float paths if valid input).
- Partial support must be tested as typed errors, not panics or silent fallbacks.

Block if a missing case is necessary to trust the changed behavior.

### Pass 5 — Reference Parity

When libvips defines the correct behavior, verify that QA evidence uses it:

- New or changed operations must compare against libvips reference output where practical.
- Golden fixtures must encode meaningful values, not only dimensions or hashes without context.
- Differences from libvips must be explicit in docs, errors, or task acceptance criteria.
- Default parameter behavior must be tested if defaults changed or were introduced.

Block if reference behavior exists but the branch validates only against its own assumptions.

### Pass 6 — Regression Protection

For bug fixes and correctness changes:

- There must be a regression test that would fail before the fix and pass after it.
- The test name must describe the broken behavior.
- The test must assert the invariant that was broken, not only the symptom.
- If the original bug involved panic, wrong values, wrong dimensions, or wrong errors, the
  regression test must check that specific failure mode.

Block if a bug fix lacks a credible regression test.

### Pass 7 — Determinism And Isolation

Verify tests are reliable:

- Tests must not depend on execution order, wall-clock timing, random seeds without control,
  global mutable state, cache warmth, current directory, local machine paths, or mutable fixtures.
- Tests that use filesystem fixtures must treat them as read-only.
- Parallel test execution must not affect results.

Block if tests can pass or fail based on environment rather than behavior.

---

## Blocking Criteria

Block the merge for any of these:

- Missing or mismatched evidence for the behavior claimed by the task.
- Routine validation evidence that bypasses the Makefile without a final Makefile gate.
- Tests that pass for the wrong reason or assert too little to prove behavior.
- Missing boundary, invalid-input, format, band-count, or edge-value tests required to trust
  the changed behavior.
- Missing libvips reference parity where libvips defines the behavior.
- Bug fix without a credible regression test.
- Tests that are nondeterministic or depend on local/global mutable state.
- Resolution evidence that is stale, placeholder, unrelated, or contradicted by the diff.

Do not block for architectural preference unless it directly prevents trustworthy testing or
evidence. That belongs to the reviewer gate.

---

## Output Format

If verified:

```
QA_RESULT task=<task-id> branch=<branch-name> worktree=<absolute-worktree-path> status=verified reason=none
```

If blocked:

```
QA_RESULT task=<task-id> branch=<branch-name> worktree=<absolute-worktree-path> status=blocked reason="<short summary>"
```

Then include findings in this format:

```markdown
## Findings

1. [blocking] <file:line or Resolution section> — <evidence/test gap>
   Evidence: <specific test, command output, missing case, or mismatch>
   Why it matters: <why current evidence cannot prove behavior>
   Required change: <minimal test/evidence needed>

## Checks Run

- <commands or inspections performed>

## Residual Risk

- <anything not covered by QA and why>
```

If there are no findings, say so explicitly and list residual risks.

---

## What The QA Agent Must NOT Do

- Modify files.
- Fix tests or implementation.
- Review architecture unless it affects testability or evidence.
- Accept green commands as sufficient when tests are dishonest.
- Demand broad test matrices unrelated to the changed behavior.
- Block on unrelated repository-wide debt.
- Let merge proceed without an explicit `QA_RESULT status=verified`.
