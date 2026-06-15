# Merger agent workflow

The merger agent runs continuously and listens for agent completion notifications.
When a developer agent finishes, the merger validates the state of its work and
merges its branch to master.

The merger acts conservatively: **when in doubt, do not merge, do not delete.**
It only removes the worktree it received a notification for — never any other.

Read before operating:
```bash
cat GUIDELINES.md   # coding rules and invariants to enforce during review
```

## Friction protocol

**Any friction in the merge validation workflow is filed as a high-priority issue.**
A merger that struggles with unclear checklist items or broken validation commands
will either block valid merges or let bad ones through — both are costly.

```bash
# create issue for the gap
  --priority high \
  -l friction \
  -d "## Friction Report

**Agent:** merger
**Friction type:** <checklist | tooling | docs | environment>

## Description
<exact description — which check, which command, what was unclear>

## Impact
<how it made the merge decision uncertain or forced a workaround>

## Suggested fix
<concrete suggestion>

## Agent opinion
<honest assessment>

## Severity score
<1–10>"
```

Emit `MERGE_RESULT status=failed reason="friction: <description>"` and stop.

---

---

## Trigger

The merger activates when it receives a `MERGE_REQUEST` signal (see `docs/ai/agents/protocol.md`):

```
MERGE_REQUEST agent_id=<agent-id> task=<task-id> branch=<branch-name> worktree=<absolute-worktree-path>
```

One signal = one merge attempt for that specific agent only.

---

## Pre-merge checklist

Before merging, verify each of the following. If any check fails, **stop and report** —
do not merge, do not delete the worktree.

### 1. Task is marked Done

```bash
# Find which task the agent was working on (look for "In Progress" → "Done" transition)
# list active tasks | grep Done
# Or if the branch name encodes the task ID (e.g., viprs-the task), run from the repo root on master:
issue view the task --plain
```

The task status must be `Done`. If it is still `In Progress` or `Pending`, the agent
did not finish cleanly. **Do not merge.**

### 2. Task is archived

```bash
ls issue tracker/archive/tasks/ | grep -i "b-nnn\|p-nnn"
```

The task file must exist in `issue tracker/archive/tasks/`. If it is still in `issue tracker/tasks/`,
the agent forgot to archive. **Do not merge** — contact the orchestrator to resolve.

### 2b. Resolution section is present in the archived task

```bash
ARCHIVED=$(ls issue tracker/archive/tasks/ | grep -i "<task-id>" | head -1)
grep -q "RESOLUTION:BEGIN" "issue tracker/archive/tasks/$ARCHIVED" && echo "OK" || echo "MISSING"
```

The archived task file **must** contain the `<!-- RESOLUTION:BEGIN -->` marker. This proves
the agent documented how the task was resolved before closing — with evidence, root cause,
and an honest verification checklist.

If the marker is absent:
- The task was archived without a Resolution section.
- This is **false confidence**: a Done task with no documented evidence of correctness.
- **Do not merge.** Emit `MERGE_RESULT failed` with `reason="Resolution section missing from archived task. Agent must reopen, fill the Resolution block (see agent close flow docs), re-archive, and resubmit MERGE_REQUEST."`.

If the marker is present but the checklist items are all unchecked (all `- [ ]`):
- The agent filled the template but did not verify anything.
- **Do not merge.** Emit `MERGE_RESULT failed` with `reason="Resolution checklist is entirely unchecked. At minimum, cargo test and clippy items must be verified."`.

**What counts as acceptable:**
- At least the `cargo test --lib` and `cargo clippy` checklist items are checked (`- [x]`)
- Evidence fields are non-empty (not left as `<paste output here>` placeholders)
- Unchecked items have a `<!-- reason: ... -->` comment explaining why they were skipped

### 3. Worktree is clean

```bash
git -C <worktree-path> status --short
```

Must return empty output. If there are uncommitted changes, the agent left work in an
inconsistent state. **Do not merge.**

### 3b. xtask compiles (unconditional)

**Always run this**, regardless of what files the branch touches:

```bash
git checkout <branch-name>
cargo check -p xtask
```

`xtask/src/bench/pipeline.rs` imports 40+ internal viprs types. Any API rename, move,
or removal in the lib that was not propagated to xtask breaks `cargo xtask bench` on
master. This is the most common cause of xtask failures and it is invisible to agents
who only run `cargo check` (lib-only).

If this command fails → **do not merge**. The branch author must fix the xtask import
before resubmitting.

### 4. Branch is ahead of master and behind by 0

```bash
git log master..<branch-name> --oneline   # must show at least 1 commit
git log <branch-name>..master --oneline   # must show 0 commits (no divergence)
```

If the branch has diverged from master (behind > 0), rebase is needed. The merger
does not rebase — **do not merge**, report to orchestrator. When handing it back, use
the standard worktree prep before `git rebase master` if the worktree is dirty:

```bash
git status --short
git stash push -u -m "pre-rebase <task-id>"   # only if status is not empty
git rebase master
git stash pop || git stash drop
```

### 5. No performance regression

Run criterion benchmarks for the modules touched by the branch:

```bash
# From the repo root (master):
git stash
cargo bench -- --save-baseline pre-merge 2>/dev/null

# Switch to the branch:
git checkout <branch-name>
cargo bench -- --baseline pre-merge 2>&1 | grep -i "regressed"
```

If ANY benchmark shows regression > 5% (criterion reports "Performance has regressed"):
- Emit `MERGE_RESULT failed` with `reason="perf regression in <bench_name>: <details>"`
- Do NOT merge
- Do NOT delete the worktree

If `cargo bench` fails to compile or run, treat as a build failure — do not merge.

**Shortcut for large benchmark suites:** if the branch only touches files in a specific
module (e.g., `src/domain/ops/arithmetic/`), you may scope the bench run:

```bash
cargo bench --bench <module_name> -- --baseline pre-merge
```

But when in doubt, run the full suite. A 5-minute bench run is cheaper than a merged
regression.

### 5b. Thumbnail e2e smoke test (vs libvips)

If the branch touches any file in `src/domain/ops/resample/`, `src/adapters/pipeline.rs`,
`src/adapters/scheduler/`, or `xtask/src/bench/`, run the thumbnail e2e gate:

```bash
# From the branch worktree:
cargo xtask bench tests/fixtures/images/bench_2048x2048.jpg thumbnail 400 \
  --iterations 5 --no-cache --scenarios no-cache 2>&1 | tee /tmp/perf-gate.txt

# Extract the ratio:
RATIO=$(grep "latency p50:" /tmp/perf-gate.txt | awk '{print $3}' | sed 's/x//')
```

**Gate rules:**
- If viprs/libvips ratio > 2.0 → **do not merge** (catastrophic regression).
  Known-good baseline: JPG 2048 e2e ≈ 0.80x (viprs wins).
- If `cargo xtask bench` panics or fails to compile → treat as build failure, do not merge.
- If ratio is between 1.0–2.0 → merge is allowed but emit a warning in the merge commit.
- If ratio < 1.0 → viprs wins, no issue.

This gate catches bugs like passing wrong types to pipeline construction that
silently break shrink-on-load (which caused a 12x regression that went undetected
through 15+ merges).

**Skip this gate** for branches that only touch docs, tests (not xtask), issue tracker,
or files outside `src/`.

### 6. CI passes (if configured)

If a CI check is defined, it must be green before merging.

---

## Merge

All checks passed → merge to master:

```bash
cd /Users/mbertogliati/Documents/proof_of_concept/viprs
git merge --no-ff <branch-name> -m "Merge <branch-name>: <task title>

Co-authored-by: Copilot <223556219+Copilot@users.noreply.github.com>"
```

Use `--no-ff` to preserve the branch history.

After a successful merge, emit the result signal (see `docs/ai/agents/protocol.md`):

```
MERGE_RESULT task=<task-id> branch=<branch-name> worktree=<worktree-path> status=merged reason=none
```

## Worktree cleanup

After a successful merge **and only after**, remove the worktree for the agent
that sent the notification:

```bash
WORKTREE_PATH=<worktree-path from notification>

# Confirm this path is registered as a worktree (not main repo, not another agent's)
git worktree list | grep "$WORKTREE_PATH"
```

If the path does not appear in `git worktree list`, **do not run the remove command** —
the worktree is already gone or was never registered.

If the path appears:

```bash
git worktree remove "$WORKTREE_PATH" --force
git branch -d <branch-name>
```

**Never** use `git worktree prune` — it would remove worktrees belonging to other
agents that are still running.

---

## Failure handling

On any failed check, emit before stopping:

```
MERGE_RESULT task=<task-id> branch=<branch-name> worktree=<worktree-path> status=failed reason="<description>"
```

| Failure | Action |
|---------|--------|
| Task not Done | Emit MERGE_RESULT failed. Do not merge. Do not delete worktree. |
| Task not archived | Emit MERGE_RESULT failed. Do not merge. Do not delete worktree. |
| Resolution section missing (`RESOLUTION:BEGIN` not found) | Emit MERGE_RESULT failed. Do not merge. Do not delete worktree. |
| Resolution checklist entirely unchecked | Emit MERGE_RESULT failed. Do not merge. Do not delete worktree. |
| Dirty worktree | Emit MERGE_RESULT failed. Do not merge. Do not delete worktree. |
| Branch diverged | Emit MERGE_RESULT failed. Do not merge. Do not delete worktree. |
| Worktree path not in `git worktree list` | Skip worktree removal (already clean). Merge proceeds if other checks passed. |
| Merge conflict | Emit MERGE_RESULT failed. Do not force. Leave branch and worktree intact. |

The merger never destroys state it is not certain about. Leaving a branch and worktree
intact is always the safe fallback.

---

## What the merger must NOT do

- Merge a branch whose task is not `Done` and archived.
- Delete a worktree it did not receive a notification for.
- Use `git worktree prune` (affects all worktrees, not just the target one).
- Force-push or rebase branches.
- Assume a worktree path is safe to delete without first checking `git worktree list`.
- Process more than one notification at a time — handle one merge per activation.
