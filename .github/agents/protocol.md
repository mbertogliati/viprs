# Agent notification protocol

All inter-agent signals in viprs use a single structured format so the orchestrator
can parse them unambiguously.

---

## Developer → Orchestrator: task finished

When a developer agent completes its task (Done + archived + PR opened), it emits:

```
AGENT_DONE agent_id=<agent-id> task=<task-id> branch=<branch-name> worktree=<absolute-worktree-path> status=<success|blocked>
```

Examples:
```
AGENT_DONE agent_id=dev-42 task=task-42 branch=task-42 worktree=/path/to/viprs/.worktrees/task-42 status=success
AGENT_DONE agent_id=dev-17 task=task-17 branch=task-17 worktree=/path/to/viprs/.worktrees/task-17 status=blocked
```

`status=blocked` means the agent could not complete the task and has left it `In Progress`
in the issue tracker with an explicit blocked note in the description. The orchestrator should
NOT retry it automatically. Do **not** use an issue status named `Blocked` — it does
not exist.

---

## Developer → GitHub: PR with auto-merge

When a developer agent finishes implementation (all quality gates passed, task archived),
it must open a PR and enable auto-merge **before** emitting `AGENT_DONE`:

```bash
# 1. Push the branch
git push -u origin <branch-name>

# 2. Open the PR
gh pr create --title "<issue title>" \
             --body "<resolution summary from RESOLUTION section>" \
             --base main

# 3. Enable auto-merge — GitHub merges when all required checks pass
gh pr merge <PR-number> --auto --squash
```

**CRITICAL: Never call `gh pr merge` without `--auto`.** Direct merge bypasses GitHub's
required status checks. Any CI failure → branch waits, no exception.

The worktree must NOT be deleted until the PR is merged. After `gh pr merge --auto` the
agent can remove the worktree because GitHub owns the merge from that point.

---

## Orchestrator → Agents: spawn payload

The orchestrator always passes the full task context in the spawn prompt.
The prompt format is defined in `.github/agents/orchestrator.md` (see "Prompts to include").
It is not a structured signal — it is a natural-language prompt.

---

## Status dump format

When a human requests a status dump (or the orchestrator emits one every N completions),
the orchestrator prints:

```
=== Orchestrator status ===
completed: <agents_completed>
active developer agents: <count>/<max>
  [dev-42] task=task-42 running for 4m
  [dev-17] task=task-55 running for 12m
  [dev-09] task=task-60 running for 1m
analyzer: idle (last run: after completion 20)
performance_engineer: idle (last run: after completion 10)
next checkpoint: after completion <(floor(completed/10)+1)*10>
ready tasks in the issue tracker: <count>
===========================
```
