# Agent notification protocol

All inter-agent signals in viprs use a single structured format so the orchestrator
and merger can parse them unambiguously.

---

## Developer → Orchestrator: task finished

When a developer agent completes its task (Done + archived), it emits:

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

## Orchestrator → Merger: merge request

When the orchestrator receives an `AGENT_DONE` with `status=success`, it forwards
a merge request to the always-on merger:

```
MERGE_REQUEST agent_id=<agent-id> task=<task-id> branch=<branch-name> worktree=<absolute-worktree-path>
```

---

## Merger → Orchestrator: merge result

```
MERGE_RESULT task=<task-id> branch=<branch-name> worktree=<absolute-worktree-path> status=<merged|failed> reason=<none|"description of failure">
```

Examples:
```
MERGE_RESULT task=task-42 branch=task-42 worktree=... status=merged reason=none
MERGE_RESULT task=task-42 branch=task-42 worktree=... status=failed reason="task not archived"
```

---

## Orchestrator → Agents: spawn payload

The orchestrator always passes the full task context in the spawn prompt.
The prompt format is defined in `docs/ai/agents/orchestrator.md` (see "Prompts to include").
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
merger: running
analyzer: idle (last run: after completion 20)
performance_engineer: idle (last run: after completion 10)
next checkpoint: after completion <(floor(completed/10)+1)*10>
ready tasks in the issue tracker: <count>
===========================
```
