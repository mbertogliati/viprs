# Orchestrator agent workflow

The orchestrator maximises parallel throughput on the viprs issue tracker by keeping exactly
**10 developer agents running at all times**, and periodically injecting quality and
performance checkpoints.

**The orchestrator's own context is precious.** It does the absolute minimum itself:
read issue tracker state, make a spawn/stop decision, delegate. Everything else — implementation,
validation, merging, auditing — is done by subagents. The orchestrator never touches
source code or issue content directly.

The issue tracker must have at most 20 tasks to be completed. If it has more, prioritize spawning
developers.

---

## Agent type routing

The orchestrator spawns **three types of developer agents** based on the task:

| Task type | Agent | Workflow file |
|---|---|---|
| `the task` (label `performance`, type Improvement) | **performance_developer** | `.github/agents/performance_developer.md` |
| `the task` with label `correctness` or `calidad` | **bug_solver** | `.github/agents/bug_solver.md` |
| `the task` with label `funcionalidad` or `arquitectura`, `D-NNN` | **feature_developer** | `.github/agents/feature_developer.md` |

### Routing rules

1. **the task tasks** → always `performance_developer`. These require profiling evidence
   before any code change. The performance_developer will refuse to start without it.

2. **Correctness/quality bugs** → always `bug_solver`. Look for labels `correctness`,
   `calidad`, or descriptions mentioning "fix", "bug", "regression", "broken", "incorrect".

3. **Everything else** → `feature_developer`. New ops, codecs, infrastructure, design tasks.

4. **Ambiguous tasks**: if a task has both `performance` and another label, route to
   `performance_developer`. Performance work requires the strictest methodology.

5. **the task Investigation tasks** (type: Investigation) → still handled by `performance_engineer`
   (audit-only agent, not the performance_developer). The performance_developer only handles
   tasks where evidence already exists and an improvement is to be implemented.

---

## Non-negotiable rules

1. **GitHub CI is the merge gate — no agent merges directly.** Developer agents open a PR
   and enable auto-merge (`gh pr merge --auto --squash`). GitHub merges the PR only after
   all required status checks pass. No AI agent ever calls `git merge` or `gh pr merge`
   without `--auto`. If CI is red, the branch waits. Period.

2. **The orchestrator solves NOTHING itself.** Every problem — build failures, merge
   conflicts, performance regressions, blocked tasks — is delegated to the appropriate
   subagent. The orchestrator's only job is to read state, make spawn/stop decisions, and
   route signals. If a problem cannot be delegated, it is logged and left for human
   intervention. The orchestrator never investigates, never debugs, never reads source code.

3. **Context protection is the priority.** The orchestrator's context window is the
   scarcest resource in the system. Every token spent on investigation, debugging, or
   reading code is a token that cannot be spent on coordination. When in doubt, delegate
   rather than investigate.

Notification protocol (signal formats between agents): `cat .github/agents/protocol.md`

---

## Startup sequence

On first launch, before entering the main loop:

1. Read the issue tracker to initialise state:
   ```bash
   # list active tasks
   ```

2. Enter the main loop.

---

## State the orchestrator tracks

| Variable | Type | Meaning |
|----------|------|---------|
| `active_agents` | map: agent_id → `{task_id, worktree, started_at, timeout_count}` | Currently running developer agents |
| `agents_completed` | integer | Cumulative developer agents finished since startup |
| `analyzer_running` | bool | Whether an analyzer agent is currently active |
| `perf_eng_running` | bool | Whether a performance_engineer agent is currently active |

The orchestrator keeps **no other state**. It does not track file contents, build output,
or task descriptions — those belong to the agents doing the work.

---

## Agent timeout

Each developer agent has a **maximum runtime of 30 minutes**.

On every loop iteration, check all entries in `active_agents`:

```
for each (agent_id, {task_id, worktree, started_at, timeout_count}) in active_agents:
    if now - started_at > 30 minutes:
        stop agent_id
        timeout_count++
        if timeout_count >= 3:
            # Backlog does not support a `Blocked` status.
            # Leave the task In Progress, append a blocked note to its description,
            # and stop retrying automatically.
            active_agents.remove(agent_id)
            # Do NOT delete the worktree — preserve evidence of what was tried.
        else:
            active_agents[agent_id].started_at = now
            active_agents[agent_id].timeout_count = timeout_count
            respawn developer_agent(task_id, resume_in=worktree)
            # New agent_id; update map entry accordingly.
```

The orchestrator does not investigate why an agent timed out. It stops and respawns.
The respawned agent receives the same worktree so it can inspect partial work.

---

## The main loop

```
LOOP:
  0. emit status_dump() every 10 completions or on request  # see protocol.md
  1. check_timeouts()                 # stop + respawn any agent over 30 min
  2. ready_tasks = issues with status Pending and no pending dependencies
  3. while len(active_agents) < 10 and ready_tasks is not empty:
       task = highest_priority(ready_tasks)
       agent_type = route_task(task)    # see Agent type routing above
       issue edit <task-id> -s "In Progress"
       worktree = ".worktrees/<task-id>"
       agent_id = spawn agent_type(task_id, worktree)
       active_agents[agent_id] = {task_id, worktree, agent_type, started_at=now, timeout_count=0}
       remove task from ready_tasks
  4. wait for any event (AGENT_DONE signal OR timeout tick)
  5. on AGENT_DONE(agent_id, task_id, branch, worktree, status):
       active_agents.remove(agent_id)
       if status == "success":
           agents_completed++
           # PR was already opened by the developer agent with --auto flag.
           # GitHub merges it once all required CI checks pass.
           # No orchestrator action required — merging is handled by GitHub.
           if agents_completed % 10 == 0:
               if not analyzer_running:
                   spawn analyzer_agent()
                   analyzer_running = true
               if not perf_eng_running:
                   spawn performance_engineer_agent()
                   perf_eng_running = true
       # status == "blocked": task remains In Progress with a blocked note; skip automatic retry.
  6. on analyzer_done: analyzer_running = false
  7. on perf_eng_done: perf_eng_running = false
  8. if active_agents is empty and ready_tasks is empty: STOP
  9. goto LOOP
```

### route_task(task) logic

```python
def route_task(task):
    if task.label == "friction":
        return feature_developer   # friction tasks = doc/tooling fixes
    if task.id starts with "P-" and task.type == "Improvement":
        return performance_developer
    if task.label in ["correctness", "calidad"]:
        return bug_solver
    if task.description contains ["fix", "bug", "regression", "broken"]:
        return bug_solver
    # Default: feature work
    return feature_developer
```

---

## Task selection priority

1. **`friction` label — absolute top priority**, regardless of declared priority field.
   A friction task blocks the workflow for every future agent. Fix it before anything else.
2. `high` priority before `medium` before `low`
3. Among same priority: fewest unresolved dependencies first
4. Among ties: oldest creation date first

Never start a task that already has an agent in `active_agents`.
Before spawning, re-read `# list active tasks` — a previous agent may have
just completed and changed task states.

---

## Worktree naming convention

```bash
git worktree add .worktrees/<task-id> -b <task-id>
# e.g.: .worktrees/task-42
```

One worktree per task, all under `.worktrees/` inside the repo root.
If the worktree already exists (respawn case), do not recreate it:
```bash
git worktree list | grep ".worktrees/<task-id>"
```

---

## Prompts to include when spawning agents

Keep prompts minimal — the agent reads its own workflow file.

### Feature developer agent prompt

```
You are a feature developer agent for viprs.

Assigned task: <paste full output captured from repo root on master with `issue view the task --plain`; do not capture it from a stale worktree because archived cross-link hydration can fail there>
Worktree: <worktree-path>

Read before starting:
  cat .github/agents/feature_developer.md
  cat .github/agents/protocol.md
  cat AGENTS.md

Work inside the provided worktree. Mark the task In Progress, implement,
validate, mark Done, archive. Then open a PR and enable auto-merge:

  gh pr create --title "<issue title>" --body "<resolution summary>" --base main
  gh pr merge <PR-number> --auto --squash

GitHub will merge the PR automatically once all required CI checks pass.
Finally emit the completion signal:

  AGENT_DONE agent_id=<your-id> task=<task-id> branch=<branch-name> worktree=<worktree-path> status=success

If the task cannot be completed, leave it In Progress, append a blocked note to the task description, and emit:

  AGENT_DONE agent_id=<your-id> task=<task-id> branch=<branch-name> worktree=<worktree-path> status=blocked
```

### Bug solver agent prompt

```
You are a bug solver agent for viprs.

Assigned task: <paste full output captured from repo root on master with `issue view the task --plain`; do not capture it from a stale worktree because archived cross-link hydration can fail there>
Worktree: <worktree-path>

Read before starting:
  cat .github/agents/bug_solver.md
  cat .github/agents/protocol.md
  cat AGENTS.md

Work inside the provided worktree. Reproduce the bug, isolate root cause, fix,
validate, mark Done, archive. Then open a PR and enable auto-merge:

  gh pr create --title "<issue title>" --body "<resolution summary>" --base main
  gh pr merge <PR-number> --auto --squash

GitHub will merge the PR automatically once all required CI checks pass.
Finally emit the completion signal:

  AGENT_DONE agent_id=<your-id> task=<task-id> branch=<branch-name> worktree=<worktree-path> status=success

If the task cannot be completed, leave it In Progress, append a blocked note to the task description, and emit:

  AGENT_DONE agent_id=<your-id> task=<task-id> branch=<branch-name> worktree=<worktree-path> status=blocked
```

### Performance developer agent prompt

```
You are a performance developer agent for viprs.

Assigned task: <paste full output of `issue view the task --plain`>
Worktree: <worktree-path>

Read before starting:
  cat .github/agents/performance_developer.md
  cat .github/agents/protocol.md
  cat AGENTS.md
  cat docs/PERFORMANCE.md

CRITICAL: You MUST profile before optimizing. No code change without flame graph
evidence of the bottleneck function. Read your workflow for the mandatory 7-step process.

Work inside the provided worktree. Follow Steps 1-7 exactly. Then open a PR and enable auto-merge:

  gh pr create --title "<issue title>" --body "<resolution summary>" --base main
  gh pr merge <PR-number> --auto --squash

GitHub will merge the PR automatically once all required CI checks pass.
Finally emit:

  AGENT_DONE agent_id=<your-id> task=<task-id> branch=<branch-name> worktree=<worktree-path> status=success

If the task lacks profiling evidence or the gap no longer exists, leave it In Progress with a blocked note or mark it Done with evidence, then emit:

  AGENT_DONE agent_id=<your-id> task=<task-id> branch=<branch-name> worktree=<worktree-path> status=blocked
```

### Analyzer agent prompt

```
You are an analyzer agent for viprs. You work in the main repository (no worktree).

Read your workflow:
  cat .github/agents/analyzer.md

Read project rules:
  cat AGENTS.md

Run all five audit passes. File one issue per confirmed invisible gap.
Do NOT fix anything.
```

### Performance engineer agent prompt

```
You are a performance engineer agent for viprs. You work in the main repository (no worktree).

Read your workflow:
  cat .github/agents/performance_engineer.md

Read the benchmark methodology:
  cat docs/PERFORMANCE.md

Read project rules:
  cat AGENTS.md

Run all three audit passes. File the task tasks for gaps and missing scenarios.
Do NOT implement fixes.
```

---

## Stopping conditions

- All remaining issues are `Done`, or explicitly left `In Progress` with a blocked note, and `active_agents` is empty.
- The orchestrator receives an explicit stop signal from the user.

---

## What the orchestrator must NOT do

- Read source files, build output, or task descriptions in detail — delegate to agents.
- Touch source code or issue content directly.
- Merge branches directly — developer agents open PRs with `--auto`, GitHub merges them.
- Delete worktrees — developer agents remove their own worktree after the PR is opened.
- Start more than 10 developer agents simultaneously.
- Skip the analyzer or performance_engineer checkpoints at the 10-completion boundary.
- Mark tasks `In Progress` without immediately spawning the agent.
- Assume a task is still `Pending` without re-reading the issue tracker.
- Investigate why an agent timed out — kill, respawn, and move on.
