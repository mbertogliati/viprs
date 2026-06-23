# docs/ai — AI agent guidance

This folder contains workflow definitions for AI agents working on viprs.
Each file in `agents/` defines a **specialized agent role**: its input contract,
step-by-step process, and expected output format.

These files are not documentation for humans — they are read by AI agents at the
start of a task to understand what they are expected to do and how.

## Reading a workflow

```bash
cat .github/agents/orchestrator.md            # pool management: 10 agents + checkpoints + routing
cat .github/agents/feature_developer.md       # new features, ops, codecs
cat .github/agents/bug_solver.md              # bug fixes, correctness gaps
cat .github/agents/performance_developer.md   # performance improvements (profiling-first)
cat .github/agents/analyzer.md                # invisible-gap detection (audit-only, no fixes)
cat .github/agents/performance_engineer.md    # benchmark honesty + gap measurement (no fixes)
cat .github/agents/protocol.md                # inter-agent signal formats (AGENT_DONE, etc.)
```

## Available agents

| File | Role | Spawned by |
|------|------|-----------|
| `agents/orchestrator.md` | Keeps 10 developer agents running. Routes tasks by type to the correct agent. Every 10 completions fires one analyzer + one performance engineer. | Human / top-level trigger |
| `agents/feature_developer.md` | Implements new features, ops, codecs, infrastructure. Works in a dedicated worktree. Opens PR with auto-merge. | Orchestrator (for `funcionalidad`/`arquitectura`/`D-NNN` tasks) |
| `agents/bug_solver.md` | Fixes bugs, correctness gaps, quality issues. Reproduce-first methodology. Works in a dedicated worktree. Opens PR with auto-merge. | Orchestrator (for `correctness`/`calidad` tasks) |
| `agents/performance_developer.md` | Implements performance improvements with mandatory profiling evidence. No guessing. Works in a dedicated worktree. Opens PR with auto-merge. | Orchestrator (for `the task` Improvement tasks) |
| `agents/analyzer.md` | Audits for invisible gaps. Produces issues. Does NOT fix anything. Works in main repo. | Orchestrator (every 10 completions) |
| `agents/performance_engineer.md` | Audits benchmark honesty and gaps vs libvips. Produces the task tasks. Does NOT fix anything. Works in main repo. | Orchestrator (every 10 completions) |
| `agents/protocol.md` | Signal format reference. Not an agent — read by all agents. | — |

## Reference documents

| File | Purpose |
|------|---------|
| `GUIDELINES.md` | Engineering guidelines: architecture, TDD, type design, performance model |
| `PERFORMANCE.md` | Performance investigation methodology and case studies |
| `CI_GUIDELINES.md` | GitHub Actions CI structure, caching, visibility, and security baseline for PRs |
| `protocol.md` | Signal format reference for inter-agent communication |

## Adding a new agent

1. Create `docs/ai/agents/<name>.md`.
2. Define: input contract, process steps, output format, what the agent must NOT do.
3. Add a row to the table above.
4. Reference the file from `AGENTS.md` with a `cat` command.
