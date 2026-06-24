# Agent Workflows

The repository includes agentic workflow files under `.github/agents/`. They are working
instructions for AI agents and reviewers, not general user documentation.

Useful files include:

- `.github/agents/GUIDELINES.md` for engineering rules.
- `.github/agents/PERFORMANCE.md` for benchmark and profiling methodology.
- `.github/agents/CI_GUIDELINES.md` for CI structure.
- `.github/agents/feature_developer.md` for feature implementation flow.
- `.github/agents/bug_solver.md` for correctness work.
- `.github/agents/performance_developer.md` for profiling-led performance work.
- `.github/agents/reviewer.md` and `.github/agents/qa.md` for review and validation.

Agents working on this repository should use dedicated worktrees for implementation PRs.
They must avoid editing unrelated changes in the main working tree and must report the
branch, PR, validation, and caveats at the end of the task.

For documentation-only changes, keep the PR focused on docs. Do not mix API redesigns,
operation changes, or benchmark changes into the same PR unless the documentation cannot
be made accurate without them.
