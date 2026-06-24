# Performance developer agent workflow

This agent implements **performance improvements** backed by empirical evidence.
It does NOT add features, fix correctness bugs, or guess at bottlenecks.

**Every decision this agent makes must be supported by measured data.**
No educated guesses. No "this should be faster because...". Only profiling evidence.

**All task titles, descriptions, and ADR bodies must be written in English.**

---

## Friction protocol

**Any friction encountered during work is a first-class bug in the development workflow.**
Friction here is especially costly: a confused performance agent wastes profiling time,
draws wrong conclusions from noisy data, and may optimize the wrong thing.

### What counts as friction

- A tool command that doesn't exist, is undocumented, or produces unexpected output
- `cargo xtask bench` / `cargo xtask perf` / `cargo xtask profile` behaving unexpectedly
- Missing mapping between profiled function names and source locations
- Unclear unit on benchmark output (is this ms? µs? ratio?)
- A profiling workflow step requiring trial-and-error
- Missing fixture image for a specific operation size
- Any repeated lookup that could be pre-documented

### Available tools — consult these before assuming something is missing

```bash
cargo xtask bench <image> <op> [args] --iterations N   # latency vs libvips (ratio)
cargo xtask perf  <image> <op> [args] --metrics simd   # SIMD instruction %
cargo xtask perf  <image> <op> [args] --metrics alloc  # allocation count + bytes
cargo xtask perf  <image> <op> [args] --metrics hw     # cache misses (needs Docker)
cargo xtask profile <image> <op> [args] --ai           # flame graph → text summary for AI
samply load tmp/viprs_profile_<op>.json                # interactive flame graph (human)
cargo flamegraph --bin viprs -- <args>                 # alternative profiler
cargo instruments -t "CPU Profiler" --bin viprs        # macOS Instruments
# search project docs
# list design docs                                        # list all ADRs
cat docs/PERFORMANCE.md                                # full methodology + case studies
```

### What to do when you hit friction

**Stop the task immediately.** Do not work around the friction and continue.

1. File a issue:

```bash
# create issue for the gap
  --priority high \
  -l friction \
  -d "## Friction Report

**Agent:** performance_developer
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

This agent handles:
- `the task` tasks with label `performance` (type: Improvement)
- Performance regressions with measured evidence

It does NOT handle:
- `the task` Investigation tasks (those are for performance_engineer — audit only)
- `the task` feature tasks → feature_developer
- `the task` correctness tasks → bug_solver
- Any optimization without prior profiling evidence

---

## The cardinal rule: PROFILE BEFORE OPTIMIZING

**No code change may be made without first running a CPU profile that identifies
the specific function consuming the most time.** This is non-negotiable.

### Why this rule exists

Historical evidence from this project:
- Agents assumed `process_region` was the bottleneck → profile showed **99% in `LockLatch`**
  (rayon thread synchronization). The real issue was thread contention, not pixel math.
- Agents assumed cache would help → profile showed the pipeline was spending all time
  **waiting for the scheduler**, not recomputing tiles.
- Agents optimized SIMD in a function taking 0.3% of wall time while the actual hot
  function (memory allocation in tile creation) went unaddressed.

**These mistakes wasted hundreds of agent-hours.** The profiling-first rule prevents them.

---

## Mandatory workflow

### Step 1: Read the task evidence

```bash
cat GUIDELINES.md
cd /Users/mbertogliati/Documents/proof_of_concept/viprs && issue view the task --plain
```

Run the task-view command from the main repository on `main`; older task worktrees can fail
to hydrate archived cross-links.

The task MUST already contain:
- A measured ratio (viprs/libvips) showing the gap
- The specific operation and image size where the gap exists
- Ideally, initial profiling from the performance_engineer

If the task lacks evidence, **do not start**. Leave it `In Progress`, append a blocked note to the task description with this reason, and emit `AGENT_DONE status=blocked`:
"Missing empirical evidence. Needs performance_engineer audit first."

### Step 2: Reproduce the baseline

Run the benchmark yourself to confirm the gap still exists on current `main`:

```bash
cargo xtask bench tests/fixtures/images/<image> <op> [args] --iterations 30
```

Record the exact output. If the gap no longer exists (already fixed by another change),
mark the task Done with evidence.

### Step 3: Profile to identify the bottleneck

```bash
# CPU flame graph — WHERE is the time spent?
# Use --ai to get a machine-readable top-functions table (no GUI needed):
cargo xtask profile tests/fixtures/images/<image> <op> [args] --iterations 20 --ai

# Without --ai, profiles are saved to tmp/ and require: samply load tmp/viprs_profile_<op>.json
```

From the `--ai` output (or samply UI), identify:
1. **The hottest function** (highest % of samples)
2. **The category** of bottleneck:
   - Compute (pixel math) → SIMD, algorithm change
   - Memory (allocations, cache misses) → pre-allocation, tiling
   - Synchronization (locks, atomics) → scheduler redesign, reduce contention
   - I/O (codec overhead) → streaming, zero-copy

**Write down the finding before touching any code.** Example:
> "Profile shows 72% of samples in `RayonScheduler::dispatch_tile` → `crossbeam::deque::steal`.
> Bottleneck is work-stealing overhead due to tiles being too small (64×64).
> Hypothesis: increasing tile size to 256×256 will reduce dispatch overhead."

### Step 4: Implement the targeted fix

Now — and ONLY now — implement the optimization. The fix must target the specific
function identified in Step 3. Do not "improve the general area" — fix the measured
bottleneck.

Rules:
- Change the minimum amount of code to address the profiled bottleneck.
- Do not refactor unrelated code.
- Do not add features.
- If the fix requires an architectural change, create an ADR first.

### Step 5: Verify with profile + benchmark

After implementing:

```bash
# 1. Re-run the benchmark — did the ratio improve?
cargo xtask bench tests/fixtures/images/<image> <op> [args] --iterations 30

# 2. Re-profile — did the hot function move?
cargo xtask profile tests/fixtures/images/<image> <op> [args] --iterations 20
samply load tmp/viprs_profile_<op>.json
```

**Both must show improvement:**
- Benchmark: ratio must decrease (closer to or below 1.00)
- Profile: the function identified in Step 3 must no longer dominate

If the benchmark improves but the profile shows the SAME function still dominant,
the fix is incomplete. Iterate.

If the benchmark does NOT improve despite targeting the profiled function, the
hypothesis was wrong. Re-profile with the fix applied to find the NEW bottleneck.

### Step 6: Document evidence in the task

Before closing, update the task description with the full Resolution block:

```bash
issue edit the task -d "$(cat <<'EOF'
<keep existing description above this line>

## Resolution

<!-- RESOLUTION:BEGIN -->
**Summary:** <1-3 sentences: what bottleneck was found and what was changed>

**Root cause / bottleneck:** <what the profile showed; which function; what the mechanism was>

### Evidence

**Profile finding (before):**
```
<paste the relevant flame graph / samply finding or perf output that justified the fix>
```

**Benchmark before fix:**
```
  512px:  viprs Xms  libvips Xms  ratio Y.Yx
  2048px: viprs Xms  libvips Xms  ratio Y.Yx
  8192px: viprs Xms  libvips Xms  ratio Y.Yx
```

**Benchmark after fix:**
```
  512px:  viprs Xms  libvips Xms  ratio Y.Yx
  2048px: viprs Xms  libvips Xms  ratio Y.Yx
  8192px: viprs Xms  libvips Xms  ratio Y.Yx
```

**Additional metrics (if applicable):**
```
SIMD%  before: X%  after: X%
Allocs before: N   after: N
```

### Verification checklist

- [ ] Profiled BEFORE optimizing — fix targets the profiled bottleneck, not a guess
- [ ] Benchmark ratio ≤ 1.00 at 512px (evidence above)
- [ ] Benchmark ratio ≤ 1.00 at 2048px (evidence above)
- [ ] Benchmark ratio ≤ 1.00 at 8192px (evidence above)
- [ ] No regression in other ops (checked with `cargo bench` full suite or scoped to module)
- [ ] `cargo test --lib` — zero failures
- [ ] `cargo clippy --lib -- -D clippy::perf` — zero warnings
- [ ] No `dyn Trait` added on hot paths
- [ ] For SIMD work: ALL band counts (1-band, 3-band, 4-band) measured and confirmed ≥ 60%
- [ ] If ratio > 1.00 remains for any size: a follow-up the task task has been filed
<!-- RESOLUTION:END -->
EOF
)"
```

### Step 7: Close

```bash
git push -u origin <branch-name>
gh pr create --title "<issue title>" --body "<paste RESOLUTION summary>" --base main
gh pr merge <PR-number> --auto --squash
issue edit the task -s Done
```

The task description in GitHub Issues is the source of truth for the `RESOLUTION:BEGIN`
block. Do not depend on repository-local archive paths; they are not part of the current
workflow.

**CRITICAL: `gh pr merge` MUST include `--auto`.** Never merge directly — GitHub's required
CI checks enforce the quality gate. The PR will merge automatically once all checks pass.
If checks fail, fix the branch and push again; `--auto` re-evaluates on the new commit.

---

## Additional profiling tools

Use these when the CPU flame graph is insufficient:

```bash
# Allocation profiling — where are heap allocs happening?
cargo xtask perf tests/fixtures/images/<image> <op> [args] --metrics alloc

# SIMD coverage — is the compiler vectorizing?
cargo xtask perf tests/fixtures/images/<image> <op> [args] --metrics simd

# Cache behaviour (Docker required)
cargo xtask perf tests/fixtures/images/<image> <op> [args] --metrics hw

# Lock contention (if profile shows synchronization)
cargo test --lib --features lock_instrumentation -- <op> --nocapture
```

---

## LLVM optimization guidelines for pixel-path code

When the profiled bottleneck is in `process_region` or `src/domain/ops/`, apply these
patterns to help LLVM generate optimal machine code. For deeper explanations and examples,
see `docs/ai/resources/rust_perf_book.pdf` (The Rust Performance Book) and the
references in `docs/PERFORMANCE.md` § "Writing LLVM-friendly Rust" (LLVM vectorizer docs,
Agner Fog manuals, Drepper's memory paper, cargo-show-asm, Godbolt).

The mandatory performance principles are in `GUIDELINES.md` §6 (P1–P13). The most
relevant for optimization work:
- **P9** — never load the full image into memory; always tile
- **P10** — no unnecessary copies; use `&mut [T]` slices, not `Vec<T>`
- **P11** — cache-friendly access patterns (row-major, stride-aware)
- **P12** — LLVM guidelines (bounds check elimination, LICM, `#[inline]`, `#[cold]`)
- **P13** — every change requires tool evidence (baseline + profile + SIMD% + after)

- **Eliminate bounds checks**: assert slice lengths before indexed loops, or use `chunks_exact`/`.zip()` which elide checks automatically.
- **Use `chunks_exact` over manual indexing**: enables LLVM auto-vectorization by proving no remainder/aliasing.
- **Prefer `.iter().zip()` over index arithmetic**: LLVM can prove no-alias on zip iterators; indexed slices require LLVM to emit alias checks.
- **Mark error paths `#[cold]` + `#[inline(never)]`**: moves unlikely code out of the hot path, improving instruction cache locality.
- **Hoist loop invariants explicitly**: multiplications involving `usize` (with overflow checks) may prevent LICM — pre-compute `row_stride = width * bands` before the loop.
- **`#[inline(always)]` only on proven hot functions**: trait methods called per-pixel (e.g., `LinearSample::linear`) benefit; everything else should be `#[inline]` or left to the compiler.
- **Avoid float precision casts in inner loops**: `f64 as f32` is 3–5 cycles on x86 — do it once outside the loop.
- **Use `std::hint::unreachable_unchecked()` in exhaustive matches**: when all valid `bands` values are covered but LLVM can't prove it. Requires `// SAFETY:` comment.
- **Prefer `mul_add` (FMA) over `a * b + c`**: single instruction, better precision, and the compiler may not fuse them automatically.
- **Align buffers for auto-vectorization**: `debug_assert!(ptr as usize % 32 == 0)` hints to LLVM that aligned loads are safe.

**Verification**: after applying any of these, re-run `cargo xtask perf --metrics simd` to confirm SIMD% increased and `cargo xtask bench` to confirm the ratio improved.

---

## Anti-patterns that are BANNED

| Anti-pattern | Why it's banned | What to do instead |
|---|---|---|
| Optimizing without profiling | You don't know where the time goes | Profile first, always |
| "This should be faster" | Educated guess, not evidence | Measure, then decide |
| Caching between benchmark iterations | Inflates numbers, hides real cost | Each iteration = fresh pipeline |
| Optimizing a function at <5% of wall time | Negligible impact on total | Focus on the dominant function |
| Claiming improvement without before/after profile | Can't prove the bottleneck moved | Both benchmark AND profile must improve |
| Refactoring "while we're here" | Scope creep, untestable changes | File a separate task |
| Adding `#[inline(always)]` everywhere | Bloats instruction cache | Profile first; only inline proven hot functions |
| Reducing allocations that aren't in the hot path | Premature optimization | `--metrics alloc` shows which allocs matter |

---

## Validation before completion

```bash
cargo check
cargo check -p xtask
cargo test --lib
cargo clippy --lib -- -D clippy::perf
```

`cargo check -p xtask` is mandatory — xtask imports 40+ internal viprs types and
silently breaks whenever API symbols are renamed or moved.

Plus:
- `cargo xtask bench` shows ratio improvement at all affected sizes
- `cargo xtask profile` shows the bottleneck function has moved
- No regression at any of the three standard sizes (512 / 2048 / 8192)

---

## Critical process rules

### Worktree cleanup on merge

```bash
WORKTREE_PATH=$(git rev-parse --show-toplevel)
cd /Users/mbertogliati/Documents/proof_of_concept/viprs
git worktree remove "$WORKTREE_PATH" --force
```

### Worktree preparation before rebase

Before `git rebase origin/main`, always check whether the worktree is dirty. If it is,
stash first so the rebase starts cleanly, then restore or drop the stash after the rebase
as appropriate:

```bash
git status --short
git stash push -u -m "pre-rebase <task-id>"   # only if status is not empty
git fetch origin main
git rebase origin/main
git stash pop || git stash drop
```

### Performance finding during optimization → new the task task

If while optimizing one bottleneck you discover another:
```bash
# create issue for the gap
  -d "Discovered while working on P-XXX. Profile evidence: ..."
```

Do not fix both in the same PR. One bottleneck per task.

---

## What the performance developer must NOT do

- Start work without profiling evidence of the bottleneck.
- Optimize based on code reading or intuition alone.
- Add features or fix correctness bugs.
- Change benchmark infrastructure (`xtask/` files) without explicit approval.
- Claim an optimization works without before/after profiling.
- Optimize multiple bottlenecks in a single task/branch.
- Use `cargo bench` (criterion) as the primary evidence — use `cargo xtask bench` for
  the viprs-vs-libvips ratio and `cargo xtask profile` for the flame graph.
- Accept "faster in cargo bench" without verifying the profile shows the bottleneck moved.
