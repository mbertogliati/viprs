# Performance engineer agent workflow

The performance engineer audits viprs for performance gaps and benchmark honesty.
It produces issues. It does NOT fix anything, does NOT modify benchmarks,
and does NOT interpret results charitably.

**Mission: viprs must be faster than libvips, not merely competitive.**
The target ratio is ≤ 1.00 (viprs p50 ÷ libvips p50). Any operation slower than
libvips is a gap, regardless of magnitude. Rust gives us zero-FFI-overhead SIMD,
ownership-based zero-copy pipelines, and predictable allocator behaviour — every
gap is an opportunity to exploit those advantages.

**This agent is the protector against false measurements.** A benchmark that
measures the wrong thing is worse than no benchmark. Every finding that could
let a lie pass as truth must become a issue.

**PROFILING IS MANDATORY.** No performance task may be filed without first running
`cargo xtask profile` and viewing the result with `samply load`. The profile
determines the actual bottleneck. Without it, you are guessing.

---

## Friction protocol

**Any friction is reported immediately as a high-priority issue.**
A performance engineer working around broken or missing tooling produces unreliable evidence,
which leads to wrong the task tasks and wasted optimization work downstream.

### Available tools — consult these before assuming something is missing

```bash
cargo xtask bench <image> <op> [args] --iterations N   # latency vs libvips (ratio)
cargo xtask perf  <image> <op> [args] --metrics simd   # SIMD instruction %
cargo xtask perf  <image> <op> [args] --metrics alloc  # allocation count + bytes
cargo xtask perf  <image> <op> [args] --metrics hw     # cache misses (needs Docker)
cargo xtask profile <image> <op> [args] --ai           # flame graph → text for AI
samply load tmp/viprs_profile_<op>.json                # interactive flame graph
# search project docs
cat docs/PERFORMANCE.md                                # full methodology + case studies
```

```bash
# create issue for the gap
  --priority high \
  -l friction \
  -d "## Friction Report

**Agent:** performance_engineer
**Audit pass:** Pass N — <pass name>
**Friction type:** <tooling | docs | benchmark_infra | environment>

## Description
<exact description — which command, which step, what was unclear>

## Impact
<how it made measurements unreliable or forced guessing>

## Suggested fix
<concrete suggestion>

## Agent opinion
<honest assessment>

## Severity score
<1–10>"
```

Stop that audit pass. File the friction task, then continue with remaining passes if possible.

---
<1% of wall time while the real bottleneck (e.g., LockLatch thread contention
in rayon) went unaddressed.

**BENCHMARKS: ONE MODE ONLY.** There is no "with-cache" vs "no-cache" comparison.
Internal tile cache is always ON. Each iteration builds a fresh pipeline (cold cache).
No inter-iteration caching is allowed. Any result claiming >5x faster than libvips
must be scrutinized for cache cheating.

---

## Performance target

| Ratio (viprs / libvips p50) | Classification | Required action |
|---|---|---|
| < 0.80 | ✅ Win — exceeds target | Document; keep |
| 0.80 – 1.00 | ✅ Acceptable — at parity | Monitor; no task needed |
| 1.01 – 1.50 | 🟡 Gap — small | File `medium` priority task with evidence |
| 1.51 – 2.00 | 🟠 Gap — significant | File `high` priority task with evidence |
| > 2.00 | 🔴 Gap — critical | File `high` priority task with full profiling output |

**Do not accept "< 2x" as a passing grade.** Anything above 1.00 is a gap.

**Apply the target per size class, not as a single number.** An op may win at 512×512
(compute-bound, SIMD dominates) and lose at 8192×8192 (memory-bandwidth-bound, tiling
strategy dominates). Both are real. A "win" that only holds at small images is not a win.

---

## Scaling behaviour and crossover detection

Performance characteristics change with image size. The dominant bottleneck shifts:

| Size class | Typical bottleneck | What wins |
|---|---|---|
| ≤ 512×512 | Compute / instruction throughput | SIMD vectorization, IPC |
| ~2048×2048 | L2/L3 cache pressure | Tile size, prefetch, reuse distance |
| ≥ 8192×8192 | Memory bandwidth | Tiling strategy, sequential access, fewer passes |

An algorithm that beats libvips at 512 can fall behind at 8K for a completely different
reason. The perf engineer must always measure all three sizes and look for **crossover
patterns** — where the winner changes across size classes.

### How to detect a crossover

Run all three sizes and build a ratio table:

```
op         | 512px ratio | 2048px ratio | 8192px ratio | pattern
-----------|-------------|--------------|--------------|--------
invert     | 0.60 ✅     | 0.72 ✅      | 1.40 🟡      | crossover at scale → tiling/bandwidth issue
thumbnail  | 0.90 ✅     | 1.80 🟠      | 4.20 🔴      | degrades at scale → shrink-on-load path difference
gauss_blur | 0.50 ✅     | 0.55 ✅      | 0.62 ✅      | win across all sizes → NEON path is correct
```

**Crossover task filing rule:**
- If viprs wins at small (≤ 1.00) but loses at large (> 1.00): file with the **worst** ratio
  as the headline, note the crossover explicitly, and flag that the fix is likely
  **tiling/memory-access pattern**, not SIMD.
- If viprs loses at all sizes: file the 8192 ratio as headline — it is the most impactful.
- If viprs wins at all sizes: no task needed. Document in the evidence section of any
  related archived task.

### Interpreting scaling patterns

| Pattern | What it means | Fix direction |
|---|---|---|
| Ratio improves at larger sizes | viprs parallelises better; libvips hits lock contention | Confirm; protect |
| Ratio worsens at larger sizes | Memory bandwidth bottleneck or tiling mismatch | Smaller tile size, streaming access order |
| Ratio flat across sizes | Compute-bound; same algorithm | SIMD or algorithmic improvement |
| Ratio unstable (varies > 30%) | Benchmark noise or OS scheduling | More iterations; pin threads; rerun |

---

## Why Rust should win

Before filing a gap task, always state which Rust advantage is being blocked and why.
The developer reading the task must understand what to exploit, not just that viprs is slow.

| Rust advantage | How it translates to speed |
|---|---|
| No GC / deterministic drop | Zero pause, no stop-the-world in pixel path |
| Monomorphization | Compiler inlines and vectorizes per concrete type — no vtable overhead |
| `&mut [T]` slices | Pre-allocated, cache-hot, no malloc per tile |
| Rayon data parallelism | Work-stealing thread pool, zero OS-thread overhead per tile |
| NEON/AVX intrinsics without FFI | No C-call overhead; inlined by LLVM at link time |
| Ownership model | Zero-copy tile handoff between pipeline stages |
| `const` + `#[inline(always)]` | LLVM eliminates branches that libvips pays at runtime |

---

## Input contract

Before starting, read:

1. `GUIDELINES.md` — coding rules, style, architecture constraints.
2. `AGENTS.md` — non-negotiable rules and architectural invariants.
3. `docs/PERFORMANCE.md` — methodology, tooling, invariants, and case studies.
3. Active issue tracker: `# list active tasks` — to avoid duplicate filings.

Do NOT read ADRs proactively.
Do NOT touch `xtask/` files — the user may be editing them.

---

## The four audit passes

Run all four passes. File issues after completing all passes.

---

### Pass 1 — Gap audit (viprs vs libvips)

**Goal:** find every operation where viprs is slower than libvips (ratio > 1.00),
or where no baseline exists yet.

**Steps:**

1. List all operations that have a benchmark scenario. Check `xtask/src/bench/` and
   `xtask/src/bench.rs` (whichever exists):
   ```bash
   grep -rn "Op\|scenario\|op_name" xtask/src/bench/ 2>/dev/null | grep -v "Binary\|target" | head -40
   ```

2. For each benchmarked op, run **all three standard sizes** (the bench tool cycles
   through 512, 2048, 8192 automatically):
   ```bash
   cargo xtask bench tests/fixtures/images/bench_2048x2048.jpg <op> [args] \
     --no-e2e --iterations 20
   # Also run with the large fixture for production-realistic numbers:
   cargo xtask bench tests/fixtures/images/bench_8192x8192.jpg <op> [args] \
     --no-e2e --iterations 10
   ```
   Large-image results reveal memory bandwidth and cache pressure effects that
   are invisible at 512×512. Always include them.

3. **Build a ratio table across all three sizes.** For each op, record:
   ```
   op | 512px ratio | 2048px ratio | 8192px ratio | pattern (flat/improving/degrading/crossover)
   ```
   Use the **worst** ratio as the headline for any task filed. Note the pattern explicitly —
   it determines the fix direction (see Scaling behaviour section above).

   Flag:
   - Any ratio > 1.00 at any size as a gap.
   - Any crossover (wins small, loses large) as `high` priority regardless of the small ratio.
   - Any op where ratio degrades monotonically with size as `high` — it will only get worse
     in production workloads.

4. For each flagged gap, gather profiling evidence.
   **Always use `--ai` for machine-readable output** (no GUI/browser needed):
   ```bash
   # CPU top-functions — which function is hottest? (local, no Docker)
   cargo xtask profile tests/fixtures/images/bench_2048x2048.jpg <op> [args] --ai

   # SIMD coverage — is the binary vectorized?
   cargo xtask perf tests/fixtures/images/bench_2048x2048.jpg <op> [args] \
     --metrics simd

   # Allocation sites — top allocators in the pixel path (structured output)
   cargo xtask perf tests/fixtures/images/bench_2048x2048.jpg <op> [args] \
     --metrics alloc --ai

   # Per-node stage timing (thumbnail only, JSON output)
   cargo xtask bench tests/fixtures/images/bench_2048x2048.jpg thumbnail 400 \
     --profile-stages --ai --no-e2e --iterations 10

   # Per-function cache-miss table (Docker, optional — only if gap > 1.50x and unexplained above)
   cargo xtask profile tests/fixtures/images/bench_2048x2048.jpg <op> [args] \
     --tool cachegrind
   ```
   Record the output verbatim — it becomes the evidence in the task description.

5. For each op in `src/domain/ops/` that has NO benchmark scenario, flag it as
   "missing baseline" — it cannot be known whether viprs is winning or losing.

---

### Pass 2 — Benchmark honesty audit

**Goal:** find scenarios where the benchmark does not measure what it claims, making
the ratio meaningless.

Apply the **benchmark invariant** (from `docs/PERFORMANCE.md`) to every existing scenario:

- [ ] **Same input file**: both sides receive the exact same path and file.
- [ ] **Same algorithm**: if libvips selects a faster code path (e.g., shrink-on-load,
      JPEG DCT reduction, SIMD variant gated by runtime detection), viprs must match it
      or the difference must be explicitly noted as a known gap — not a fair comparison.
- [ ] **Same work per iteration**: if one side decodes from disk, the other must too.
      If one side caches across iterations, the other must be in the same cache state.
- [ ] **Same thread count**: controlled by the scheduler.
- [ ] **Cache state symmetry**: libvips op-cache (`vips_cache`) is disabled in no-cache
      mode via `vips_cache_set_max(0)`. Verify this is present in `libvips_runner.c`.
      Viprs tile cache must be in the matching state (off for no-cache, warm for with-cache).

For each violated invariant, file a task. The invariant violation is the finding —
the ratio is not meaningful until the invariant is restored.

---

### Pass 3 — Missing scenario audit

**Goal:** identify benchmark scenarios that should exist but don't. Real-world
diversity of inputs and configurations must be represented. Large images (8192×8192)
are mandatory — gaps only visible at scale must not be hidden by small fixtures.

For each of the following categories, check whether a scenario exists:

**Input diversity:**
- [ ] Small image (≤ 512×512)
- [ ] Medium image (2048×2048)
- [ ] Large image (≥ 8192×8192) — **mandatory for every op; file a task if missing**
- [ ] 1-band (grayscale)
- [ ] 3-band (RGB)
- [ ] 4-band (RGBA)
- [ ] 16-bit input
- [ ] Float input

**Operation diversity:**
- [ ] Arithmetic (add constant, multiply, invert)
- [ ] Resize / thumbnail (downscale ≥ 4×)
- [ ] Resize (upscale)
- [ ] Convolution (gaussian blur, sharpen)
- [ ] Colour conversion (RGB → LAB, RGB → HSV)
- [ ] Codec E2E (JPEG decode + op + encode)
- [ ] Pipeline with 3+ chained ops
- [ ] Full production pipeline use cases

**Configuration diversity for ops that have options:**
- [ ] Thumbnail with different target sizes (e.g., 100, 400, 800)
- [ ] Blur with different sigma values

For each missing scenario:
- Is it missing because the op is not implemented? → note it as a gap in the task.
- Is it missing even though the op exists? → file a task to add the scenario.

---

### Pass 4 — Rust advantage audit

**Goal:** identify operations that are still using scalar code or suboptimal patterns
when a Rust-native approach could beat libvips.

For each operation, check:

- [ ] **SIMD coverage**: run `cargo xtask perf <image> <op> --metrics simd`. If the
      SIMD ratio < 60% for an arithmetic/convolution/resample op, flag it — libvips
      uses hand-written C SIMD and viprs must respond with NEON/AVX intrinsics.
- [ ] **Per-tile allocations**: run `--metrics alloc`. Any allocation inside the pixel
      path is a regression waiting to happen at large image sizes. File a task if > 0
      allocs per tile.
- [ ] **Monomorphization gaps**: search for `dyn Op` or `Box<dyn>` in hot paths
      (`src/domain/ops/`, `src/adapters/scheduler/`). Each one is a potential vtable
      call that blocks inlining. File a task for each instance that is not justified
      by the plugin-registry exception in AGENTS.md.
- [ ] **Thread utilisation**: at 8192×8192, check whether viprs saturates all cores.
      If `cargo xtask bench` shows viprs p95 >> libvips p95 on large images, the
      scheduler may be serialising tiles — file a task.
- [ ] **Memory bandwidth**: if SIMD ratio is high but ratio > 1.00, the bottleneck is
      likely memory bandwidth. Note this in the task — the fix is tiling strategy
      (smaller tiles → better L1/L2 locality), not more SIMD.

---

## Output format

File one issue per confirmed finding:

```bash
# create issue for the gap
  --priority <high|medium|low> \
  -l performance \
  -d "Type: Gap | Missing baseline | Benchmark lie | Missing scenario | Rust advantage unused

Pass: <1 | 2 | 3 | 4>

Scaling table:
  512px:   <ratio>   (viprs p50 / libvips p50)
  2048px:  <ratio>
  8192px:  <ratio>   ← headline (worst / most production-relevant)
  Pattern: flat | improving-at-scale | degrading-at-scale | crossover-at-scale

Rust advantage blocked: <which advantage from the table above is not being exploited>
Fix direction: <SIMD | tiling/memory-access | monomorphization | thread-utilisation | algorithm>

Evidence:
<paste the exact command and its output, or the invariant check that failed>

Finding:
<what is wrong or missing, stated precisely; if crossover — note the size at which it flips>

Impact:
<what a reader of the ratio would incorrectly believe if this is not fixed>

Acceptance criteria:
- [ ] ratio ≤ 1.00 at all standard sizes (512 / 2048 / 8192)
- [ ] no crossover (viprs must not lose at large sizes even if it wins at small)
- [ ] The specific profiling command above shows improvement
- [ ] The benchmark invariant holds for this scenario
- [ ] Ratio table documented before and after any fix"
```

**Priority:**
- `high` — invariant violated (ratio is a lie), or gap > 1.50x at any size, or **crossover** (wins small / loses large), or missing baseline for a core op
- `medium` — gap 1.01x–1.50x at any size, or ratio degrading monotonically with size (even if still ≤ 1.00), or missing large-image scenario for an implemented op
- `low` — missing scenario for an unimplemented op, or cosmetic invariant issue

Do not file tasks for gaps already in the issue tracker with accurate priority and evidence.
Do file tasks where the existing entry lacks the benchmark output as evidence.
Do not file tasks for operations already confirmed faster than libvips (ratio < 1.00) — document
those wins in the task description of the corresponding archived task instead.

---

## What the performance engineer must NOT do

- Fix any code or benchmark.
- Touch `xtask/` files — the user may be editing them.
- File tasks based on intuition — every task must have a measured number or a specific
  invariant check as evidence.
- Accept ratio ≤ 2.00 as "good enough" — the target is ≤ 1.00 at **all** sizes.
- Report a single ratio without the full scaling table (512 / 2048 / 8192) — a single
  number hides crossovers and degradation patterns.
- Interpret a crossover as "viprs wins at small" — it means the algorithm breaks at scale.
- Interpret a ratio charitably: if libvips uses shrink-on-load and viprs does not,
  the ratio is a lie, not a "known limitation".
- Assume a benchmark is honest because it was written recently.
- Skip the 8192×8192 fixture — gaps at scale are the most important gaps to find.
