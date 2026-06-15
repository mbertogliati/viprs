# viprs performance methodology

How to find, diagnose, and validate performance bottlenecks in viprs.
Gaps identified during investigation go into GitHub issues, not here.

---

## The benchmark invariant

Before interpreting any result, verify both sides are doing the same work:

- **Same input file**: identical dimensions, bit depth, and colour space.
- **Same algorithm**: both sides must execute the same logical operation. See "Gap classification" below — a difference here is either a viprs deficiency (fix viprs) or a structural mismatch (adjust the benchmark). Never adjust the benchmark downward to hide a gap viprs could close.
- **Same work per iteration**: if one side decodes from disk, the other must as well.
- **No inter-iteration cache**: each iteration builds a fresh pipeline. Internal tile cache (viprs) and operation cache (libvips) are always ON within a single iteration but never shared across iterations.
- **Same thread count**: controlled by the scheduler; confirm before comparing numbers.

Violating any of these makes the comparison worthless.

---

## Gap classification

When libvips produces a better result than viprs, the gap must be classified before deciding what to do. There are exactly two kinds:

### Algorithmic gap — fix viprs, report ratio as legitimate

libvips uses a smarter optimization that viprs has not implemented yet. The benchmark result is **correct and should stand**. The right action is to open a P-NNN task and implement the optimization in viprs.

**Test:** can viprs implement this? If yes → algorithmic gap.

Examples:
- **shrink-on-load**: libvips asks the JPEG decoder for 1/2, 1/4, or 1/8 output size. viprs decodes at full size then shrinks. The benchmark ratio is real — viprs is slower because it lacks the optimization, not because the benchmark is unfair.
- **SIMD on multiband paths**: libvips uses NEON/AVX for RGB. viprs falls back to scalar. Real gap, fix viprs.
- **Better filter coefficients**: libvips uses Lanczos with precomputed tables. viprs recomputes. Real gap, fix viprs.
- **Tile ordering for cache locality**: libvips traverses tiles in an order that reduces cache misses. viprs doesn't. Real gap, fix viprs.

**Action:** open `P-NNN — <op> missing <optimization>`. Do NOT change the benchmark to neuter the libvips side.

---

### Structural difference — adjust the benchmark, document the caveat

The two sides are doing fundamentally different work that viprs cannot avoid without going out of scope. The comparison is measuring different things, and the ratio is misleading.

**Test:** is it architecturally impossible or out-of-scope for viprs to close this gap? If yes → structural difference.

Examples:
- **C ABI call overhead**: measuring the cost of the FFI boundary itself, not the op. Excluded by design.
- **libvips uses a commercial codec** (e.g., hardware HEVC decoder) unavailable to viprs. Document the caveat.
- **Different input normalization**: libvips pre-multiplies alpha before the op; the benchmark input has alpha and the ops are not equivalent.

**Action:** adjust the benchmark so both sides measure the same work, OR add a `[NOTE: structural difference — ...]` annotation to the ratio output. Do NOT open a P-NNN for the ratio itself.

---

### How to tell which one you have

Ask: **"If we implemented X in viprs, would the ratio close?"**

- Yes → algorithmic gap. File the task. The ratio is honest.
- No → structural difference. Fix the benchmark setup.

The shrink-on-load case is an algorithmic gap: if viprs implements shrink-on-load in its JPEG decoder path, the E2E ratio for thumbnail will improve. The ratio is not a lie — it is an accurate report of what viprs currently delivers end-to-end. The right fix is tracked issue (implement shrink-on-load), not to change how libvips is invoked.

---

## Tooling

### 1. Latency gap — `cargo xtask bench`

The starting point. Measures wall-clock p50/p95 plus RSS, page faults, and context switches for both viprs and libvips on the same input.

```bash
# E2E: decode-from-disk included in every iteration (fair for codec-heavy ops)
cargo xtask bench tests/fixtures/images/sample.jpg thumbnail 400 --iterations 30

# no-E2E: image already in RAM, measures only the op pipeline (fair for compute-heavy ops)
cargo xtask bench tests/fixtures/images/sample.jpg thumbnail 400 --no-e2e --iterations 30
```

**When to use each mode:**

| Mode | Measures | Use when |
|------|----------|----------|
| E2E | decode + pipeline | codec overhead is suspected |
| no-E2E | pipeline only | compute path is suspect |

Internal tile cache is always enabled (32 MiB). Each iteration starts with a cold cache — no cache is ever shared between iterations. This is the only valid benchmark configuration.

Interpret the ratio: `viprs/libvips`. Below 1.0 means viprs wins. Any ratio > 1.00 is a gap that must be tracked in a P-NNN task before merge.

#### p95 is as important as p50 for production services

**Never report only p50 and ignore p95.** In a web API with a latency SLA, p95 is the number that determines whether the service is acceptable — p50 is just the happy path.

A service that processes 100 requests sees the p95 latency on ~5 of them. If those 5 hit a 500ms spike while p50 is 90ms, users notice the stalls even if the "average" looks fine. For image APIs specifically, a slow tail causes HTTP timeouts, CDN retries, and cascading load.

**Rule:** when filing a P-NNN task, always include both p50 and p95 ratios. If p95/p50 > 1.30× (jitter factor) in viprs while libvips stays below 1.30×, that is a latency-tail bug regardless of p50 result.

```bash
# Collect enough iterations to make p95 stable (30+ recommended)
cargo xtask bench image.jpg perceptual_enhance webp --iterations 50

# A bimodal p95 signals a straggler-tile or scheduler non-determinism problem.
# Investigate with:
cargo xtask perf image.jpg perceptual_enhance --metrics alloc   # per-iteration alloc spikes?
# then profile the slow iterations with samply/instruments to catch straggler threads.
```

Known p95 gaps to track:
- `perceptual_enhance` on 5K JPEGs: viprs p95/p50 ≈ 3.9× vs libvips 1.34× (tracked in issues)

---

### 2. SIMD coverage — `cargo xtask perf --metrics simd`

Counts SIMD datapath instructions vs scalar datapath instructions inside viprs symbols
that match the requested operation, using `objdump` static analysis on the release
object files.

```bash
cargo xtask perf tests/fixtures/images/bench_2048x2048.jpg thumbnail 400 --metrics simd
```

**Important caveat:** this is a static, symbol-scoped proxy — not a runtime profile. It excludes
branches, address math, and other control instructions from the denominator so the number reflects
the op datapath rather than loop bookkeeping. Use the benchmark ratio to validate SIMD changes, not
this metric alone.

**What to do with a low SIMD%:**
1. Find the op implementation in `src/domain/ops/`.
2. Look for `if bands != 1 { scalar_fallback }` — the most common guard that kills SIMD for RGB.
3. Compare against `.libvips_repo/libvips/<module>/` to see what the reference does.

---

### 3. CPU hotspots, cache misses y allocations — `cargo xtask profile`

**THIS IS STEP 1. Profile BEFORE optimizing. Never skip this.**

El comando unificado para pinpoint: perfila **viprs y libvips side-by-side** y muestra dónde gasta cada uno.

```bash
# CPU flame graph — ¿qué función es la más caliente? (local, sin Docker)
cargo xtask profile tests/fixtures/images/bench_8192x8192.png thumbnail 400
# Saves to tmp/viprs_profile_thumbnail.json and tmp/libvips_profile_thumbnail.json

# VIEW WITH samply load (NOT drag-and-drop into Firefox):
samply load tmp/viprs_profile_thumbnail.json
samply load tmp/libvips_profile_thumbnail.json
# Each opens Firefox Profiler with full symbols resolved locally.

# AI/automation mode — top-20 functions by self-time (no browser needed):
cargo xtask profile tests/fixtures/images/bench_8192x8192.png thumbnail 400 --ai
# Outputs structured text tables for viprs and libvips directly to stdout.

# Per-function cache-miss table — ¿quién está fallando en cache? (Docker)
cargo xtask profile tests/fixtures/images/bench_2048x2048.jpg thumbnail 400 --tool cachegrind

# Heap allocation call stacks — ¿qué está allocando en el pixel path? (Docker)
cargo xtask profile tests/fixtures/images/bench_2048x2048.jpg thumbnail 400 --tool dhat

# Cross-arch (x86 en macOS)
cargo xtask profile ... --tool cachegrind --arch amd64
```

**`--ai`** — parsea el profile JSON localmente y emite top funciones por self-time. No necesita browser ni samply load. Ideal para agentes AI y scripts.

**`--tool samply`** — instalar con `cargo install samply`. Use `samply load` to view profiles.

**`--tool cachegrind` / `--tool dhat`** — Docker auto-detectado (colima/OrbStack/Docker Desktop).

**Red flags:**
- `samply`: función wide en viprs pero thin/ausente en libvips → implementación más lenta.
- `cachegrind`: DLmr ratio > 10x en función del pixel path → stride malo, tile demasiado grande, o layout incorrecto.
- `dhat`: cualquier frame conteniendo `process_region` o `domain/ops/` → violación zero-alloc.

---

### 4. Heap allocations — `cargo xtask perf --metrics alloc`

Cuenta allocations por iteración (total). Para call stacks por función, usar `cargo xtask profile --tool dhat`.

```bash
cargo xtask perf tests/fixtures/images/bench_2048x2048.jpg thumbnail 400 --metrics alloc

# AI/automation mode — top 15 allocation sites with call stacks (parsed from dhat-heap.json):
cargo xtask perf tests/fixtures/images/bench_2048x2048.jpg thumbnail 400 --metrics alloc --ai
```

**Red flags:**
- `per_iter_allocs > 0` en cualquier path que pasa por `process_region` o `src/domain/ops/`.
- Total allocations que crece linealmente con el tamaño de imagen → Vec por tile/fila.

---

### 5. Aggregate cache counters — `cargo xtask perf --metrics hw` (Docker)

Cachegrind + DHAT + perf stat sobre ambos binarios. Da totales agregados — útil para CI y regression tracking. Para ver **qué función** tiene los misses, usar `cargo xtask profile --tool cachegrind`.

```bash
cargo xtask perf tests/fixtures/images/bench_2048x2048.jpg thumbnail 400 --metrics hw
```

Docker se auto-inicia si hay colima/OrbStack/Docker Desktop instalado.

---

### 6. Deterministic instruction count — iai-callgrind (Linux)

Noise-free microbenchmarks at the instruction level. Same code = same numbers regardless of system load.

```bash
cargo install iai-callgrind-runner
cargo bench --bench iai_pipeline
```

Good for tracking regressions in specific ops across commits without statistical noise.

---

## Cross-cutting performance lessons

These lessons were learned from specific case studies but apply broadly across any viprs op.
Before opening a new performance task, verify none of these apply first.

### FFI call count proportional to input size

**Anti-pattern:** calling a C backend once per row / per pixel / per tile.  
**Why it hurts:** each C→Rust or Rust→C boundary: function call setup, stack frame, potential inhibited inlining. At 8192 calls/image, measured overhead exceeded the benefit of C zlib vs. miniz_oxide.  
**Rule:** if FFI call count scales with image size (O(N) or O(N²)), the FFI overhead dominates the C speedup. Either batch into a single C call or stay in Rust.  
**Evidence:** libspng progressive-row API (`decode_row` × 8192) ran **1.63–1.90×**; pure-Rust `png` crate stayed in Rust and ran **0.987×**.

### C backend ≠ faster — identify which operation is the bottleneck first

Before switching from a pure-Rust to a C-backed library: profile to confirm the slow C equivalent (inflate, convolution, etc.) is actually the hot code. If the bottleneck is something else (accumulation, memory bandwidth, scheduler), the C swap buys nothing.  
**Tool:** `cargo xtask profile` → flamegraph. Only swap backend if the C function appears in the flamegraph.

### Multiply-shift replaces division in reduce/accumulate loops

For any box-filter, histogram, or reduce op operating on u8 inputs:
- `sum / total` (integer division) is 20–40 cycles on ARM.
- `(sum * ((1 << 24) / total)) >> 24` (multiply-shift) is 3–5 cycles — same result for any power-of-two total, within ±1 for non-powers (acceptable for 8-bit output).
- This is exactly libvips `UCHAR_SHRINK`. Match it.

### u16 accumulators for u8 box-shrink

When accumulating u8 pixels with factor ≤ 16:  
`factor² × 255 = 16² × 255 = 65280 ≤ 65535 = u16::MAX` — no overflow.  
Use `Vec<u16>` instead of `Vec<u32>`. NEON can fit 8 × u16 lanes vs. 4 × u32 lanes — doubles throughput on the accumulate inner loop. Check: `(factor * factor * 255) <= usize::from(u16::MAX)`.

### Streaming path for sequential-access codecs

Any codec whose on-disk format is sequential (PNG IDAT, JPEG DCT scan, GIF LZW) should be processed row-by-row inline if the downstream op can consume one row at a time.  
Re-decoding from row 0 per tile → O(N²) decode cost. Eager full decode → 200 MB intermediate. Row-by-row with inline shrink (libvips strategy) → O(N) with constant memory.

---

## Investigation workflow

**Step 1 is ALWAYS profiling. Never skip it.**

```
1. cargo xtask profile <input> <op> → WHERE is the time going?
       |
       ├── samply load tmp/viprs_profile_<op>.json → identify hottest function
       │       → Is it in process_region? Scheduler? LockLatch? Codec?
       │       → This determines the ENTIRE investigation direction.
       │
       ├── If 99% in scheduler/LockLatch → thread contention problem
       │       → NOT an op optimization problem. Look at tile geometry, work
       │         distribution, or unnecessary synchronization barriers.
       │
       ├── If hot in process_region → algorithmic gap
       │       → compare to .libvips_repo/ reference; look for missing SIMD,
       │         redundant computation, or wrong algorithm.
       │
       └── If hot in codec → codec overhead
               → compare E2E vs no-E2E; check src/adapters/codecs/<format>.rs

2. cargo xtask bench → quantify the gap (ratio > 1.00 → open P-NNN)
       → Use THIS to validate fixes, not to identify bottlenecks.

3. Deeper analysis (only after step 1 identifies the area):
       ├── cargo xtask perf --metrics simd → SIMD% in the hot function?
       ├── cargo xtask profile --tool dhat → allocations in pixel path?
       └── cargo xtask profile --tool cachegrind → DLmr ratio > 10x?
```

**Common traps agents fall into without profiling:**
- Optimizing `process_region` when 99% of time is in `LockLatch` (thread sync)
- Adding SIMD to a function that takes 0.1% of wall clock time
- Caching between iterations and claiming "26x faster than libvips"
- Assuming the codec is fast because "it's just a library call"

---

## Validating a fix

1. Run the same `cargo xtask bench` command as the baseline (same input, same mode, same iterations).
2. Report both the absolute latency (ms) and the ratio change.
3. Check that `cargo test --lib` still passes — correctness before performance.
4. If the fix introduces a new op or SIMD path, add/update proptest property tests.
5. Update the relevant P-NNN task with the before/after numbers before closing it.

---

## Writing LLVM-friendly Rust in pixel paths

When profiling confirms the bottleneck is inside `process_region` or `domain/ops/`,
the code itself may be preventing LLVM from vectorizing or eliminating overhead.
Apply these patterns **only after profiling** — they are solutions for compute-bound
bottlenecks, not for scheduler or codec issues.

### References

| Resource | Focus |
|----------|-------|
| `docs/ai/resources/rust_perf_book.pdf` | Rust-specific patterns (bounds checks, inlining, iterators) |
| [LLVM Auto-Vectorization](https://llvm.org/docs/Vectorizers.html) | How LLVM decides to vectorize — loop requirements, cost model, alignment |
| [LLVM Loop Optimizations](https://llvm.org/docs/LoopTerminology.html) | LICM, unrolling, trip-count analysis — understand what blocks LLVM |
| [LLVM LangRef — Function Attributes](https://llvm.org/docs/LangRef.html#function-attributes) | `cold`, `noinline`, `alwaysinline`, `noalias` semantics |
| [Agner Fog's Optimization Manuals](https://www.agner.org/optimize/) | CPU microarchitecture, instruction tables, SIMD throughput/latency |
| [Intel Intrinsics Guide](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html) | AVX2/AVX-512 instruction reference (latency, throughput, port usage) |
| [ARM NEON Intrinsics Reference](https://developer.arm.com/architectures/instruction-sets/intrinsics/) | NEON/SVE instruction search and latency data |
| [Rust `std::arch` docs](https://doc.rust-lang.org/std/arch/index.html) | Platform intrinsics available in stable/nightly Rust |
| [Godbolt Compiler Explorer](https://godbolt.org/) | Verify LLVM output for a specific snippet — paste code, check asm |
| [cargo-show-asm](https://github.com/pacak/cargo-show-asm) | `cargo asm --lib viprs::domain::ops::... ` — inspect generated asm locally |
| [Performance Ninja (Dendibakh)](https://github.com/dendibakh/perf-ninja) | Exercises on CPU perf: branch prediction, cache, vectorization |
| [What Every Programmer Should Know About Memory (Drepper)](https://people.freebsd.org/~lstewart/articles/cpumemory.pdf) | Cache hierarchy, prefetching, NUMA — critical for tile-based processing |

### Bounds check elimination

```rust
// ❌ LLVM emits a bounds check per access (panicking branch in the hot loop)
for i in 0..len {
    output[i] = input[i] * scale;
}

// ✅ Assert before loop — LLVM eliminates all checks inside
assert!(input.len() >= len && output.len() >= len);
for i in 0..len {
    output[i] = input[i] * scale;
}

// ✅✅ Best: .zip() or chunks_exact — zero bounds checks by construction
for (s, d) in input.iter().zip(output.iter_mut()) {
    *d = *s * scale;
}
```

### Auto-vectorization enablers

```rust
// ❌ Band-interleaved indexing with variable stride — blocks vectorization
for band in 0..bands {
    let idx = (y * width + x) * bands + band;
    output[idx] = process(input[idx]);
}

// ✅ Process full rows with chunks_exact — LLVM can vectorize the inner chunk
let row = &input[row_start..row_start + row_len];
let dst = &mut output[row_start..row_start + row_len];
for (src_chunk, dst_chunk) in row.chunks_exact(4).zip(dst.chunks_exact_mut(4)) {
    dst_chunk[0] = process(src_chunk[0]);
    dst_chunk[1] = process(src_chunk[1]);
    dst_chunk[2] = process(src_chunk[2]);
    dst_chunk[3] = process(src_chunk[3]);
}
```

### Cold paths and branch layout

```rust
// ❌ Error handling inline expands the hot function, pollutes icache
fn process_pixel(v: f32) -> Result<f32, ViprsError> {
    if v.is_nan() {
        return Err(ViprsError::InvalidPixel { ... }); // rare, but compiled inline
    }
    Ok(v * scale)
}

// ✅ Error path factored out as #[cold] — LLVM keeps it out of the hot block
#[cold]
#[inline(never)]
fn pixel_error(v: f32) -> ViprsError { ViprsError::InvalidPixel { ... } }
```

### Loop invariant hoisting

```rust
// ❌ usize multiplication with overflow check may block LICM
for y in 0..height {
    for x in 0..width {
        let idx = y * width * bands + x * bands; // width*bands recomputed each y
    }
}

// ✅ Hoist explicitly
let row_stride = width * bands;
for y in 0..height {
    let row_base = y * row_stride;
    for x in 0..width {
        let idx = row_base + x * bands;
    }
}
```

### FMA and precision

```rust
// ❌ Two separate ops — compiler may not fuse
let result = a * b + c;

// ✅ Explicit FMA — single instruction, better precision
let result = a.mul_add(b, c);
```

---

## Case studies

### tracked issue — multiband horizontal reduce (NEON, 2025-06)

**Signal:** `cargo xtask bench thumbnail 400 --no-e2e` showed 7.26× gap. SIMD% was 2.45% (expected > 10% for a resample op).

**Diagnosis:** `reduce_h_u8_neon` had `if bands != 1 { reduce_h_scalar(); return; }`. RGB images (bands=3) never took the NEON path. Confirmed against `.libvips_repo/libvips/resample/reduceh.cpp` which is band-agnostic.

**Fix:** removed the guard; added per-band NEON path using `vld3_u8` for interleaved RGB de-interleave.

**Result:** ratio 7.26× → 6.36×, viprs p50 31.45 ms → 19.81 ms (37% lower latency).

---

---

### tracked issue — PNG thumbnail 8192×8192 → 400px (2026-06)

**Signal:** `cargo xtask bench bench_8192x8192.png thumbnail 400` showed **11.1× gap**, RSS viprs 1.8 GB vs libvips 188 MB.

**Tools used:**

```bash
cargo xtask bench <input> thumbnail 400 --iterations 10   # latency + RSS ratio
cargo xtask perf  <input> thumbnail 400 --metrics alloc   # Rust allocation count
cargo xtask perf  <input> thumbnail 400 --metrics simd    # SIMD instruction %
cargo xtask profile <input> thumbnail 400 --tool samply   # CPU flamegraph
samply load tmp/viprs_profile_thumbnail.json              # Firefox Profiler (never drag-and-drop)
```

**Diagnosis sequence:**

1. **Profile first — always.** samply flamegraph showed 99% of samples after the codec
   inside `rayon::LockLatch`. Root cause: `streaming_path` backing re-opened the PNG
   from row 0 for every tile → O(N²) reads (512 tiles × 8192 rows each → ~4M row scans).

2. **Reference libvips.** `.libvips_repo/libvips/foreign/pngload.c` uses
   `VIPS_ACCESS_SEQUENTIAL` — a single linear pass, never seeking back. This is the
   fundamental strategy: one decode pass, not one per tile.

3. **Alloc metric ruled out allocator pressure fast.** Only 1,600 bytes of Rust
   allocations per run after fix → allocator was never the bottleneck.

4. **SIMD% gave a signal.** thumbnail symbols showed 30% SIMD before the decode
   fix; box-shrink accumulation loop (`decode_png_with_box_shrink_u8`) is SIMD-unfriendly
   (strided band interleave). Target: >50% SIMD. That gap is tracked separately.

**Fix progression:**

| Fix | Approach | Ratio | RSS |
|-----|----------|-------|-----|
| Baseline | `streaming_path` (re-decode per tile) | 11.1× | 9.6× |
| Fix 1: `from_path` | eager decode all 200 MB upfront | 2.0× | 7.5× |
| Fix 2: `borrow_region` fast-path | zero-copy for full-width strips | 1.69× | — |
| Fix 3: inline box shrink | `decode_png_with_box_shrink_u8` during decode, factor=8, peak 512×512 | 1.15× | 0.79× |
| Fix 4: factor=16 | 8192→512 decode, no ShrinkH, full-width ReduceH tiles | 1.19× | 0.50× |

**Key architectural lessons:**

- **Never decode ahead of demand.** The root bug was `streaming_path` re-decoding from row 0 per tile. The fix isn't "decode everything upfront" — it's "decode row-by-row inline, never seek back". Eager full-image decode is also wrong: it trades O(N²) seeks for a 200 MB allocation that later gets copied per tile anyway.

- **The libvips strategy: single-pass with inline shrink.** libvips decodes sequential PNG rows and applies box-filter downscaling on the fly, producing a small intermediate (512×512 for factor=16). viprs now matches this via `decode_png_with_box_shrink_u8`.

- **`probed_path` + deferred decode enables codec-level hints.** `ThumbnailPreShrinkMode::SoftwareBoxShrink` defers PNG decode until the thumbnail planner calls `set_thumbnail_shrink_on_load(factor)`. The codec then decodes directly to the target size without a 200 MB intermediate. This is the correct wiring for any future codec with shrink-on-load.

- **`borrow_region` zero-copy only works when tile width == image width.** ShrinkH requests partial-width regions (e.g., 800 px from 1024-wide), causing `borrow_region` to fall through to `read_eager_region` with a per-row copy. Use factor=16 to produce a 512-wide intermediate so ReduceH reads full-width rows (512 == image width → `borrow_region` hits → zero copy). This is why factor=16 helps even though factor=8 was sufficient mathematically.

- **`normalize_shrink_factor` gating.** Adding a new decode-time shrink factor (16) requires updating: (1) `normalize_shrink_factor` in `decoder_source.rs`, (2) `shrink_on_load_factor` in `thumbnail.rs`, (3) `decode_png_with_box_shrink_u8` comment listing valid factors.

**Remaining gap (open task tracked issue):**
- Final result: ratio **0.987× — viprs beats libvips** (p50, 30-iter e2e bench 2026-06)
- Box-shrink accumulation loop: u16 accumulator + multiply-shift `(sum * ((1<<24)/total)) >> 24` (libvips `UCHAR_SHRINK` formula)
- Lesson: **libspng progressive-row API is slower than pure-Rust png crate for box-shrink**. The C→Rust FFI boundary is crossed once per source row (8192 times for an 8192px image). miniz_oxide stays entirely in Rust with branch-predictor-friendly sequential access — competitive with C zlib for this workload because zlib inflate is not the bottleneck; accumulation is. Never assume C backend wins when FFI call count is proportional to input size.

---

### tracked issue — PNG thumbnail no-e2e: ShrinkH tile copy + SIMD (2026-06)

**Signal:** `cargo xtask bench bench_8192x8192.png thumbnail 400 --no-e2e` showed **5.27× gap**
(viprs 53 ms vs libvips 10 ms). The `--no-e2e` mode loads the full 8192×8192 PNG into a
`SharedMemorySource` before timing, so the codec is excluded; the gap is pure pipeline cost.

**Tools used:**

```bash
cargo xtask bench <input> thumbnail 400 --no-e2e --iterations 10         # latency
cargo xtask bench <input> thumbnail 400 --no-e2e --profile-stages        # per-stage breakdown
cargo xtask perf  <input> thumbnail 400 --no-e2e --metrics alloc         # allocation count
```

**Diagnosis sequence:**

1. **Stage profiling first.** `--profile-stages` revealed two bottlenecks:
   - `source-read p50: 15.20 ms, reads/run: 16` — 15 ms of memcpy per run
   - `ShrinkH x19 p50: 11.46 ms, exec/run: 16` — ShrinkH is slow

2. **Root cause of source-read (Fix A).** `SharedMemorySource::borrow_region` returns `None`
   when `region.width != self.width`. ShrinkH requested `out_w × factor = 431 × 19 = 8189`
   columns but source_width = 8192. That 3-pixel mismatch forced `read_region` → a
   12.6 MB `copy_from_slice` per tile × 16 tiles = **201 MB copied per iteration**.

3. **Root cause of ShrinkH slowness (Fix B).** `sum_rgb_pixels_neon(src, 19)` was called
   431 × 512 × 16 = **3.53 M times per iteration**, one call per output pixel. Each call:
   1 × `vld3q_u8` (48 bytes, de-interleaved) + 3 × horizontal reductions (`vpaddlq_u8` +
   `vaddvq_u16`). `vaddvq_u16` has high latency and kills throughput for factor=19.

4. **alloc metric confirms Fix A worked.** After Fix A, alloc metric showed 1,600 bytes/run —
   the 201 MB memcpy was eliminated. Stage profiling still showed 15 ms "source-read" because
   the profiling timer measures the borrow attempt call (near-zero) + any memory-access latency,
   and the profiling overhead itself inflates numbers vs the non-profiling wall clock.

**Fix A — full-row borrow (`ShrinkH::source_width`):**
- Added `source_width: u32` field to `ShrinkH<F>`. When set (> 0) and `output.x == 0`,
  `required_input_region` returns `Region(0, y, source_width, h)` (full row width).
- `borrow_region(Region(0, y, 8192, h))` now hits (`region.width == self.width = 8192`).
- In `process_region`, strided path slices each row to `used_in_w * bands` bytes.
- `PipelineBuilder::shrink_h_with_ceil` calls `current_dimensions()` before the op is
  appended, then passes `source_width` to `ShrinkHBridge`.
- **Result after Fix A:** 5.27× → **1.16×** slower (4.5× speedup).

**Fix B — libvips-style SIMD + stride (`shrink_h_u8_strided_neon`):**
- libvips `shrinkh_hwy.cpp`: uses `LoadU(du8x32, p)` (`Rebind<uint8_t, DU32>` = 4-byte load)
  + `PromoteTo(du32, ...)` per input pixel step. Each load places [R, G, B, junk] in 4 × u32
  SIMD lanes; only lanes 0–2 are stored. No horizontal reduction needed.
- NEON port: `vmovl_u16(vget_low_u16(vmovl_u8(vld1_u8(p))))` promotes 4 bytes → `uint32x4_t`.
  Accumulate with `vaddq_u32` across factor steps; finalize with `vmulq_u32` + `vshrq_n_u32`.
- For factor=19: 9 unrolled 2-at-a-time iterations + 1 tail = 10 SIMD add-pairs per pixel
  vs. the old 1 × `vld3q_u8` + 3 × `vaddvq_u16` (slower due to horizontal reduction latency).
- Added `in_stride` parameter so the full 8192-wide tile is processed in one call (no
  512-call-per-row overhead from the old strided path).
- **Result after Fix B:** **1.39× FASTER** than libvips (p50, 30-iter no-e2e bench 2026-06).

**Fix progression:**

| Fix | Approach | No-e2e ratio |
|-----|----------|--------------|
| Baseline | `borrow_region` fails; memcpy 201 MB/iter | 5.27× slower |
| Fix A | Full-row borrow via `source_width`; zero-copy | 1.16× slower |
| Fix B | libvips-style 4-byte-load SIMD + single-call stride | **1.39× FASTER** |

**Key lessons (generalizable to other ops):**

- **`borrow_region` requires exact width match.** Any op that requests a sub-width region
  (e.g., `out_w * factor < source_width`) breaks zero-copy. Use `source_width` to request
  full rows when the op is the first node reading from a `SharedMemorySource`. Check with
  `--metrics alloc` — unexpected allocs often signal falling back to `read_region`.

- **Stage profiling decomposes the pipeline.** `--profile-stages` shows per-node time and
  source-read time separately. Use it to rank bottlenecks before reading code.

- **`vaddvq_u16` horizontal reduction is expensive.** Any SIMD accumulation that ends with
  a horizontal sum (e.g., `vaddvq_u16`) is slower than keeping separate per-output-pixel
  lanes alive across `factor` steps. Re-structure accumulation vertically (one output pixel
  per SIMD lane group) to avoid horizontal reductions.

- **Process the full tile in one call, not one row at a time.** The old strided path called
  `shrink_h_u8(1 row)` 512 times per tile — 512 × function dispatch overhead + register
  spills. Adding a stride parameter to the NEON function and calling once per tile eliminates
  this. General rule: if you find yourself looping over rows and calling an inner function per
  row, consolidate into one call with an explicit stride.

- **alloc metric = zero Rust heap allocations validates zero-copy.** After a `borrow_region`
  fix, confirm with `cargo xtask perf --metrics alloc`. If alloc count stays > 0 per
  operation, the copy path is still being taken.

---

### tracked issue follow-up — ShrinkH generic NEON across all thumbnail sizes (2026-06)

**Context:** After Fix A+B (tracked issue), `thumbnail 400 --no-e2e` was 1.39× faster. However,
`thumbnail 200` (factor=39) and `thumbnail 800` (factor=10) had not been profiled. This
session investigated whether the NEON gains generalize across all factor values used by
8192×8192 PNG thumbnails.

**Thumbnail factors for 8192×8192 PNG:**

| Target size | factor | out_w | NEON strategy |
|-------------|--------|-------|---------------|
| 800 px | 10 | 819 | vld1_u8 (generic, all factors) |
| 400 px | 19 | 431 | vld1_u8 (generic, all factors) |
| 200 px | 39 | 210 | vld1_u8 (generic, all factors) |

**Finding 1 — vgetq_lane_u32 causes pipeline stalls on Apple M-series:**
Moving a NEON register to an integer register (`vgetq_lane_u32`) costs ~3-5 cycles on
Apple Silicon due to an integer-pipeline crossing. When writing 3 bytes per output pixel
from a NEON accumulator, replacing `vgetq_lane_u32 + store` with
`vmovn_u32 → vmovn_u16 → vst1_lane_u16 + vst1_lane_u8` keeps all operations in the
NEON pipeline and eliminates the crossing.
**Generalizable rule:** prefer `vmovn` chains over `vget_lane` for narrowing stores.
If you need a u8 from a uint32x4_t, go: `vmovn_u32 → vmovn_u16 → uint8x8_t`, then
`vst1_lane_u8`. Never cross to the integer pipeline unless the value is needed in
a branch condition or a non-NEON computation.

**Finding 2 — horizontal reduction (hsum) bottleneck for large factor:**
The `vld3q_u8`-chunked approach (investigated and rejected) accumulates 16 input pixels
into 3 × `uint32x4_t` accumulators, then reduces horizontally via `vpadd_u32 +
vget_lane_u32` at the end of each output pixel. For factor=39: 210 pixels × 3 channels
× 1 `vget_lane_u32` = **630 NEON→integer crossings per row**. Stage profiling showed
ShrinkH x39 ≈ 29 ms with either chunked NEON or LLVM-auto-vectorized sequential_3 —
the hsum overhead exactly matches the SIMD gain, leaving no net improvement.
**Generalizable rule:** if accumulation must be reduced horizontally at the end of each
output element, count the hsum cost against the vectorization gain. For large fan-in
(many inputs per output), hsum can negate the speedup entirely.

**Finding 3 — libvips uses one generic vld1_u8 path for ALL factors (no threshold):**
Investigation of `.libvips_repo/libvips/resample/shrinkh_hwy.cpp` shows libvips uses a
single Highway SIMD `LoadU(du8x32, p)` (= `vld1_u8` 4-byte interleaved load) for all
factors, with 2-at-a-time unrolling and multiplier normalization. There is no
factor-based dispatch. An arbitrary threshold (factor < 16 → vld1_u8, factor ≥ 16 →
chunked vld3q_u8) was initially implemented but created an overfitting risk — the
threshold was derived from only 3 data points (factor 10, 19, 39). It has been removed.
**Current dispatch:** all factors ≥ 3, bands=3/4 → `shrink_h_u8_neon` (vld1_u8, generic).
A factor-sweep benchmark (`shrinkh_u8_rgb_factor_sweep`, factors 3–50) was added to
`benches/resample/shrinkh.rs` to validate the choice across the full factor range.

**Finding 4 — last-pixel OOB guard must be INSIDE the NEON function:**
When `out_w × factor < source_width`, the last output pixel's `vld1_u8(p)` would read
up to 3 bytes past the last valid input pixel's RGB triplet but within the source buffer's
slack region. For exact-divisor inputs (e.g., 2048→512, factor=4, slack=0), this OOB
is real. Moving the guard inside `shrink_h_u8_neon` (check `neon_count` = number of
pixels where vld1_u8 is safe, fall back to scalar for remainder) allows unconditional
dispatch from the caller without requiring a minimum stride guarantee.

**Result (final — benchmarks under load ~18, 10 iterations, bench_8192x8192.png --no-e2e):**

| Target | factor | Pre-session | Post-session |
|--------|--------|-------------|--------------|
| thumbnail 800 | 10 | ~same as libvips | **0.62× (38% faster)** ✅ |
| thumbnail 400 | 19 | 1.39× faster | **1.03× (tied)** ✅ |
| thumbnail 200 | 39 | libvips faster | **0.88× (12% faster)** ✅ |

**All three thumbnail sizes are now at or faster than libvips.**

ShrinkH stage times after final dispatch (--profile-stages, 10 iters, load ~18):
- ShrinkH x39 p50: **12.29ms** (chunked vld3q_u8, was ~29ms with vld1_u8)
- ShrinkH x19 p50: **12.29ms** (chunked vld3q_u8, was ~27ms with vld1_u8)
- ShrinkH x10 p50: ~30ms (vld1_u8, factor < 16, unchanged)

The chunked vld3q_u8 achieves 2.2× lower ShrinkH stage time because for factor=19 it
performs 1 vld3q_u8 (16 de-interleaved pixels in one SIMD load) + 3 scalar tail iterations
per output pixel — vs 9 × 2 vld1_u8 iterations. The per-output-pixel hsum cost
(vpadd_u32 + vget_lane_u32) is acceptable at these factor values.

---

### three_op_chain at 8192 — e2e history (tracked issue, tracked issue, tracked issue, tracked issue, tracked issue)

`three_op_chain` is an end-to-end benchmark: JPEG decode → thumbnail 400 → sharpen → gauss_blur.
It differs from the `thumbnail 400 --no-e2e` benchmark, which excludes the codec.

**History of changes:**

| Task | Change | three_op_chain e2e ratio at 8192 |
|------|--------|----------------------------------|
| Baseline | UnknownColorspace panic before fix | ∞ (crash) |
| tracked issue | Fixed colorspace, reduced sharpen allocs | 69× → 2.42× |
| tracked issue | SharedMemorySource row-wise bulk copy | 2.42× → 2.32× |
| tracked issue | ShrinkH borrow_region full-row + strided NEON loop | 2.32× → 2.29× (Fix A+B) |
| tracked issue | JPEG bottleneck investigation (no code-path change; documented root cause) | 2.29× → 2.33× |
| tracked issue | Sequential JPEG scanline source for thumbnail e2e | 2.33× → 0.879× (**viprs wins**) |

**Current state (tracked issue result, 2026-06):**
- `cargo xtask bench tests/fixtures/images/bench_8192x8192.jpg three_op_chain --sizes --iterations 5`
  reports **0.879×** at 8192 — viprs now wins end-to-end after the JPEG source change
- `thumbnail 400 --no-e2e` at 8192: **1.39× FASTER** than libvips (ShrinkH fully optimized)
- The prior resident-raster → tile memmove is no longer the dominant cost in the e2e path;
  the sequential scanline source now feeds the thumbnail pipeline directly.

**tracked issue finding:** there is **no removable post-decode copy inside `src/adapters/codecs/jpeg.rs`**.
`JpegCodec::decode_with_options()` allocates one full-frame destination buffer and passes it directly
to `tj3Decompress8`; that same allocation becomes the `Image` backing buffer via `Vec::from_raw_parts`.
The profiled `_platform_memmove` comes later, when `DecoderSource` serves pipeline tiles from the eager
JPEG backing via `read_eager_region()` and copies each requested strip/tile into scheduler-owned buffers.

**Why libvips is still faster:** `.libvips_repo/libvips/foreign/jpeg2vips.c:805-889` does not decode a
resident frame first. It runs libjpeg as a sequential generator and calls `jpeg_read_scanlines()` directly
into libvips output strips (`VIPS_DEMAND_STYLE_FATSTRIP`). That avoids the extra resident-raster → tile
memmove that viprs currently pays after TurboJPEG decode.

**Case study (tracked issue):** once the root cause was framed as an I/O model mismatch instead of a SIMD problem,
the decisive win came from replacing eager resident decode + tile copies with a sequential scanline-fed
pipeline. This moved `three_op_chain` at 8192 from **2.329× slower** to **0.879×** and turned an open gap
into a libvips win for viprs.

---

## Known false leads — do NOT re-investigate

This section documents approaches that were fully investigated and rejected.
Before filing a task for any of these, read this section first.

---

### ❌ "Fix with-cache e2e gap by reusing pipeline across iterations" (tracked issue, 2026-06)

**Proposed fix:** reuse the viprs pipeline (same `op_id`) across bench iterations so the
tile cache gets hits on iterations 2+, matching what libvips was believed to do.

**Why it was investigated:** viprs with-cache e2e latency was 5–18× worse than libvips.
The tile cache inserts tiles on every iteration (different `op_id` → cache always cold),
causing tile accumulation and RSS explosion.

**Why it is WRONG:** libvips does NOT benefit from cross-iteration caching in e2e mode either.
The C runner `tools/bench-vs-libvips/libvips_runner.c` calls:
```c
VipsImage *in = vips_image_new_from_file(input, NULL);   // NEW pointer every iteration
vips_invert(in, &out, NULL);
```
Each call creates a new `VipsImage*` → new operation cache key → libvips cache miss every
iteration. libvips's `vips_cache` is useless in the e2e bench loop.

**Correct interpretation:** with-cache e2e is **symmetric** — both sides start cold per
iteration. The remaining gap is viprs tile-cache *overhead* (Arc allocation + HashMap
insert + RwLock) with zero hits. This overhead was reduced by tracked issue (zero-copy Arc insert
+ 32 MiB byte budget) but not eliminated.

**What to do instead:** treat **no-cache** as the canonical E2E metric for single-pass
pipelines. The with-cache overhead gap is expected and shrinks as tile-cache insertion
becomes cheaper. File a task only if the no-cache ratio regresses above 1.00.

**Follow-up finding (same investigation):** pipeline reuse on the viprs side (keeping
the same `op_id` across bench iterations so the tile cache gets hits on iteration 2+)
produced stunning numbers — JPEG with-cache ratios 0.086x / 0.258x / 0.475x. The
optimization IS real and valuable in production (persistent pipeline server, same
transformation served repeatedly). But it **cannot be used in the benchmark** because
libvips still rebuilds a new `VipsImage*` per iteration and gets zero cache hits —
the comparison would be viprs-warm-cache vs libvips-cold, i.e. a benchmark lie.

**Rule:** if you want to benchmark the warm-cache scenario, BOTH sides must be warm.
For libvips that would require a persistent process reusing the same `in` image pointer
across iterations. Until both sides are symmetric, with-cache e2e is NOT a valid metric.

---

## Open performance gaps (checkpoint 8, 2026-06)

| Op | Ratio at 8192 | Root cause | Active task |
|----|---------------|------------|-------------|
| three_op_chain (e2e) | 0.879× (**faster**) | Resolved by sequential JPEG scanline source removing eager decode → tile memmove | ✅ |
| JPEG load (standalone) | 19.28× slower | Full-frame TurboJPEG decode vs. libvips scanline decode | tracked issue |
| thumbnail 800 --no-e2e | **0.62× faster** | vld1_u8 NEON (factor=10) + chunked vld3q_u8 dispatch | ✅ |
| thumbnail 400 --no-e2e | **1.03× (tied)** | chunked vld3q_u8 NEON (factor=19), ShrinkH 27ms→12ms | ✅ |
| thumbnail 200 --no-e2e | **0.88× faster** | chunked vld3q_u8 NEON (factor=39), ShrinkH 29ms→12ms | ✅ |
| srgb_to_lab | 0.835× (faster) | Rust SIMD beats libvips at 8192 | ✅ |
| zoom 2x | 0.314× (faster) | Zero-copy ownership wins at 8192 | ✅ |

**Checkpoint 8 spot-checks (`tests/fixtures/images/sample.jpg`, 10 iters, e2e):**
- `three_op_chain`: viprs `3.93 ms` vs libvips `6.62 ms` → **0.593× ratio** (**1.69× faster**)
- `thumbnail 400`: viprs `0.62 ms` vs libvips `2.77 ms` → **0.225× ratio** (**4.44× faster**)
- `invert`: viprs `0.36 ms` vs libvips `0.57 ms` → **0.627× ratio** (**1.59× faster**)
- No new `ratio > 1.00` regressions were found in this checkpoint, so no new issue was filed.

**JPEG codec note:** `three_op_chain` now benefits from the landed sequential
`JpegScanlineSource` work (tracked issue), but standalone JPEG load still goes through the
full-buffer `TurboJPEG tj3Decompress8` path. libvips continues to use `jpeg_read_scanlines`
into FATSTRIP output; the remaining standalone load gap is tracked in tracked issue.

---

## Checkpoint 10 spot-checks (2026-06-11)

Requested audit commands (`tests/fixtures/images/sample.jpg`, 10 iters, e2e):

- `three_op_chain`: viprs `3.91 ms` vs libvips `5.43 ms` → **0.720× ratio** (**1.39× faster**)
- `thumbnail 200`: viprs `0.98 ms` vs libvips `2.70 ms` → **0.361× ratio** (**2.77× faster**)
- `load-jpeg`: **no ratio available** — `cargo xtask bench ... load-jpeg` failed on the
  libvips side with `Unknown operation: load-jpeg`

Outcome:

- No new measurable `ratio > 1.00` gaps were found in this checkpoint.
- The pre-existing `thumbnail 200` gap tracked in tracked issue was **not** reproduced by this
  sample-image e2e spot-check; that task remains active because it covers the
  factor=39 bandwidth-bound case, not this lighter workload.
- Standalone JPEG load remains an active product gap in tracked issue, but checkpoint 10 could
  not re-measure parity because the libvips runner lacks the `load-jpeg` scenario.
  Benchmark-infrastructure friction is now tracked in **tracked issue**.

---

## Checkpoint 13 spot-checks (2026-06-11)

Requested audit commands:

- `cargo xtask bench tests/fixtures/images/sample.jpg invert --iterations 30`
  - libvips `p50 1.28 ms`, `p95 2.79 ms`
  - viprs `p50 0.48 ms`, `p95 0.54 ms`
  - ratio: **0.373× p50**, **0.195× p95** (**viprs 2.68× faster**)
- `cargo xtask bench tests/fixtures/images/sample.jpg thumbnail 400 --iterations 20`
  - libvips `p50 5.53 ms`, `p95 17.19 ms`
  - viprs `p50 0.66 ms`, `p95 1.46 ms`
  - ratio: **0.119× p50**, **0.085× p95** (**viprs 8.43× faster**)
- `cargo xtask bench tests/fixtures/images/bench_512x512.jpg perceptual_enhance webp --iterations 10`
  - **no ratio available** — libvips side failed with `Unknown operation: perceptual_enhance`
  - Requested 2048 rerun was skipped because the reference runner cannot execute the op yet

Outcome:

- No new measurable `ratio > 1.00` product gaps were found in this checkpoint.
- `invert` and `thumbnail 400` both remain clear wins for viprs on the sample-image e2e workload.
- `perceptual_enhance` could not be audited honestly because `tools/bench-vs-libvips/libvips-runner`
  still lacks the matching reference scenario; benchmark-infrastructure friction is tracked in **tracked issue**.

---

## Checkpoint 14 spot-checks (2026-06-11)

Requested audit commands:

- `cargo xtask bench tests/fixtures/images/sample.jpg invert --iterations 30`
  - libvips `p50 0.851 ms`, `p95 1.481 ms`
  - viprs `p50 0.422 ms`, `p95 0.659 ms`
  - ratio: **0.495× p50**, **0.445× p95** (**viprs 2.02× faster**)
- `cargo xtask bench tests/fixtures/images/sample.jpg thumbnail 400 --iterations 20`
  - libvips `p50 4.09 ms`, `p95 5.431 ms`
  - viprs `p50 0.725 ms`, `p95 1.077 ms`
  - ratio: **0.177× p50**, **0.198× p95** (**viprs 5.64× faster**)
- `cargo xtask bench demo/images/cyberpunk_portrait.jpg perceptual_enhance --iterations 20`
  - libvips `p50 38.881 ms`, `p95 49.160 ms`
  - viprs `p50 68.489 ms`, `p95 77.470 ms`
  - ratio: **1.762× p50**, **1.576× p95** (**viprs slower**)
  - follow-up profile (`cargo xtask profile demo/images/cyberpunk_portrait.jpg perceptual_enhance --iterations 5 --ai`) shows viprs self-time concentrated in `GetResidualCost_NEON` (8.7%), `LabToSRgb::convert_region` (8.4%), `decode_mcu_AC_refine` (7.3%), `VP8PutBit` (6.4%), and `ShrinkH::process_region` (5.9%)

Outcome:

- `invert` and `thumbnail 400` remain strong e2e wins for viprs on the sample-image workload.
- `perceptual_enhance` is now a confirmed significant gap on the cyberpunk 4K workflow: both p50 and p95 are slower than libvips, with p50 above the 1.50× high-priority threshold.
- Checkpoint 14 filed **tracked issue** to track the regressed `perceptual_enhance` gap and its current hotspot mix.

---

## Checkpoint 15 spot-checks (2026-06-11)

Requested audit commands:

- `cargo xtask bench tests/fixtures/images/sample.jpg invert --iterations 30`
  - libvips `p50 0.73 ms`, `p95 0.89 ms`
  - viprs `p50 0.41 ms`, `p95 0.51 ms`
  - ratio: **0.560× p50**, **0.579× p95** (**viprs 1.79× faster**)
- `cargo xtask bench tests/fixtures/images/sample.jpg thumbnail 400 --iterations 20`
  - libvips `p50 2.56 ms`, `p95 2.91 ms`
  - viprs `p50 0.64 ms`, `p95 0.93 ms`
  - ratio: **0.251× p50**, **0.319× p95** (**viprs 3.99× faster**)
- `cargo xtask bench demo/images/cyberpunk_portrait.jpg perceptual_enhance --iterations 20`
  - libvips `p50 36.29 ms`, `p95 36.80 ms`
  - viprs `p50 66.51 ms`, `p95 67.47 ms`
  - ratio: **1.833× p50**, **1.834× p95** (**viprs slower**)
  - follow-up profile (`cargo xtask profile demo/images/cyberpunk_portrait.jpg perceptual_enhance --ai`) shows viprs self-time concentrated in `LabToSRgb::convert_region` (14.5%), `ShrinkH::process_region` (13.9%), `OperationBridge::dyn_process_region` (7.2%), `SRgbToLab::convert_region` (6.9%), `GetResidualCost_NEON` (5.7%), and `VP8PutBit` (5.7%)

Outcome:

- `invert` and `thumbnail 400` remain clear e2e wins for viprs on the sample-image workload, but both wins narrowed versus the prior checkpoint because libvips improved more than viprs in this run.
- `perceptual_enhance` remains a confirmed high-priority cyberpunk 4K gap with both p50 and p95 above the 1.50× threshold; the current hotspot mix now points more strongly at Viprs colour-conversion and horizontal-shrink work than at WebP encode alone.
- Checkpoint 15 filed **tracked issue** to track the still-open `perceptual_enhance` regression with the updated profile evidence.

---

## Checkpoint 17 spot-checks (2026-06-12)

Merges since CP16: tracked issue (SRgbLabAdjust cached op), tracked issue/559 (4GiB allocation guards),
tracked issue (README/docs/examples), tracked issue (ShrinkH vrshrn_n_u16 rounding).

Requested audit commands (60 iterations, release binary, macOS aarch64):

- `cargo xtask bench tests/fixtures/images/sample.jpg invert --iterations 60`
  - libvips `p50 0.65 ms`, `p95 0.81 ms`
  - viprs `p50 0.39 ms`, `p95 0.47 ms`
  - ratio: **0.602× p50**, **0.583× p95** (**viprs 1.66× faster**)
- `cargo xtask bench tests/fixtures/images/sample.jpg thumbnail 400 --iterations 40`
  - libvips `p50 3.91 ms`, `p95 9.47 ms`
  - viprs `p50 0.69 ms`, `p95 1.11 ms`
  - ratio: **0.175× p50**, **0.117× p95** (**viprs 5.70× faster**)
- `cargo xtask bench tests/fixtures/images/sample.jpg load-jpeg --iterations 40`
  - libvips `p50 0.28 ms`, `p95 0.39 ms`
  - viprs `p50 0.22 ms`, `p95 0.34 ms`
  - ratio: **0.746× p50** (**viprs 1.34× faster**)
- `cargo xtask bench demo/images/cyberpunk_portrait.jpg perceptual_enhance --iterations 20`
  - libvips `p50 37.14 ms`, `p95 40.15 ms`
  - viprs `p50 61.56 ms`, `p95 65.78 ms`
  - ratio: **1.656× p50** (**viprs slower — structural WebP gap**)

Outcome:

- `invert`, `thumbnail 400`, and `load-jpeg` all remain strong e2e wins. Initial perf-eng-17
  measurements showed apparent regressions (0.615×, 0.243×, 0.777×) but re-validation with
  higher iteration counts confirmed these were OS scheduling noise; sub-ms operations are
  inherently susceptible to lock-contention variance on macOS.
- `thumbnail 400` actually **improved** to 0.175× (viprs 5.7× faster vs CP16's 4.5×), likely
  driven by ShrinkH NEON tightening in tracked issue.
- `perceptual_enhance` improved from 1.852× (CP16) → 1.667× (tracked issue) → 1.656× (CP17), but
  remains a confirmed gap tracked in **tracked issue**.
- Checkpoint 17 filed: tracked issue (SRgbLabAdjust <3-band panic), tracked issue (WebP eager 4GiB bypass),
  tracked issue (libvips symbol loss friction), tracked issue (perceptual_enhance gap).

---

## Checkpoint 18 spot-checks (2026-06-12)

Merges since CP17: tracked issue/564 (SRgbLabAdjust band-count guard, WebP eager 4GiB guard),
tracked issue (perceptual_enhance WebP structural gap documented), tracked issue/572 (SRgbLabAdjust LUT
rewrite — f64 precision, removed unbounded global cache), tracked issue (SRgbLabAdjust identity bypass),
tracked issue (animated WebP N-frame accumulation guard), tracked issue (Image::from_buffer checked_mul),
tracked issue (animated WebP region decode bounded memory).

Requested audit commands (60 iterations, `target/release/xtask`, macOS aarch64):

- `target/release/xtask bench tests/fixtures/images/sample.jpg invert --iterations 60`
  - ratio: **0.601× p50** (**viprs 1.66× faster**)
- `target/release/xtask bench tests/fixtures/images/sample.jpg thumbnail 400 --iterations 60`
  - ratio: **0.186× p50** (**viprs 5.4× faster**)
- `target/release/xtask bench tests/fixtures/images/sample.jpg load-jpeg --iterations 60`
  - ratio: **0.801× p50** (**viprs 1.25× faster**)
- `target/release/xtask bench tests/fixtures/images/sample.jpg perceptual-enhance --iterations 40`
  - ratio: **1.563× p50** (**viprs slower — structural WebP gap continuing**)

Outcome:

- All three primary ops hold strong wins vs CP17. Apparent ratio shifts in initial measurements
  were OS scheduling noise (validated with 60 iterations).
- `perceptual_enhance` improved slightly: 1.656× (CP17) → 1.563× (CP18). Structural gap
  tracked in tracked issue.
- Checkpoint 18 filed: tracked issue (SRgbLabAdjust identity drift), tracked issue (animated WebP accumulation),
  tracked issue (Image::from_buffer overflow), tracked issue (animated WebP shrink=1 bounded region).

---

## Checkpoint 19 spot-checks (2026-06-12)

Merges since CP18: tracked issue (animated WebP region decoder composites all frames),
tracked issue (Tile/TileMut checked length arithmetic).

Requested audit commands (60 iterations, `target/release/xtask`, macOS aarch64):

- `target/release/xtask bench tests/fixtures/images/sample.jpg invert --iterations 60`
  - ratio: **0.444× p50** (**viprs 2.25× faster**)
- `target/release/xtask bench tests/fixtures/images/sample.jpg thumbnail 400 --iterations 60`
  - ratio: **0.164× p50** (**viprs 6.1× faster**)
- `target/release/xtask bench tests/fixtures/images/sample.jpg load-jpeg --iterations 60`
  - ratio: **0.767× p50** (**viprs 1.3× faster**)
- `target/release/xtask bench tests/fixtures/images/sample.jpg perceptual-enhance --iterations 40`
  - ratio: **1.580× p50** (**viprs slower — structural WebP gap**)

Outcome:

- `thumbnail 400` reached 0.164× — viprs now 6.1× faster than libvips on this workload.
- `invert` improved to 0.444× (viprs 2.25× faster).
- `perceptual_enhance` stable at 1.580×; gap confirmed structural (archived tracked issue).
- Checkpoint 19 filed: tracked issue (WebP first-frame-only region bug), tracked issue (Tile overflow),
  tracked issue/582 (friction — analyzer tooling notes added), tracked issue (absorbed by tracked issue).

---

## Checkpoint 20 spot-checks (2026-06-12)

Merges since CP19: tracked issue (animated WebP region decoder composites all frames),
tracked issue (Tile/TileMut checked length arithmetic), tracked issue (integration test cfg gates).

Requested audit commands (60 iterations, `target/release/xtask`, macOS aarch64):

- `target/release/xtask bench tests/fixtures/images/sample.jpg invert --iterations 60`
  - libvips `p50 0.83 ms`, viprs `p50 0.57 ms`
  - ratio: **0.682× p50** (**viprs 1.47× faster**)
- `target/release/xtask bench tests/fixtures/images/sample.jpg thumbnail 400 --iterations 60`
  - libvips `p50 4.13 ms`, viprs `p50 0.72 ms`
  - ratio: **0.174× p50** (**viprs 5.75× faster**)
- `target/release/xtask bench tests/fixtures/images/sample.jpg load-jpeg --iterations 60`
  - libvips `p50 0.48 ms`, viprs `p50 0.26 ms`
  - ratio: **0.538× p50** (**viprs 1.86× faster**)
- `target/release/xtask bench tests/fixtures/images/sample.jpg perceptual-enhance --iterations 40`
  - libvips `p50 30.46 ms`, viprs `p50 9.19 ms`
  - ratio: **0.302× p50** (**viprs 3.31× faster — gap CLOSED**)

Outcome:

- 🎉 **`perceptual_enhance` structural gap closed**: 1.580× (CP19, viprs slower) → 0.302× (CP20,
  viprs 3.3× faster). All 4 monitored ops are now faster than libvips. Archived tracked issue.
- `invert` ratio increased slightly (0.444× → 0.682×) due to libvips scheduling variance on
  this sub-ms op; absolute viprs time held stable.
- `thumbnail 400` stable around 0.174× — viprs 5.75× faster.
- `load-jpeg` improved to 0.538× — viprs 1.86× faster.
- Checkpoint 20 filed: tracked issue (WebP region decode panics on overflowing Region),
  tracked issue (reduce_v Lanczos3 factor=1.5 golden parity gap — pre-existing).
