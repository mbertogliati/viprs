# AGENTS.md — Viprs agent instructions

Viprs is a **native Rust reimplementation of libvips**: a demand-driven, horizontally-threaded
image processing library. Performance is the primary constraint — every architectural decision
must be justified against its runtime cost.

Read `.github/agents/GUIDELINES.md` before touching any code.
For specialized agent workflows, see `.github/agents/`.

## Toolchain

This repo pins Rust `1.96.0` via `rust-toolchain.toml`.

```bash
rustup default 1.96.0
cargo check -p xtask
```

---

## Reference implementation

The libvips source is available locally in `.libvips_repo/`.
**Consult it before implementing any operation** to ensure algorithms match the reference:
- `.libvips_repo/libvips/arithmetic/` — arithmetic ops
- `.libvips_repo/libvips/colour/` — colorspace conversions
- `.libvips_repo/libvips/convolution/` — convolution and morphology
- `.libvips_repo/libvips/resample/` — resize, affine, thumbnail
- `.libvips_repo/libvips/foreign/` — codecs (tiff, gif, heif, avif)
- `.libvips_repo/libvips/iofuncs/` — pipeline, tile cache
- `.libvips_repo/libvips/create/` — generator sources

Do not implement from memory or external sources when the correct behaviour is defined
by libvips. The goal is pixel-exact compatibility where possible.

---

## Non-negotiable rules

### 1. No `dyn Trait` on hot paths
Monomorphization is mandatory wherever the concrete type is known at compile time.
`dyn Trait` is only acceptable for plugin/codec registries that must be runtime-extensible
(e.g., `foreign` format adapters stored in a `Vec<Box<dyn ImageCodec>>`).
If you add `dyn` anywhere else, explain in a comment why static dispatch is impossible.

### 2. Zero heap allocations in pixel-path code
Operations inside `domain/ops/` must not allocate on the heap per-pixel or per-tile.
Pre-allocate buffers at pipeline construction time. Use `&mut [T]` slices, not `Vec`.

### 3. Infrastructure traits stay in `ports/`; domain traits stay in `domain/`
Traits that abstract over external infrastructure (codecs, schedulers, I/O sources/sinks)
live under `src/ports/`. Traits that define central domain behaviour — `Op`, `DynOperation`,
`ColourConvert`, `TileReducer`, `ResampleOp` — live under `src/domain/`.
Domain ops and reducers (`src/domain/ops/`, `src/domain/reducers/`) also live in `domain/`.
Concrete infrastructure implementations live under `src/adapters/` (codecs, scheduler, pipeline, sources, sinks).
Domain types and domain traits (`src/domain/`) import nothing from `ports/` or `adapters/`.

### 4. Errors must be typed
Never use `Box<dyn Error>` in library-facing APIs. Define concrete error enums per module
in `src/domain/error.rs`. Use `thiserror` for derive macros.

### 5. `unsafe` requires a safety comment
Every `unsafe` block must be preceded by a `// SAFETY:` comment explaining the invariant
that makes it sound. No exceptions.

### 6. No `unwrap` / `expect` outside tests
All fallible paths in library code return `Result<T, ViprsError>`. `unwrap()`/`expect()`
are banned outside of `#[cfg(test)]` blocks and `fn main()` in example binaries.
CI enforces the repository-side ban through the `Cargo.toml` `[lints.clippy]` table and
`.github/workflows/lint-no-unwrap.yml`, scoped to production lib/bin/example targets so
test-only `unwrap()`/`expect()` remains allowed.
`#[allow(clippy::unwrap_used)]` / `#[allow(clippy::expect_used)]` are not an approved escape
hatch; if any lint suppression is truly necessary, it must carry a nearby `// REASON:` comment
and must not be used to bypass this repository rule or CI.

### 7. PRs must include a Resolution summary
Before merging any PR, include a structured summary of what was done and why.
This prevents false confidence: a merged PR with no evidence of correctness is
worse than an open one.

---

## Architectural map

```
src/
├── domain/          ← pure Rust, zero external deps
│   ├── image.rs     ← Image<F: BandFormat>, Region, Tile
│   ├── format.rs    ← BandFormat trait + concrete types (U8, F32, …)
│   ├── error.rs     ← ViprsError and sub-error enums
│   ├── op.rs        ← Op, DynOperation, ViewOp, DynViewOp, OperationBridge, NodeSpec
│   ├── colour.rs    ← ColourConvert — colorspace conversion interface
│   ├── reducer.rs   ← TileReducer — aggregate/scalar reduction interface
│   ├── resample.rs  ← ResampleOp, ReduceConfig, FilterOrientation
│   ├── ops/         ← arithmetic/, colour/, convolution/, … (mirrors libvips modules)
│   └── reducers/    ← histogram.rs, stats.rs
├── ports/           ← infrastructure traits only (I/O, external systems)
│   ├── codec.rs     ← ImageDecoder, ImageEncoder
│   ├── scheduler.rs ← TileScheduler, ReducingScheduler
│   ├── sink.rs      ← ImageSink, ConcurrentSink
│   └── source.rs    ← ImageSource, DynImageSource
├── adapters/        ← concrete impls, may have external deps
│   ├── codecs/      ← jpeg.rs, png.rs, webp.rs, …
│   └── scheduler/   ← thread_pool.rs, rayon.rs
└── lib.rs           ← public re-exports only
```

---

## Performance checklist (run before every PR)

- [ ] `cargo bench` shows no regression vs. `main` on the affected operation benchmarks.
      A regression is any throughput drop > 5% on any of the three standard sizes (512/2048/8192).
- [ ] `cargo xtask bench` viprs/libvips ratio ≤ 1.00 for all affected ops. Any ratio > 1.00 is a
      performance gap that must be filed as an issue before merging. The target is for viprs to be
      faster than libvips, exploiting Rust's SIMD-without-FFI, zero-copy ownership, and
      monomorphized dispatch.
- [ ] Every new op has a benchmark file in `benches/<module>/` registered in `Cargo.toml`.
- [ ] `cargo flamegraph` (or `cargo instruments` on macOS) shows no unexpected allocator
      calls in the pixel path.
- [ ] `cargo clippy -- -D clippy::perf` passes with zero warnings.
- [ ] No new `dyn Trait` on hot paths (grep: `dyn ` in `src/domain/`).

---

## Test strategy

| Layer | Tool | Scope |
|---|---|---|
| Domain types | `#[test]` | unit, no I/O |
| Operations | `proptest` | property-based, pixel math |
| Codecs | `#[test]` with fixture files | round-trip encode/decode |
| Pipeline | `#[test]` | integration, compare output hashes |
| Benchmarks | `criterion` | in `benches/` |

Do NOT use `mockall` or any mocking framework. Test with real data or simple in-memory
images constructed via `Image::from_buffer`.

**Minimum coverage:** line coverage ≥ 90% on all code in `src/domain/ops/` and `src/adapters/codecs/`.

```bash
cargo llvm-cov --lib --ignore-filename-regex '(benches|tests)' 2>&1 | tail -5
```

For feature-gated modules, run the matching feature-enabled alias as part of the same audit.
FFT coverage is enforced with:

```bash
cargo cov-lib-fft 2>&1 | grep -E 'freqfilt|fwfft|invfft'
```

A PR that drops coverage below 90% in any `ops/` or `codecs/` module cannot be merged.

Complementary rules:
- Each `process_region` must have at least one proptest identity test and one boundary-value test.
- Each codec must have a round-trip encode→decode test that verifies buffer integrity.
- Tests that pass for incorrect reasons count as debt, not coverage.

---

## Code style

- `rustfmt` with project defaults (`cargo fmt`) — non-negotiable before commit.
- `clippy::pedantic` is enabled. Fix warnings; don't `#[allow]` without a comment.
- **All code, comments, and documentation must be in English.** No exceptions.
- Every public item (`pub fn`, `pub struct`, `pub trait`, `pub enum`) must have a `///` doc
  comment explaining what problem it solves and a short usage example.
- Comments only when the WHY is non-obvious. No paraphrasing of what the code does.
- Prefer named struct fields over positional tuples for anything with more than 2 fields.
- `const` over `static` for compile-time values; `static` only for true global state.

---

## What NOT to do

- Do not wrap the C libvips via FFI — this is a **native** reimplementation.
- Do not add `async` to the pixel-processing path. Concurrency is thread-based (rayon).
- Do not introduce a dependency without checking it compiles to minimal binary size
  (`cargo bloat --release`).
- Do not derive `Clone` or `Debug` on types that hold large pixel buffers without
  implementing them manually (auto-derive copies the buffer in `Clone` with no warning).
- Do not use `Arc<Mutex<T>>` for pixel data. Use ownership transfer or `Arc<RwLock<T>>`
  only when reads dominate; document the concurrency model in a comment.

---

## Benchmark framework: viprs vs libvips

Location: `tools/bench-vs-libvips/` (C runner) + `xtask/` (orchestrator) + `docker/` (profiling container)

Performance investigation methodology and case studies: **[.github/agents/PERFORMANCE.md](.github/agents/PERFORMANCE.md)**

### Two commands

| Command | Purpose | Docker? |
|---|---|---|
| `cargo xtask bench` | Latency E2E + RSS + page faults + context switches | No |
| `cargo xtask perf` | Hard metrics: SIMD ratio, allocations, cache misses | Partial |

### Quick reference

```bash
# ── Latency comparison (local, no Docker) ──
cargo xtask bench <input> <op> [args] --iterations N

# ── Hard metrics (local for simd/alloc, Docker for hw) ──
cargo xtask perf <input> <op> [args] --metrics simd|alloc|hw|all [--arch arm64|amd64]

# Examples:
cargo xtask bench tests/fixtures/images/sample.jpg invert --iterations 50
cargo xtask perf tests/fixtures/images/sample.jpg invert --metrics alloc
cargo xtask perf tests/fixtures/images/sample.jpg thumbnail 400 --metrics simd
```

### Metrics collected

| Category | Metric | Source | Docker? |
|---|---|---|---|
| Latency | Wall-clock p50/p95 | `Instant` / `clock_gettime` | No |
| Resources | Peak RSS, page faults, ctx switches | `getrusage` | No |
| Allocations | count, bytes, peak live, call stacks | `dhat` crate (primary), counting allocator (fallback) | No |
| SIMD | Instruction ratio (NEON/AVX vs scalar) | `objdump` disassembly | No |
| Cache | L1/LL misses, branch mispredictions | cachegrind (deterministic) | Yes |
| HW counters | Real IPC, cycles, TLB misses | `perf stat` (real PMU) | Yes (bare-metal Linux) |

### Docker setup (one-time)

```bash
colima start --arch aarch64 --cpu 4 --memory 4
docker buildx build --platform linux/arm64 -t viprs-perf:arm64 -f docker/Dockerfile .
cargo xtask perf tests/fixtures/images/sample.jpg invert --metrics hw
```

### Adding a new benchmark scenario

~10 lines per side. See `xtask/README.md` for step-by-step guide.
