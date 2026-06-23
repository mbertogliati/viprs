# Viprs Engineering Guidelines

**Viprs** is a native Rust reimplementation of [libvips](https://github.com/libvips/libvips):
a demand-driven, horizontally-threaded image processing library.

> Performance is the primary architectural constraint. Every abstraction layer
> must justify its existence against its runtime cost.

---

## Table of Contents

1. [Architecture boundaries](#1-architecture-boundaries)
2. [TDD workflow](#2-tdd-workflow)
3. [Type design](#3-type-design)
4. [Trait design (ports)](#4-trait-design-ports)
5. [Generics and monomorphization](#5-generics-and-monomorphization)
6. [Performance model and rules](#6-performance-model-and-rules)
7. [SOLID applied to Rust](#7-solid-applied-to-rust)
8. [Error handling](#8-error-handling)
9. [Concurrency model](#9-concurrency-model)
10. [Module structure (libvips parity)](#10-module-structure-libvips-parity)
11. [Dependency policy](#11-dependency-policy)
12. [Issue filing obligation](#issue-filing-obligation)

**Related:** [CI_GUIDELINES.md](CI_GUIDELINES.md) — GitHub Actions workflow structure for PRs.

---

## 1. Architecture boundaries

The project is organized as a hexagon. The dependency rule is strict:

```
adapters/ → ports/ ← domain/
             ↑
         (infrastructure ports only)
```

### Qué va en cada capa

- **`domain/`** — tipos puros Y interfaces de dominio. No importa nada de `ports/` ni
  `adapters/`. Contiene:
  - Tipos: `Image`, `Region`, `Tile`, `BandFormat`, `Colorspace`, `ViprsError`
  - Interfaces de dominio: `Op`, `ViewOp`, `DynOperation`, `ColourConvert`,
    `TileReducer`, `ResampleOp` — traits que definen comportamiento central del
    dominio, sin dependencias externas.

- **`ports/`** — solo traits de infraestructura: abstracciones sobre el mundo exterior.
  Contiene: `ImageDecoder`, `ImageEncoder`, `TileScheduler`, `ImageSource`, `ImageSink`.
  Regla: si el trait abstrae sobre una biblioteca externa, concurrencia, I/O, o hardware
  → va en `ports/`. Si es comportamiento puro del dominio → va en `domain/`.

- **`adapters/`** — implementaciones concretas. Importa `domain/` y `ports/`;
  nunca un adapter importa a otro adapter directamente.

**Regla**: si necesitás que dos adapters colaboren, extraé un trait en `ports/` y
dependé de eso. Nunca cross-import entre adapters.

**Regla**: `lib.rs` solo re-exporta. No hay lógica ahí.

### Módulos: sintaxis nueva (Rust 2018+)

Usar la sintaxis de módulo inline en lugar de `mod.rs`:

```
// CORRECTO — foo.rs al lado del directorio foo/
src/adapters.rs        ← en lugar de src/adapters/mod.rs
src/domain/ops.rs      ← en lugar de src/domain/ops/mod.rs

// INCORRECTO — evitar
src/adapters/mod.rs
src/domain/ops/mod.rs
```

Nunca crear archivos `mod.rs` nuevos. Si ves un `mod.rs` existente y lo tocás,
migrarlo a la sintaxis nueva en el mismo commit.

---

## 2. TDD workflow

TDD is the default development mode. The cycle is: Red → Green → Refactor, repeated
per behavior, not per file.

### Write the test first

Before writing any implementation:

1. Write a `#[test]` that expresses the intended behavior in terms of the public contract.
2. Confirm it fails to compile or fails at runtime (Red).
3. Write the minimum code to make it pass (Green).
4. Refactor without breaking the test (Refactor).

This order is non-negotiable for all new operations, codecs, and domain types.

### Unit tests live with the code

Every `.rs` file that contains logic must end with a `#[cfg(test)] mod tests` block.
Tests in external `tests/` files are for integration scenarios only.

### Property-based tests for pixel math

For any function that transforms pixel values, prefer `proptest` over hand-picked
examples. Properties to encode:

- **Identity**: applying a no-op operation leaves pixels unchanged.
- **Inverse**: applying an operation and its inverse returns the original.
- **Commutativity / associativity**: where the operation contract guarantees it.
- **Boundary**: zero-sized regions, single-pixel images, maximum sample values.

Write the property test before the implementation so the contract is clear before any
code exists.

### Golden-output tests as acceptance criteria

For each libvips-parity operation, capture the expected output from libvips and store it
in `tests/fixtures/<op>/<case>.bin` before writing the Rust implementation.
The golden test is the acceptance criterion; the operation is complete only when it passes.

### Benchmark as a test

Benchmarks in `benches/<module>/<op>.rs` are not optional. Write the benchmark shell
before implementing the operation — this forces you to define the public interface before
the internals, and prevents interfaces that are hard to benchmark (a proxy for interfaces
that are hard to use).

### Test the contract, not the implementation

Tests must not import private modules, inspect private fields, or rely on internal
struct layout. If a test requires accessing internals, the interface is wrong — redesign
the public API so the behavior is observable from outside.

---

## 3. Type design

### Wrap domain primitives in newtypes

Never pass bare `u32` for width, height, or band count. Wrap each distinct concept in a
newtype. This prevents argument transposition bugs (a historically common libvips error)
and makes function signatures self-documenting without comments.

Implement `From`/`Into` conversions for newtypes only when the conversion is lossless
and unambiguous in all call sites.

The `Region` / `Tile` / `TileMut` / `DemandHint` core types and the reasoning behind
their exact representation (signed coordinates, no stride, band count in tile, interleaved
layout).

### Seal the format trait

The `BandFormat` trait must be sealed: implementors outside this crate must not be
allowed. Use the standard Rust sealing pattern (a private super-trait in a `private`
module). This ensures the compiler can enumerate all concrete formats and enables
exhaustive `match` in adapters.

### Phantom types for format parameterization

When a struct is generic over a format but does not store samples directly, use
`PhantomData<F>` to carry the type parameter. This costs nothing at runtime and lets the
compiler enforce format compatibility at compile time.

### Prefer `&[T]` slices over `Vec<T>` in function signatures

Slices borrow without implying ownership. Operations receive pixel data as mutable slice
references; they do not own buffers. This rule applies to all types on the pixel path.

### Builder pattern for operations with optional parameters

Any operation with more than two optional parameters must expose a builder. The builder
validates all parameters in `build()` and returns `Result<Op, ViprsError>`. The built
struct is plain data with no heap allocation beyond what the parameters themselves require.
Write the builder test (including the error cases for invalid parameters) before
implementing `build()`.

### Avoid interior mutability in operation structs

Operation structs must be `Send + Sync`. Mutable state during pixel processing lives
exclusively in the `Tile` output buffer passed as a parameter. `Cell`, `RefCell`, and
`Mutex` inside an operation struct are prohibited.

---

## 4. Trait design (ports)

### Traits are narrow

Each port trait expresses exactly one capability. If you find yourself adding a method
that belongs to a different concern, split the trait. A struct can implement multiple
narrow traits; a wide trait cannot be split later without breaking callers.

### Associated types over generic parameters where the type is determined by the implementor

If an implementor fixes its input or output format, use an associated type. If the same
implementor can handle multiple formats, use a generic parameter. Never default to one
form without considering this distinction.

### `Send + Sync` bounds on all port traits

Every port trait must require `Send + Sync`. The scheduler dispatches work across
threads; an operation that cannot cross thread boundaries breaks the pipeline.

### Exclude non-object-safe methods from the vtable with `where Self: Sized`

When a trait needs both a static helper (e.g., `probe`) and object-safe methods, mark
the static helper `where Self: Sized`. This keeps the vtable clean for the cases where
`dyn Trait` is legitimately needed (codec registries).

### Write the trait before the first implementor

Define the trait signature, document its contract (preconditions, postconditions,
thread safety), and write a failing test that calls through the trait — all before any
struct implements it. The test forces the API to be callable from the outside before
the inside is designed.

---

## 5. Generics and monomorphization

### Default to static dispatch

Prefer `impl Trait` parameters and generic structs over `dyn Trait`. Static dispatch
enables inlining and eliminates the vtable indirection. The compiler generates one
specialized copy of the function per concrete type — this is monomorphization, and it
is the primary mechanism for achieving libvips-level throughput in Rust.

### Reserve `dyn Trait` for runtime-extensible registries

The only legitimate use of `dyn Trait` in hot paths is a codec or operation registry
that must accept types not known at compile time. Even then, the `dyn` boundary should
be at registration time, not at per-tile processing time.

The dispatch strategy — a static `Operation<F>` trait for monomorphized impls and a
`DynOperation` bridge for runtime pipelines — is documented in full in


### Apply trait bounds at the function level, not the struct level

Prefer:

```rust
impl<F: BandFormat> MyOp<F> {
    pub fn process<S: Scheduler>(&self, scheduler: &S) { … }
}
```

over:

```rust
struct MyOp<F: BandFormat, S: Scheduler> { … }
```

Embedding the scheduler as a type parameter in the struct locks downstream code into
providing both at construction time, limiting flexibility without a performance benefit.

### Use `#[inline]` with intent

- Mark the per-tile entry point of every `Operation` impl with `#[inline]`.
- Mark inner per-pixel loops with `#[inline(always)]` only after profiling shows
  the compiler is not inlining them automatically.
- Do not `#[inline]` functions that call the allocator or have significant branch
  trees — forced inlining at those sites bloats the caller.

---

## 6. Performance model and rules

These are constraints, not suggestions. A PR that violates any of these requires
an explicit performance justification and benchmark evidence.

### Why libvips is fast — the mechanisms viprs must replicate

libvips achieves its performance through a hierarchy of mechanisms. Understanding
the hierarchy matters because lower-level mechanisms enable the upper ones; violating
a lower rule silently breaks all the ones above it.

**Demand-driven evaluation is the foundation.**
A pipeline of operations allocates no intermediate buffers. When operations are chained,
each one stores a generate callback rather than computing pixels. Pixels are only computed
when a sink (disc writer, memory writer) pulls a specific rectangular region. The entire
pipeline runs inside a single tile of memory — small enough to fit in L2 cache. This is
why libvips can process multi-gigabyte images with tens of megabytes of RAM.

**Horizontal threading multiplies that foundation.**
libvips threads across tiles, not across pipeline stages. Multiple threads each run
a complete private copy of the pipeline simultaneously on different output tiles.
Because each thread has its own private mutable state (input regions, scratch buffers),
generate functions need no locks between each other. The result documented in the libvips
source: **four lock operations per output tile, regardless of pipeline length or
complexity.** This lock count does not scale with concurrency or pipeline depth — which
is why libvips scales almost linearly with CPU count.

**Per-thread buffer cache eliminates allocator pressure.**
Each thread holds a private pool of pixel buffers (`GPrivate` in libvips, thread-local
storage in viprs). Buffers are recycled between tiles without touching the global
allocator. The allocator is the bottleneck in multi-threaded code; removing it from the
hot path is what enables the lock-free throughput.

**Demand hints select the optimal tile geometry.**
Each operation declares the tile shape it works best with. The pipeline propagates the
most restrictive shape to the sink, which uses it to size tiles to fit L2 cache.
Scanline-only processing is the degenerate case — it prevents 2D cache locality and
serializes operations that could otherwise tile.

**Zero-copy region sharing eliminates memcpy on structural operations.**
Operations that reposition or combine images (extract, insert, flip, rotate by 90°)
implement their effect by adjusting pointer arithmetic inside the region, not by copying
pixel data. A region can be a view into another region with zero allocation.

**SIMD is the inner-loop multiplier.**
After demand-driven evaluation eliminates unnecessary work and threading parallelizes
the remaining work, SIMD multiplies the throughput of each individual pixel operation
by 3–4×. It is important but not foundational — SIMD on a pipeline that allocates
intermediates per-tile yields far less gain than SIMD on a pipeline that does not.

---

### P1 — Zero heap allocation on the pixel path

The pixel path is the call chain from `TileScheduler::run` to `Operation::process_region`.
No `Vec`, `Box`, or `Arc` construction is allowed inside this chain.
Buffers are allocated once at pipeline-build time, stored in thread-local pools, and
reused across all tiles. This is the Rust expression of libvips's per-thread buffer cache.

The mechanism that makes this possible — topological sort at build time, buffer index
assignment, and pre-sized `thread_local!` pools — is documented in


### P2 — Demand-driven evaluation is the contract of `required_input_region`

An operation computes only the pixels that a downstream consumer requests.
`required_input_region` must return the exact input rectangle needed — no over-fetch
beyond a necessary halo. The entire memory-efficiency guarantee of the system depends on
this: an operation that requests more input than it needs forces upstream operations to
compute pixels that are immediately discarded, burning L2 cache and CPU cycles.

### P3 — Tile geometry via demand hints

Every operation must declare its preferred tile shape by returning a `DemandStyle` value.
The pipeline scheduler selects the most restrictive shape across the chain and sizes tiles
to fit L2 cache. Valid shapes:

| Shape | When to use |
|---|---|
| `SmallTile` | Operations with a 2D neighborhood (convolution, morphology, rank filters) |
| `FatStrip` | Operations with a small row halo or that are row-local |
| `ThinStrip` | Purely pixel-local operations (arithmetic, colour conversion) |
| `Any` | Operations with no locality preference (statistics, reductions) |

Scanline-only processing of large images is prohibited. It prevents 2D cache reuse and
serializes threads onto a single row at a time.

### P4 — SIMD for throughput-critical inner loops

Operations in `arithmetic/` and `convolution/` must provide SIMD implementations.
Gate them with `#[cfg(target_feature = "...")]` and always provide a scalar fallback.
Write and test the scalar path first; add the SIMD path after and assert both produce
identical output on the same input (property test, not golden file).
Document the measured throughput ratio in the benchmark output.

### P5 — Zero-copy on structural operations

Operations that do not transform pixel values — extract, insert, crop, flip, transpose —
must implement their effect by adjusting region coordinates and returning a view into the
input buffer. A structural operation that copies pixel data is a correctness error, not
just a performance issue.

### P6 — Benchmark every new operation

A new operation is not mergeable without a `criterion` benchmark covering at minimum
three image sizes (512×512, 2048×2048, 8192×8192). Write the benchmark shell before the
implementation, for the same reason as writing the test first: it forces the public
interface to be defined before the internals, and surfaces APIs that are hard to call.

### P7 — No inter-iteration cache in benchmarks

Benchmarks run ONE mode only: direct comparison. Internal tile cache (viprs, 32 MiB) and
operation cache (libvips) are always ON within a single iteration. Each iteration builds
a fresh pipeline — no cache is ever warm or shared across iterations. There is no
"with-cache" vs "no-cache" flag. Do not re-introduce cache comparison modes.

### P8 — No async on the pixel path

Pixel processing is synchronous and CPU-bound. See section 9 for the full technical
argument. The short version: async is designed to solve a different problem (I/O latency)
and imposes costs (future state machines, waker overhead, executor scheduling) that make
tight pixel loops slower, not faster.

### P9 — Never load the full image into memory

Operations must process images in tiles, exactly as libvips does. Loading the entire image
into a single flat buffer is **prohibited**, regardless of image size. The tile pipeline
guarantees that only the active working set (one tile per thread, sized to fit L2 cache)
is resident at any time. This is the mechanism that lets viprs process multi-gigabyte
images with tens of megabytes of RAM. Any code path that materializes the full image
violates the memory model of the system.

### P10 — No unnecessary copies

Never copy pixel data when a view or borrow suffices. Copies are prohibited on:
- Structural operations (covered by P5)
- Buffer handoff between pipeline stages — pass `&mut [T]` slices, not `Vec<T>`
- Region arguments to `process_region` — these are always slices into pre-allocated buffers

If you find yourself constructing a `Vec` to hold pixel data at call sites inside the
pixel path, you have introduced a copy. Use `&mut [T]` instead.

### P11 — Cache-friendly access patterns

Tile geometry (P3) is the primary cache tool, but individual operations must also
access memory in sequential, stride-aware order. Rules:
- Iterate pixels in row-major order (x inner, y outer) to match how tiles are laid out.
- Never access pixels in column-major order unless the operation semantics require it
  (e.g., vertical convolution) — in that case, declare `FatStrip` demand so the
  scheduler buffers enough rows for the halo.
- Prefer `chunks_exact(bands)` over index arithmetic — LLVM can vectorize the former
  and eliminate bounds checks on the latter.
- Halo operations must not fetch more rows than the kernel radius demands (P2).

### P12 — Follow LLVM optimization guidelines on pixel paths

The LLVM optimization section in `docs/PERFORMANCE.md` (§ "Writing LLVM-friendly Rust
in pixel paths") is mandatory reading before implementing any op. Key rules:

- **Bounds check elimination**: assert slice lengths before entering the hot loop so
  LLVM can eliminate per-iteration bounds checks.
- **Loop-invariant code motion**: hoist all lookups, table accesses, and clamp
  computations outside the inner loop; LLVM LICM is not always reliable.
- **`#[inline]`**: mark `process_region` and inner pixel functions `#[inline]` so
  LLVM can specialize across monomorphized call sites.
- **`#[cold]`**: mark error and fallback paths `#[cold]` so LLVM keeps them out of
  the hot block and optimizes layout for the common case.
- **No branches inside pixel loops**: use saturating arithmetic, lookup tables, or
  `min`/`max` clamps rather than `if` inside the inner loop. Branches prevent SIMD
  vectorization.
- **Verify with `cargo xtask perf --metrics simd`**: after implementing, confirm SIMD%
  is >70% for throughput-critical ops. If not, use `cargo xtask profile --ai` to find
  the instruction the vectorizer rejected.

Reference: [LLVM Auto-Vectorization docs](https://llvm.org/docs/Vectorizers.html),
[Godbolt](https://godbolt.org/) for checking emitted asm.

### P13 — Every performance change requires tool evidence

No performance change may be merged without empirical evidence from the performance
toolchain. "It should be faster" is not evidence. Required for every P-NNN task:

| Evidence type | Tool |
|---|---|
| Baseline ratio (viprs vs libvips) | `cargo xtask bench <image> <op> --iterations 30` |
| Bottleneck function | `cargo xtask profile <image> <op> --ai` |
| SIMD instruction ratio | `cargo xtask perf <image> <op> --metrics simd` |
| Allocations on pixel path | `cargo xtask perf <image> <op> --metrics alloc` |
| After-fix ratio confirmation | `cargo xtask bench` (same command, same image) |

A PR that claims a performance improvement without before/after benchmark output
will not pass review regardless of the measured result.

---

## 7. SOLID applied to Rust

### Single Responsibility

Each `Operation` impl does one thing and names that thing precisely.
If a name requires "and" or "or" to describe what it does, split it.
Operations do not parse files, manage threads, or allocate persistent state.

### Open / Closed

The system is extended by implementing a port trait, never by modifying an existing
`match` arm, `if let`, or function. If adding a feature requires editing a `match` on
an operation type, the operation type is wrongly modeled — replace it with a trait.

### Liskov Substitution

Every implementation of a port trait must satisfy the trait's documented contract —
not just compile. Tests for the contract belong to the trait definition itself
(in `ports/`), exercised via a generic test function that any implementor can run
against itself. Write these contract tests before the first implementor exists.

### Interface Segregation

When a caller needs only one method from a trait, that method should be its own trait.
Port traits are split by caller need, not by implementation convenience. An operation
struct may implement multiple narrow traits; callers depend only on what they use.

### Dependency Inversion

Concrete types depend on abstractions (port traits), not on other concrete types.
The `Pipeline` depends on the `Operation` trait, not on `Sharpen`. Codecs are injected
into the foreign adapter registry; they are never imported by name inside the pipeline.
Construction of concrete types happens at the application boundary (or test setup),
not inside domain or port code.

---

## 8. Error handling

### Use typed error enums, not trait objects

Library functions return `Result<T, ViprsError>`. `Box<dyn Error>` and `anyhow::Error`
are forbidden in library-facing APIs. Typed variants let callers match and recover
without downcasting.

### Error variants carry context

An error variant must include enough information to produce a useful message without
a backtrace. "Invalid argument" is not enough; "Invalid argument: sigma must be > 0,
got -1.5" is. Encode the context in the variant fields, not in a free-form string.

### Errors in TDD

Write tests for error paths before writing the validation logic. A function that can
fail has two kinds of tests: success path and each distinct failure mode. Both are
written in Red before Green.

### No `panic!` for user-triggered conditions

`panic!` is reserved for violated programmer invariants (logic bugs). Bad user input,
unsupported formats, and out-of-range parameters return `Err(ViprsError::…)`.
`unwrap()` and `expect()` are banned in library code outside `#[cfg(test)]`.

---

## 9. Concurrency model

### Horizontal threading — threads run across tiles, not across stages

Most image processing systems parallelize *vertically*: stage A runs in thread 1, stage
B in thread 2, connected by a bounded queue. libvips — and viprs — parallelizes
*horizontally*: each thread runs the **entire pipeline** on a different output tile
simultaneously.

The consequence is that the number of locks per output tile is constant regardless of
pipeline depth or thread count. Adding a new operation to a ten-stage pipeline does not
add synchronization overhead to any existing stage. This is the property that makes
linear CPU scaling achievable.

The execution model that enables this — compiled linear pipeline (topological sort +
pre-assigned buffer indices + rayon tile dispatch) — is documented with full evidence in


### Per-thread state isolation is the mechanism, not a nice-to-have

The `start/generate/stop` contract from libvips maps directly to the `Operation` trait:

- **Construction** (`Pipeline::build`): allocates per-thread state — input region
  handles, scratch buffers. Happens once per thread per pipeline run.
- **`process_region`**: uses only its own per-thread state and the mutable output slice.
  Contains **zero shared mutable state**. Runs fully in parallel across threads.
- **Drop**: releases per-thread state.

This is why generate functions need no locks between each other. If `process_region`
needs to read from a shared immutable structure (a convolution kernel, a colour matrix),
that structure must be `Arc<[T]>` or a `&'static` reference — never guarded by a mutex
that multiple threads contend on during processing.

### Thread-local buffer pools

Each thread maintains a private pool of pixel buffers. When a tile is done, the buffer
returns to the pool rather than being freed. This eliminates global allocator calls from
the pixel path — the allocator is a shared resource protected by locks, and contention on
it is a primary scaling bottleneck in multi-threaded code.

### Shared pixel buffers are immutable after write

Input tiles are immutable (`&Tile`). Output tiles are exclusively owned by the operation
for the duration of `process_region`. No two operations write to the same memory region
concurrently. Enforce this through the type system, not through runtime locks.

### No `Arc<Mutex<T>>` around pixel data

If two pipeline stages need to share a buffer, one is the sole producer and the other
is the sole consumer. Express this with ownership transfer or `Arc<[T]>` (immutable
shared reference after write is complete), never with `Arc<Mutex<[T]>>`.

### No async runtimes

Async runtimes (Tokio, async-std, smol) are banned from the pixel path.
The short reason: they solve I/O-latency problems; viprs has a CPU-throughput problem.


---

## 10. Module structure (libvips parity)

Each libvips module maps to a subdirectory under `src/domain/ops/`:

- `arithmetic/` — add, subtract, multiply, divide, abs, …
- `colour/` — colorspace conversions (sRGB ↔ Lab ↔ XYZ …)
- `conversion/` — cast, flip, rotate, flatten, …
- `convolution/` — conv, compass, convsep, sharpen, …
- `create/` — black, eye, grey, sines, zone, …
- `draw/` — draw_circle, draw_line, draw_flood, …
- `foreign/` — top-level format dispatcher (delegates to `adapters/codecs/`)
- `freqfilt/` — fwfft, freqmult, spectrum, …
- `histogram/` — hist_find, hist_norm, hist_cum, …
- `morphology/` — morph, rank, …
- `mosaicing/` — merge, mosaic, globalbalance, …
- `resample/` — resize, affine, thumbnail, mapim, …

When implementing an operation from libvips:

1. **Name parity**: map `vips_sharpen` → `Sharpen`. Keep the libvips name so the
   reference docs apply directly.
2. **Parameter parity**: support all non-deprecated parameters; reject deprecated ones
   with a clear `ViprsError::InvalidArgument`.
3. **Streaming correctness**: implement `required_input_region` before `process_region`.
   Operations that require a halo (e.g., convolution radius) must expand the input
   region accordingly. Write a test for the region expansion before the pixel math.
4. **Format generics**: generic over all applicable band formats via a bound
   (e.g., `F: NumericBand`). Never restrict to a single concrete format.
5. **Golden test before pixel math**: store the libvips reference output in
   `tests/fixtures/<op>/` and write the golden test before implementing the
   `process_region` body.

---

## 11. Dependency policy

### Pre-approved (no review needed)

- `thiserror` — error derives
- `rayon` — data parallelism (feature-gated, enabled by default)
- `memmap2` — file-backed and anonymous memory maps (feature `mmap`, enabled by default)
- `proptest` — property-based testing (dev-dependency only)
- `criterion` — benchmarks (dev-dependency only)

### Allowed with justification

- `wide` — stable SIMD (document which operations use it and the throughput gain)
- Format codec crates (`mozjpeg`, `png`, `webp`) — each behind its own feature flag

### Forbidden

- Any async runtime (`tokio`, `async-std`, `smol`, …)
- `anyhow` in library code — callers cannot match on `anyhow::Error`; use typed errors
- Any crate that transitively links to C libvips — viprs is a native reimplementation
- `serde` on hot-path domain types — serialization is an adapter concern

### Adding a dependency

1. Run `cargo bloat --release` before and after; include the delta in the PR description.
2. Verify the crate's MSRV does not exceed the project's MSRV (tracked in `Cargo.toml`).
3. Prefer `no_std`-compatible crates — a future embedded target is in scope.
4. If transitive dependency count grows by more than five, the PR requires a discussion
   before merge.

---

## Issue filing obligation

Every agent — regardless of role — **must open a GitHub issue** when it encounters
a problem worth recording that is outside the scope of its current task.

Examples of what triggers an issue:

- A test that fails for reasons unrelated to the current work.
- A bug discovered while reading code (wrong algorithm, off-by-one, missing edge case).
- A `// TODO`, `// FIXME`, or `// HACK` with no corresponding issue.
- A performance anomaly noticed during profiling or benchmarking.
- A documentation gap or misleading doc comment.
- A CI or toolchain friction point that cost time.
- An architectural violation (e.g., `dyn Trait` on a hot path, `unwrap` in library code).

**Do not fix out-of-scope problems inline.** File an issue and keep working on the
assigned task. The issue is the artifact — it ensures nothing is silently forgotten.

```bash
gh issue create --title "<concise title>" \
  --body "<what you found, where, and why it matters>" \
  --label "<bug|enhancement|performance>"
```

Before filing, check that no open issue already covers the same finding:

```bash
gh issue list --search "<keywords>" --state open
```

If an existing issue partially covers it, add a comment instead of creating a duplicate.
