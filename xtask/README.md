# xtask — viprs development tooling

Cargo xtask for head-to-head benchmarking of viprs against libvips C.

## Usage

```bash
cargo xtask bench <input> <op> [op_args...] [--iterations N] [--sizes|--scenario-set NAME]
```

### Examples

```bash
# Single benchmark fixture (I/O parity baseline)
cargo xtask bench tests/fixtures/images/bench_1024x1024.jpg invert --iterations 50
cargo xtask bench tests/fixtures/images/bench_1024x1024.jpg resize 0.5 --iterations 30
cargo xtask bench tests/fixtures/images/bench_1024x1024.jpg thumbnail 400 --iterations 20
cargo xtask bench tests/fixtures/images/bench_2048x2048.jpg thumbnail --matrix --iterations 20
cargo xtask bench tests/fixtures/images/bench_2048x2048.jpg gauss_blur --matrix --iterations 20

# Multi-size sweep (512 / 2048 / 8192) — reads bench_NxN.jpg from fixtures
cargo xtask bench tests/fixtures/images/sample.jpg invert --sizes
cargo xtask bench tests/fixtures/images/sample.jpg resize 0.5 --sizes --iterations 20
cargo xtask bench tests/fixtures/images/sample.jpg thumbnail --sizes --matrix --iterations 20
cargo xtask bench tests/fixtures/images/sample.jpg colourspace --sizes --matrix --iterations 20

# Input-diversity sweep for compute baselines
cargo xtask bench tests/fixtures/images invert --scenario-set compute-baselines --iterations 20
cargo xtask bench tests/fixtures/images colourspace --scenario-set input-diversity --iterations 20
```

### Multi-size mode (`--sizes`)

Runs the benchmark across the three standard sizes defined in B-87 (512x512, 2048x2048,
8192x8192) and prints a unified summary table:

```
=== SUMMARY: invert (ratio = viprs/libvips, <1.0 means viprs wins) ===
size        lv p50 ms    lv p95 ms    vp p50 ms    vp p95 ms   ratio p50   ratio p95
------------------------------------------------------------------------------------------
512x512          2.13         2.66         1.49         1.94      0.698x      0.730x
2048x2048        7.78         8.25        15.39        16.96      1.977x      2.055x
8192x8192      145.23      1199.62       226.69       287.61      1.561x      0.240x
```

Required benchmark images are located by convention at:
  `tests/fixtures/images/bench_NxN.jpg`

Generate them once with:
```bash
tools/gen-fixtures.sh
```

The `--sizes` flag ignores the `<input>` argument (it is required by the parser but unused
in multi-size mode).

### Scenario-set mode (`--scenario-set`)

Runs the same op across a named fixture matrix. `compute-baselines` (alias:
`input-diversity`) makes non-RGB coverage explicit across the three standard sizes:

| Scenario family | Fixtures | Coverage |
|---|---|---|
| `gray-u8-{512,2048,8192}` | `tests/fixtures/images/bench_{size}x{size}_gray.png` | 1-band grayscale compute baseline |
| `rgba-u8-{512,2048,8192}` | `tests/fixtures/images/bench_{size}x{size}_rgba.png` | 4-band RGBA compute baseline |
| `rgb-u16-{512,2048,8192}` | `tests/fixtures/images/bench_{size}x{size}_u16.tif` | 16-bit compute baseline |
| `rgb-f32-{512,2048,8192}` | `tests/fixtures/images/bench_{size}x{size}.exr` | float compute baseline |

Supported ops for this matrix: `load`, `invert`, `linear`, `colourspace`, `resize`,
`shrink`, `shrinkh`, `shrinkv`, `thumbnail`, `gauss_blur`.

### Canonical parameter matrices (`--matrix`)

For ops whose performance is highly parameter-dependent, `cargo xtask bench` can expand a
maintained argument matrix automatically:

| Operation | Canonical matrix |
|---|---|
| `thumbnail` | target widths `100`, `200`, `400`, `800` |
| `gauss_blur` | sigma values `0.5`, `1.0`, `2.0`, `5.0` |
| `colourspace` | destination chains `lab`, `lab→srgb`, `xyz`, `xyz→srgb`, `cmyk`, `cmyk→srgb`, `hsv`, `hsv→srgb`, `scrgb`, `scrgb→srgb` |

Examples:

```bash
# Single-size thumbnail sweep on the 2048 fixture
cargo xtask bench tests/fixtures/images/bench_2048x2048.jpg thumbnail --matrix --iterations 20

# Multi-size thumbnail sweep across all standard fixtures
cargo xtask bench tests/fixtures/images/sample.jpg thumbnail --sizes --matrix --iterations 20

# Gaussian blur sigma sweep (kernel-only)
cargo xtask bench tests/fixtures/images/bench_2048x2048.jpg gauss_blur --matrix \
  --no-e2e --iterations 20

# Colourspace route sweep across all standard sizes
cargo xtask bench tests/fixtures/images/bench_2048x2048.jpg colourspace --matrix \
  --sizes --iterations 20
```

The matrix is codified in `xtask/src/bench/helpers.rs` so future investigations keep using
the same parameter set.

## Supported operations

| Operation | Args | Description |
|---|---|---|
| `invert` | — | Per-pixel inversion |
| `linear` | `<scale> <offset>` (f64, defaults `2.0 5.0`) | Per-sample affine transform |
| `gauss_blur` | `<sigma>` (f64, default 1.5) | Two-pass Gaussian blur |
| `convolve` | — | Fixed 3×3 box kernel via `ConvOp` / `vips_conv` |
| `sobel` | — | Sobel edge detector |
| `prewitt` | — | Prewitt edge detector |
| `laplacian` | — | Fixed 3×3 Laplacian kernel via `ConvOp` / `vips_conv` |
| `median_blur` | `<size>` (u32, default `3`) | Square median filter |
| `unsharp_mask` | `<sigma> <strength>` (f64, defaults `0.5 3.0`) | Direct Gaussian subtract/add sharpen |
| `colourspace` | `[dest ...]` (defaults by input interpretation) | Bench one or more `vips_colourspace` hops such as `lab srgb` or `hsv srgb` |
| `resize` | `<scale>` (f64, default 0.5) | Scale by factor using Lanczos3 |
| `shrinkh` | `<factor>` (u32, default 2) | Horizontal integer box shrink |
| `shrinkv` | `<factor>` (u32, default 2) | Vertical integer box shrink |
| `thumbnail` | `<width>` (u32, default 800) | Fit to width preserving aspect ratio |
| `sharpen` | `<sigma> <strength>` (f64, defaults `0.5 3.0`) | Libvips-style LabS sharpen |

## What it measures

Both sides run the same parity flow:

```
decode once from disk (pre-loop) → execute op on in-memory input → materialize pixels to memory
```

### Metrics collected

| Metric | Source | Why |
|---|---|---|
| Wall-clock latency (p50/p95) | `clock_gettime` / `Instant` | End-to-end throughput |
| Peak RSS | `getrusage.ru_maxrss` | Memory footprint |
| Page faults (minor/major) | `getrusage` | Memory pressure / cold starts |
| Context switches (vol/invol) | `getrusage` | Thread contention signal |

### Fairness guarantees

Both runners are structured to eliminate measurement bias:

| Concern | libvips C runner | viprs Rust runner |
|---|---|---|
| Operation cache | Disabled (`vips_cache_set_max(0)`) | No cache connected (B-91) |
| Output | `vips_image_write_to_memory` (no disk) | `MemorySink` (no disk) |
| Thread pool | Global (created once by libvips) | `RayonScheduler` created once, reused |
| Warmup | 3 iterations discarded | 3 iterations discarded |
| Decode | `vips_image_new_from_file` once, then `vips_image_new_from_memory` per iter | `Image::load` + `MemorySource` once per run |

## Adding a new operation

~10 lines per side, no infrastructure changes needed.

### Rust side (`xtask/src/bench.rs`)

Add a match arm in `run_viprs_pipeline`:

```rust
"new_op" => {
    // parse op_args as needed
    builder.new_op(params).expect("new_op failed")
}
```

Current encode-only ops that bypass the pipeline builder and benchmark the codec directly:
`save-avif`, `save-exr`, `save-gif`, `save-heif`, `save-jp2k`.

`save-exr` uses `tools/bench-vs-libvips/openexr-runner` for the baseline side because
libvips exposes `openexrload` but not `openexrsave`. When paired with `--sizes`, the
runner switches to the float EXR fixture set (`512 / 2048 / 8192`) instead of the default
JPEG matrix so the OpenEXR baseline receives a valid F32 input.

### C side (`tools/bench-vs-libvips/libvips_runner.c`)

1. Add a `run_new_op` function:
```c
static int run_new_op(const char *input) {
    VipsImage *in = vips_image_new_from_file(input, NULL);
    if (!in) return -1;
    VipsImage *out = NULL;
    int ret = vips_new_op(in, &out, NULL);
    g_object_unref(in);
    if (ret != 0 || !out) return -1;
    void *buf = vips_image_write_to_memory(out, NULL);
    g_object_unref(out);
    g_free(buf);
    return buf ? 0 : -1;
}
```

2. Add a wrapper + register in `main`:
```c
static int op_new_op(const char *input, void *ctx) {
    (void)ctx;
    return run_new_op(input);
}

// in main():
} else if (strcmp(op_name, "new_op") == 0) {
    fn = op_new_op;
}
```

3. Rebuild: `make -C tools/bench-vs-libvips`

## Prerequisites

```bash
brew install vips pkg-config
make -C tools/bench-vs-libvips libvips-runner  # one-time build
```

## Output

Results are saved as JSON to `tools/bench-vs-libvips/results/<op>_<timestamp>.json` and printed to stdout with comparison ratios:

```
--- comparison (ratio = viprs/libvips, <1.0 means viprs wins) ---
  latency p50: 0.605x
  latency p95: 0.530x
  RSS:         0.275x
  >>> viprs is 1.65x FASTER <<<
```

---

## `cargo xtask profile` — Side-by-side CPU and cache profiling

Profiles **both viprs and libvips** and shows where each spends its time.
Two backends — pick based on what you're debugging:

```bash
cargo xtask profile <input> <op> [args...]
                    [--tool samply|cachegrind]
                    [--arch arm64|amd64]
                    [--iterations N]
```

### Tools

| `--tool` | What you get | Docker? | Best for |
|---|---|---|---|
| `samply` (default) | CPU flame graph JSON per binary | No | "Which function is hottest?" |
| `cachegrind` | Per-function L1/LL cache-miss table | Yes | "Who's trashing the cache?" |

### Examples

```bash
# CPU flame graph — works locally on macOS and Linux
cargo xtask profile tests/fixtures/images/bench_2048x2048.jpg thumbnail 400 --tool samply

# Per-function cache-miss comparison (requires Docker)
cargo xtask profile tests/fixtures/images/bench_2048x2048.jpg thumbnail 400 --tool cachegrind

# x86 cache profile
cargo xtask profile tests/fixtures/images/bench_2048x2048.jpg invert --tool cachegrind --arch amd64
```

### samply output

Saves two JSON files to `/tmp/viprs_profile_<op>.json` and `/tmp/libvips_profile_<op>.json`.
Load both at https://profiler.firefox.com and switch tabs to compare flame graphs.

Install samply: `cargo install samply`

### cachegrind output

Prints a ranked table of last-level cache misses (DLmr) per function:

```
Function                                                      libvips DLmr   viprs DLmr    ratio
──────────────────────────────────────────────────────────────────────────────────────────────────
reduceh.rs::reduce_h_u8                                                210        18000    85.7x
scheduler.rs::dispatch_tiles                                            40           55     1.4x
jpeg.rs::decode_strip                                                 1200         1250     1.0x
```

A high ratio on a pixel-path function → wrong stride, tile too large, or bad data layout.
Cross-reference with `src/domain/ops/` vs `.libvips_repo/`.

---

## `cargo xtask perf` — Hard metrics

Deep profiling beyond wall-clock latency.

```bash
cargo xtask perf <input> <op> [args...] [--metrics all|hw|alloc|simd] [--arch arm64|amd64] [--iterations N]
```

### Metric categories

| Category | What it measures | Requires Docker? |
|---|---|---|
| `simd` | Static disassembly: SIMD datapath ratio inside op-matched viprs symbols | No |
| `alloc` | Heap allocations per iteration (count, bytes, peak live) | No |
| `hw` | perf stat: cache misses, IPC, branch mispredictions, TLB | Yes (Linux) |
| `all` | All three (default) | hw part needs Docker |

### Examples

```bash
# SIMD analysis — how vectorized is the selected op datapath? (local, instant)
cargo xtask perf tests/fixtures/images/sample.jpg invert --metrics simd

# Allocation profiling — how many heap allocs per operation? (local)
cargo xtask perf tests/fixtures/images/sample.jpg resize 0.5 --metrics alloc

# Hardware counters — cache misses, IPC (needs Docker on Linux)
cargo xtask perf tests/fixtures/images/sample.jpg thumbnail 400 --metrics hw --arch arm64

# Everything
cargo xtask perf tests/fixtures/images/sample.jpg invert --metrics all
```

### Docker setup for cache/hw profiling

```bash
# Start colima with the target architecture
colima start --arch aarch64 --cpu 4 --memory 4  # ARM64
colima start --arch x86_64 --cpu 4 --memory 4   # x86_64

# Build the profiling image (includes cachegrind, perf, DHAT)
docker buildx build --platform linux/arm64 -t viprs-perf:arm64 -f docker/Dockerfile .
docker buildx build --platform linux/amd64 -t viprs-perf:amd64 -f docker/Dockerfile .
```

### Two hw-counter backends

| Backend | How | Overhead | When to use |
|---|---|---|---|
| **cachegrind** (default) | Simulates I/D caches | 20-50x slower | Deterministic, always works, regression tracking |
| **perf stat** (bonus) | Real CPU PMU counters | ~0% | Real IPC/cycles, needs bare-metal Linux |

cachegrind is the primary tool because:
- Works in Docker Desktop (no PMU access needed)
- Deterministic: same input → same numbers (comparable across runs)
- Portable: identical event names on ARM and x86
- `perf stat` needs real PMU exposure — Docker Desktop on macOS doesn't provide this

### cachegrind events

| Event | What it tells you |
|---|---|
| `Ir` | Instruction references (total instruction count) |
| `Dr` / `Dw` | Data reads / writes |
| `D1mr` / `D1mw` | L1 data cache read/write misses |
| `DLmr` / `DLmw` | Last-level cache read/write misses |
| `Bc` / `Bcm` | Conditional branches / mispredicted |

### perf stat events (when PMU available)

| Event | What it tells you |
|---|---|
| `cycles` / `instructions` | IPC — instructions per cycle |
| `cache-references` / `cache-misses` | Last-level cache pressure |
| `L1-dcache-loads` / `L1-dcache-load-misses` | L1 data cache hit rate |
| `branches` / `branch-misses` | Branch prediction efficiency |
| `dTLB-load-misses` | TLB pressure (large working sets) |

Both tools are portable across ARM64 and x86_64. Event names are generic —
the kernel/valgrind maps them to architecture-specific PMU events.

### Counting allocator

The Rust runner uses an always-linked `#[global_allocator]` counting wrapper that is only
enabled while `cargo xtask perf --metrics alloc` is executing.

Tracked metrics:
- **alloc_count**: total `malloc` calls during measured iterations
- **alloc_bytes**: total bytes requested
- **per_iter_allocs**: average allocations per single op execution
- **per_iter_bytes**: average bytes per execution
- **peak_live_bytes**: maximum simultaneously-live heap at any point

### SIMD analysis

Disassembles release object files for the selected op and classifies datapath instructions:

- **ARM NEON/ASIMD/SVE**: `ld1`, `st1`, `fmla`, `fadd` (vector), `dup`, `tbl`, etc.
- **x86 SSE/AVX**: `vmov*`, `vadd*`, `vfma*`, `vbroadcast*`, `movaps`, `addps`, etc.
- **Scalar FP**: `fadd`, `fmul` (scalar), `addss`, `mulsd`, etc.

A low SIMD ratio (< 10%) on compute-heavy ops indicates vectorization opportunities.

---

## Future: benchmark modes

Currently all benchmarks run as E2E (decode → op → sink). Planned modes:

- `--mode e2e` (default): full production-equivalent flow
- `--mode op-only`: pre-decoded buffer, measures only the pixel operation
- `--mode decode-only`: measures codec performance in isolation
