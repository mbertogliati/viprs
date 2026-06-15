# bench-vs-libvips — Head-to-head performance tracking

Compares viprs against libvips C on real workloads.  Every CI run on `master`
and every PR appends to a persistent trend log so regressions are visible over
time, not just on the run where they occurred.

---

## Quick start (local)

```bash
# Prerequisites (macOS)
brew install vips imagemagick pkg-config

# Prerequisites (Ubuntu/Debian)
sudo apt-get install libvips-dev curl python3 pkg-config
python3 -m pip install OpenEXR Imath numpy

# 1. Build the benchmark runners
make -C tools/bench-vs-libvips

# 2. Generate benchmark fixtures and e2e goldens (only needed once)
tools/gen-fixtures.sh

# 3. Run a single operation against all three standard sizes
cargo xtask bench tests/fixtures/images/sample.jpg invert --sizes --iterations 20

# 4. Run the explicit non-RGB compute baseline matrix
cargo xtask bench tests/fixtures/images invert --scenario-set compute-baselines --iterations 20

# 5. Run a single operation on the 1024x1024 parity fixture
cargo xtask bench tests/fixtures/images/bench_1024x1024.jpg thumbnail 800 --iterations 50

# 5. Run the canonical thumbnail / Gaussian blur parameter matrices
cargo xtask bench tests/fixtures/images/bench_2048x2048.jpg thumbnail --matrix --iterations 20
cargo xtask bench tests/fixtures/images/bench_2048x2048.jpg gauss_blur --matrix \
    --no-e2e --iterations 20
cargo xtask bench tests/fixtures/images/bench_2048x2048.jpg colourspace --matrix \
    --sizes --no-e2e --iterations 20
```

### Supported operations

| Operation | Example command | Notes |
|-----------|-----------------|-------|
| `invert`    | `cargo xtask bench <img> invert` | |
| `flatten`   | `cargo xtask bench <img> flatten --sizes --no-e2e` | RGBA→RGB alpha flatten against black; `--sizes` uses RGBA PNG fixtures |
| `save-avif` | `cargo xtask bench <img> save-avif [u8\|u16] --no-e2e` | codec encode-only baseline |
| `save-exr`  | `cargo xtask bench <img> save-exr` | float EXR encode baseline via `openexr-runner`; `--sizes` uses float EXR fixtures at 512 / 2048 / 8192 |
| `save-heif` | `cargo xtask bench <img> save-heif --no-e2e` | HEIF/HEIC encode-only baseline |
| `save-jp2k` | `cargo xtask bench <img> save-jp2k --no-e2e` | JPEG 2000 encode-only baseline |
| `resize`    | `cargo xtask bench <img> resize 0.5` | scale factor |
| `shrinkh`   | `cargo xtask bench <img> shrinkh 5 --no-e2e` | isolated horizontal integer shrink |
| `shrinkv`   | `cargo xtask bench <img> shrinkv 5 --no-e2e` | isolated vertical integer shrink |
| `thumbnail` | `cargo xtask bench <img> thumbnail 800` | target width in px |
| `thumbnail` matrix | `cargo xtask bench <img> thumbnail --matrix` | widths `100/200/400/800` |
| `gauss_blur` matrix | `cargo xtask bench <img> gauss_blur --matrix` | sigma `0.5/1.0/2.0/5.0` |
| `colourspace` | `cargo xtask bench <img> colourspace [dest ...]` | one or more destination hops from the input interpretation |
| `colourspace` matrix | `cargo xtask bench <img> colourspace --matrix --sizes` | Lab / XYZ / CMYK / HSV / scRGB routes across 512 / 2048 / 8192 |
| `convolve` | `cargo xtask bench <img> convolve --no-e2e` | representative 3×3 box kernel via `conv` |
| `sobel` | `cargo xtask bench <img> sobel --no-e2e` | direct Sobel edge detector |
| `prewitt` | `cargo xtask bench <img> prewitt --no-e2e` | direct Prewitt edge detector |
| `laplacian` | `cargo xtask bench <img> laplacian --no-e2e` | fixed 3×3 Laplacian kernel via `conv` |
| `median_blur` | `cargo xtask bench <img> median_blur 3 --no-e2e` | square median filter (`vips_median`) |
| `unsharp_mask` | `cargo xtask bench <img> unsharp_mask 0.5 3.0 --no-e2e` | simple Gaussian subtract/add sharpen |
| `sharpen`   | `cargo xtask bench <img> sharpen 0.5 3.0` | sigma, strength |

### Explicit input matrix coverage

Use `--sizes` for RGB U8 size sweeps and `--scenario-set compute-baselines` for the
explicit non-RGB baseline matrix across 512 / 2048 / 8192:

| Coverage | Fixtures | Example |
|---|---|---|
| 1-band grayscale | `tests/fixtures/images/bench_{size}x{size}_gray.png` | `cargo xtask bench tests/fixtures/images invert --scenario-set compute-baselines` |
| 4-band RGBA | `tests/fixtures/images/bench_{size}x{size}_rgba.png` | same matrix |
| 16-bit RGB | `tests/fixtures/images/bench_{size}x{size}_u16.tif` | same matrix |
| float RGB | `tests/fixtures/images/bench_{size}x{size}.exr` | same matrix |

The scenario-set matrix is limited to band-agnostic ops plus `colourspace`:
`load`, `invert`, `flatten`, `linear`, `colourspace`, `resize`, `shrink`, `shrinkh`, `shrinkv`,
`thumbnail`, `gauss_blur`.

---

## Interpreting results

```
--- size 2048x2048 ---
  libvips:  p50=12.34 ms  p95=13.01 ms
  viprs:    p50=14.67 ms  p95=15.22 ms
  ratio p50: 1.189x  (viprs 1.19x slower)
```

- **ratio = viprs / libvips** — a value below 1.0 means viprs is faster.
- **p50** is the median latency; **p95** reflects tail latency.
- The CI gate fails if any ratio exceeds **2.0x** on any of the three standard
  sizes (512 / 2048 / 8192).  This is deliberately permissive to account for
  CI runner variability; use `cargo xtask bench` locally for fine-grained
  comparisons.

---

## Historical trend data

Every `--sizes` run appends one JSON record per (op, size) pair to:

```
tools/bench-vs-libvips/results/trend.jsonl
```

### Record schema

```json
{
  "date":           "2026-06-06T14:30:00Z",
  "git_sha":        "a1b2c3d",
  "op":             "invert",
  "size":           2048,
  "viprs_p50_ms":   14.67,
  "viprs_p95_ms":   15.22,
  "libvips_p50_ms": 12.34,
  "libvips_p95_ms": 13.01,
  "ratio_p50":      1.189
}
```

Fields `libvips_p50_ms`, `libvips_p95_ms`, and `ratio_p50` are `null` only when the
required baseline runner is missing. A crashing baseline runner now fails the benchmark
command instead of emitting partial JSON with null ratios.

### Querying the trend log

```bash
# All invert records at 2048px
jq 'select(.op=="invert" and .size==2048)' \
    tools/bench-vs-libvips/results/trend.jsonl

# Show ratio history for resize at 8192px
jq -r 'select(.op=="resize" and .size==8192) | [.date, .git_sha, .ratio_p50] | @tsv' \
    tools/bench-vs-libvips/results/trend.jsonl

# Detect any run where ratio exceeded 1.5x
jq 'select(.ratio_p50 != null and .ratio_p50 > 1.5)' \
    tools/bench-vs-libvips/results/trend.jsonl
```

### CI artifact retention

Each CI run also uploads per-run JSON files (one per op/size combination) as a
GitHub Actions artifact named `bench-results-<op>-<run_id>`, retained for 30 days.
The `trend.jsonl` is included in that artifact.

---

## File layout

```
tools/bench-vs-libvips/
├── Makefile           — builds libvips-runner and openexr-runner
├── libvips_runner.c   — libvips C benchmark runner (outputs JSON to stdout)
├── openexr_runner.cpp — OpenEXR encode runner for `save-exr`
├── run.sh             — thin shell wrapper (legacy; prefer cargo xtask bench)
├── README.md          — this file
└── results/           — generated output (gitignored except trend.jsonl)
    ├── trend.jsonl    — persistent trend log (one record per line)
    └── <op>_NxN_<ts>.json  — per-run detail files
```

The `results/` directory is gitignored by default.  The `trend.jsonl` file
can optionally be committed to track history in the repository itself — see the
discussion in the project issues for the tradeoffs.

---

## Adding a new operation

To add a new benchmark scenario:

1. **libvips C runner** (`libvips_runner.c`): add a `run_<op>` function and a
   case in the `op_name` dispatch block.

2. **xtask bench** (`xtask/src/bench.rs`): add a case in `run_viprs_pipeline`
   that builds the equivalent viprs pipeline branch.

3. **CI matrix** (`.github/workflows/bench-vs-libvips.yml`): add the op name
   to `jobs.bench.strategy.matrix.op`.

See existing `sharpen` as a worked example of all three steps.
