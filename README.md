# viprs

[![CI](https://github.com/mbertogliati/viprs/actions/workflows/ci.yml/badge.svg)](https://github.com/mbertogliati/viprs/actions/workflows/ci.yml)
[![crates.io](https://img.shields.io/crates/v/viprs.svg)](https://crates.io/crates/viprs)
[![docs.rs](https://docs.rs/viprs/badge.svg)](https://docs.rs/viprs)
[![License: MIT](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)
[![MSRV](https://img.shields.io/badge/MSRV-1.92-orange.svg)](https://www.rust-lang.org)

`viprs` is a native Rust reimplementation of libvips: demand-driven, horizontally-threaded image processing with a performance-first architecture.

## Why use it?

- Native Rust pipeline and scheduling model
- C-backed hot codecs where throughput matters (`jpeg`, `webp`, `heif`, `openslide`)
- Demand-driven execution instead of eager whole-image transforms
- Bench tooling to compare `viprs` against libvips directly

## Quick start

```rust,no_run
use viprs::prelude::*;

fn main() -> Result<(), ViprsError> {
    ImageApi::open("input.jpg")?
        .thumbnail(400)?
        .save("thumb.jpg")?;
    Ok(())
}
```

## Runnable examples

```bash
cargo run --example thumbnail --features jpeg -- input.jpg thumb.jpg 400
```

## Use Cases

| Scenario | How viprs handles it |
|----------|---------------------|
| **Web thumbnailing** | Decode → thumbnail → encode in a single pipeline, bytes-in/bytes-out |
| **CDN image optimization** | Chain resize + sharpen + quality reduction without intermediate buffers |
| **Batch processing** | Thread-pool scheduler processes tiles in parallel across all cores |
| **Scientific imaging** | Float32/Float64 pipelines with EXR, FITS, NIfTI codecs |
| **Color correction** | Lab/XYZ/LCh colorspace conversions + ICC profile transforms |
| **Concurrent HTTP service** | Multiple requests processed in parallel with linear scaling |
| **Large uploads** | Demand-driven evaluation — only decode tiles that are needed |

## Available Operations

### Arithmetic
`add`, `subtract`, `multiply`, `divide`, `linear`, `invert`, `abs`, `sign`, `round`,
`clamp`, `sum`, `power`, `sqrt`, `exp`, `log`, `remainder`, `recomb`,
`sin`, `cos`, `tan`, `asin`, `acos`, `atan`, `maxpair`, `minpair`

### Boolean & Bitwise
`and`, `or`, `xor`, `lshift`, `rshift`

### Relational
`equal`, `not_equal`, `less`, `less_eq`, `more`, `more_eq`

### Colour
`colourspace` (sRGB ↔ Lab ↔ XYZ ↔ LCh ↔ HSV ↔ scRGB ↔ CMYK ↔ GREY16 ↔ YXY),
`icc_transform`, `dE76`, `dE00`, `dECMC`

### Convolution
`gauss_blur`, `sharpen`, `unsharp_mask`, `conv`, `convsep`,
`sobel`, `scharr`, `prewitt`, `canny`, `compass`, `edge`, `fastcor`, `spcor`

### Resample
`resize`, `thumbnail`, `shrink`, `shrinkh`, `shrinkv`, `reduce`, `reduceh`, `reducev`,
`affine`, `similarity`, `mapim`, `quadratic`, `zoom`

### Structural
`extract_area`, `embed`, `flip_horizontal`, `flip_vertical`,
`rotate90`, `rotate180`, `rotate270`, `join`, `insert`, `replicate`, `subsample`,
`flatten`, `premultiply`, `unpremultiply`

### Frequency Domain (feature: `fft`)
`fwfft`, `invfft`, `freqmult`, `phasecor`, `spectrum`

### Morphology
`rank`, `erode`, `dilate`, `median`

### Histogram
`hist_find`, `hist_cum`, `hist_norm`, `hist_equal`, `hist_match`

### Create / Generators
`black`, `gaussnoise`, `xyz`, `zone`, `eye`, `sines`, `text`, `gaussmat`, `logmat`

## Feature flags

### Runtime and processing

| Feature | Default | Purpose | Notes |
|---|---:|---|---|
| `rayon` | yes | Parallel tile scheduler | Recommended for production pipelines |
| `mmap` | yes | Memory-mapped sources | Useful for large local files |
| `simd-pulp` | yes | SIMD dispatch helpers | Faster arithmetic/resample paths |
| `fft` | no | FFT / frequency-domain ops | Enables `fwfft` / `invfft` |
| `ffi` | no | FFI surface | For embedding from non-Rust callers |
| `lock_instrumentation` | no | Scheduler lock diagnostics | Debug-only instrumentation |

### Codec and format support

| Feature | Default | What it enables | Native/system dependency |
|---|---:|---|---|
| `jpeg` | no | JPEG decode + encode | `libjpeg-turbo` / `libturbojpeg` |
| `png` | no | PNG decode + encode | none |
| `libspng` | no | Alternative PNG backend | `png` feature plus `spng` crate |
| `webp` | no | WebP decode + encode | C-backed `libwebp` path |
| `bmp` | no | BMP decode + encode | none |
| `tiff` | no | TIFF decode + encode | none |
| `gif` | no | GIF decode + encode | none |
| `exr` | no | OpenEXR support | none |
| `radiance` | no | Radiance HDR support | none |
| `pfm` | no | Portable Float Map support | none |
| `svg` | no | SVG rasterization | none |
| `avif` | no | AVIF encode/decode path | `libheif` |
| `heif` | no | HEIF/HEIC decode | `libheif` |
| `jxl` | no | JPEG XL decode | libjxl-backed wrapper path |
| `jp2k` | no | JPEG 2000 support | OpenJPEG / `openjp2` |
| `icc` | no | ICC profile transforms | `lcms2` |
| `openslide` | no | Whole-slide microscopy | `openslide` |
| `fits` | no | FITS codec | `cfitsio` toolchain via `fitsio-sys` |
| `mat-hdf5` | no | MATLAB v7.3 / HDF5 MAT support | HDF5 Rust stack |
| `nifti` | no | NIfTI support | none |
| `uhdr` | no | Ultra HDR support | builds on `jpeg` |
| `deepzoom` | no | DeepZoom export | no extra deps beyond output codecs |
| `dcraw` | no | RAW import via external tooling | `dcraw` / `dcraw_emu` on PATH |
| `magick` | no | ImageMagick fallback load/save | `magick` CLI on PATH |
| `pdf-poppler` | no | PDF rasterization fallback | `pdfinfo` / `pdftoppm` on PATH |
| `csv` | no | CSV matrix codec | none |
| `vips-format` | no | Native `.v` / `.vips` format | none |
| `pnm` | no | PBM/PGM/PPM/PNM | none |

## System dependencies

These are the features most likely to fail with linker or header errors when their native dependencies are missing.

### macOS (Homebrew)

```bash
brew install pkgconf jpeg-turbo webp libheif openslide little-cms2 openjpeg
```

### Ubuntu / Debian

```bash
sudo apt-get update
sudo apt-get install -y \
  pkg-config \
  libturbojpeg0-dev \
  libwebp-dev \
  libheif-dev \
  libopenslide-dev \
  liblcms2-dev \
  libopenjp2-7-dev
```

### Fedora

```bash
sudo dnf install -y \
  pkgconf-pkg-config \
  libjpeg-turbo-devel \
  libwebp-devel \
  libheif-devel \
  openslide-devel \
  lcms2-devel \
  openjpeg2-devel
```

### Feature-by-feature notes

| Feature | What to install | Why |
|---|---|---|
| `jpeg` | `libjpeg-turbo` + `pkg-config` | `turbojpeg` links to `libturbojpeg` |
| `webp` | `libwebp` packages or a working C toolchain | `viprs` intentionally uses the C-backed WebP path |
| `heif`, `avif` | `libheif` | HEIF/AVIF integration goes through libheif |
| `openslide` | `openslide` | Whole-slide reader bindings need native headers/libs |
| `icc` | `lcms2` | ICC transforms use Little CMS |
| `jp2k` | `openjpeg` | JPEG 2000 stack depends on OpenJPEG |

## Benchmarks

| Command | Measures | Use it for |
|---|---|---|
| `cargo bench` | Criterion microbenchmarks | Regressions in a specific op |
| `cargo xtask bench <input> <op>` | viprs vs libvips latency/resources | Publishable side-by-side comparisons |
| `cargo xtask perf <input> <op> --metrics alloc` | Allocation count / bytes | Hot-path allocation audits |
| `cargo xtask perf <input> <op> --metrics simd` | SIMD instruction ratio | Vectorization validation |
| `cargo xtask perf <input> <op> --metrics hw` | Cache / PMU metrics | Deep performance investigations |

## Minimum Supported Rust Version (MSRV)

**1.92**

## License

[MIT](LICENSE) © Matias Bertogliati
