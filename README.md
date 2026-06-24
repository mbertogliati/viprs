# viprs

[![CI](https://github.com/mbertogliati/viprs/actions/workflows/ci.yml/badge.svg)](https://github.com/mbertogliati/viprs/actions/workflows/ci.yml)
[![crates.io](https://img.shields.io/crates/v/viprs.svg)](https://crates.io/crates/viprs)
[![docs.rs](https://docs.rs/viprs/badge.svg)](https://docs.rs/viprs)
[![License: MIT](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)
[![MSRV](https://img.shields.io/badge/MSRV-1.96-orange.svg)](rust-toolchain.toml)

`viprs` is a native Rust reimplementation of libvips: a demand-driven,
horizontally-threaded image processing library for systems where large images,
concurrency, memory use, and latency matter at the same time.

The project is not a wrapper around the C library. The goal is to keep the libvips
execution model and compatibility expectations while using Rust to explore stronger
compile-time specialization, ownership-driven zero-copy designs, typed errors,
controllable scheduling, feature-gated native dependencies, and SIMD-friendly
abstractions.

## Production use cases

- Web image services that transform uploads or remote images in HTTP handlers.
- Upload and ingest workers that generate derivatives for disk or object storage.
- CDN, edge, and media proxy optimizers that need predictable latency and memory.
- Scientific, medical, and geospatial systems that work with huge tiled images.
- Creative, document, and asset automation backends that need repeatable recipes.

## Quick start

The public facade API is still evolving, but the intended shape is a compact pipeline
that can be embedded in services and workers:

```rust,no_run
use viprs::prelude::*;

fn main() -> Result<(), ViprsError> {
    ImageApi::open("input.jpg")?
        .thumbnail(400)?
        .save("thumb.jpg")?;

    Ok(())
}
```

Runnable examples live in [`examples/`](examples/):

```bash
cargo run --example thumbnail --features jpeg -- input.jpg thumb.jpg 400
```

## Documentation

- Explanatory guide: [`docs/`](docs/)
- API reference: <https://docs.rs/viprs>
- Contributing: [`CONTRIBUTING.md`](CONTRIBUTING.md) and [`docs/src/contributing/`](docs/src/contributing/)
- Agent workflows: [`.github/agents/`](.github/agents/)
- Performance methodology: [`.github/agents/PERFORMANCE.md`](.github/agents/PERFORMANCE.md)

Routine validation goes through the Makefile:

```bash
make check
```
