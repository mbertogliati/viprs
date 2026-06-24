# Getting Started

`viprs` is under active design. The examples below describe the intended user experience
and should be read together with the current API reference on docs.rs and the examples in
the repository.

## Install

Add the crate with the feature flags needed by your codecs:

```toml
[dependencies]
viprs = { version = "0.1", features = ["jpeg", "png", "webp"] }
```

For local development in this repository, use the pinned toolchain and Makefile:

```bash
rustup default 1.96.0
make check
```

## Intended facade shape

The high-level API should make common production flows readable:

```rust,no_run
use viprs::prelude::*;

fn main() -> Result<(), ViprsError> {
    ImageApi::open("input.jpg")?
        .thumbnail(400)?
        .save("thumb.jpg")?;

    Ok(())
}
```

This facade is for the common case. Advanced users should still be able to work with
tiles, regions, schedulers, codecs, and operation nodes directly.

## Examples

Runnable examples live in the repository:

```bash
cargo run --example thumbnail --features jpeg -- input.jpg thumb.jpg 400
```

When validating repository changes, use Makefile targets rather than raw `cargo`
commands. The Makefile is the contract shared by local development and CI.
