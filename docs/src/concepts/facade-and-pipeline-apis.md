# Facade And Pipeline APIs

`viprs` needs two public layers.

The facade API is for the common production path. It should be short, readable, and easy
to embed in an HTTP handler or background worker:

```rust,no_run
use viprs::prelude::*;

fn main() -> Result<(), ViprsError> {
    ImageApi::open("input.jpg")?
        .thumbnail(400)?
        .save("thumb.jpg")?;

    Ok(())
}
```

The pipeline API is for callers that need direct control over operation nodes, regions,
tiles, schedulers, codec selection, and resource behavior. This layer can be more explicit
because its users are choosing control over convenience.

Both layers should share the same underlying execution model. The facade should not be a
separate implementation. It should build clear pipeline plans and expose sensible defaults.
