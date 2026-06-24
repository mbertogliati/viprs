# Why Rust

The case for `viprs` is not only memory safety. Safety matters, but a native Rust design
also changes what the library can optimize and how it can be deployed.

## Compile-time specialization

Image processing spends most of its time in hot loops. Rust generics allow operations to
be specialized by band format and operation type without paying for dynamic dispatch on
the pixel path. This is why the repository bans `dyn Trait` in hot domain code unless
static dispatch is impossible.

## Ownership-driven zero-copy design

Rust's ownership model makes buffer lifetimes and mutation boundaries explicit. That is
useful for demand-driven pipelines because intermediate images should be avoided unless
they are required by the algorithm. The design goal is to pass slices and tiles through
the pipeline instead of allocating per pixel or per tile.

## Scheduler control

libvips is known for horizontal threading. A Rust implementation can expose scheduling as
a typed port, letting applications choose the right thread pool, backpressure model, and
resource limits for their environment. A web service, batch worker, and edge optimizer do
not have identical scheduling needs.

## Deployment story

Rust produces static or mostly self-contained binaries for many targets. That matters for
containers, edge deployments, internal CLIs, and workers where installing a full native
image stack can be harder than shipping one binary with selected feature flags.

## Typed errors

Production callers need to map failures to HTTP status codes, retry decisions, audit
logs, and user-facing messages. `viprs` uses concrete error types instead of library
facing `Box<dyn Error>` APIs so callers can distinguish corrupt input, unsupported
formats, invalid parameters, resource limits, and infrastructure failures.

## Feature-gated native dependencies

Some codecs need mature native libraries. Others can be pure Rust. The crate should let
applications opt into exactly the codec and processing surface they need, keeping small
deployments small while still allowing heavy production builds where throughput matters.

## SIMD-friendly abstractions

Rust can express fast scalar code while still leaving room for explicit SIMD and
target-specific implementations. `viprs` should keep abstractions simple enough for the
compiler and profiling tools to show whether hot loops vectorize and whether allocator
calls are absent from the pixel path.
