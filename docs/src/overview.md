# Overview

`viprs` is a native Rust reimplementation of libvips, the demand-driven,
horizontally-threaded image processing library.

The mission is to build image infrastructure for production systems where image size,
parallelism, memory pressure, and latency all matter. A successful `viprs` user should
be able to embed the library in a web service, background worker, CDN optimizer, large
image viewer, or asset generation backend without shelling out to a command-line tool or
crossing an FFI boundary for every operation.

This guide is organized around production environments rather than operation categories.
Operation lists are useful once the system is understood, but most users first need to
know whether the library fits their deployment model, resource constraints, and error
handling needs.

`viprs` aims for compatibility with libvips behavior where possible. The local
`.libvips_repo/` checkout is the reference implementation for algorithms and edge cases.
When behavior differs, the difference should be intentional, documented, and backed by
tests or benchmarks.

The project is still early. The lower-level architecture is more mature than the ideal
public facade. Expect the public API to become simpler as the crate moves toward a more
stable user-facing shape.
