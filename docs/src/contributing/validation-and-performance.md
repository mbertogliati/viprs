# Validation And Performance

The Makefile is the routine validation interface.

Common targets:

```bash
make check
make ci
make fmt FIX=1
make bench
make bench-vs
```

Documentation-only PRs should run the cheapest relevant checks available. If `mdbook` is
installed, validate the guide with:

```bash
mdbook build docs
```

Performance-sensitive PRs need stronger evidence:

- Operation benchmarks in `benches/`.
- libvips comparison through the `xtask` benchmark tooling.
- Allocation and SIMD investigation where the hot path changes.
- No unexplained throughput regression over the allowed threshold.

The performance expectation is not aspirational text. It is part of the contract of the
project. Abstractions that make the code easier to read still need to justify their cost
in the hot path.
