# Contributor Guide

Contributions should preserve the central goal of `viprs`: native Rust image processing
with libvips-like demand-driven execution and production-grade performance.

Before touching code, read:

- `AGENTS.md`
- `.github/agents/GUIDELINES.md`
- `.github/agents/PERFORMANCE.md` for performance-sensitive work
- `.github/agents/CI_GUIDELINES.md` for CI changes

All code, comments, and documentation must be in English.

Routine validation goes through the Makefile:

```bash
make check
```

Do not leave raw `cargo` commands as the only validation path for a change. If a new
routine validation step is needed, add it to the Makefile so local development and CI stay
aligned.

For behavior defined by libvips, consult the local `.libvips_repo/` checkout before
implementing or documenting the operation.
