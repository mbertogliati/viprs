# Contributing to viprs

Thank you for your interest in contributing! This document covers how to get started.

## Development Setup

```bash
# Clone
git clone https://github.com/mbertogliati/viprs.git
cd viprs

# Ensure Rust 1.92+ is installed
rustup update stable

# Verify the build
cargo check --lib
cargo test --lib
cargo clippy --lib -- -D clippy::perf
```

## Pull Request Process

1. Fork the repository and create a branch from `main`
2. Make your changes following the conventions below
3. Ensure all checks pass: `cargo fmt`, `cargo clippy`, `cargo test`
4. Open a PR with a **conventional commit title** (e.g., `feat: add TIFF rotation support`)
5. PRs are squash-merged; the PR title becomes the commit message on `main`

## Conventions

- **Language**: All code, comments, and documentation must be in English
- **Formatting**: `cargo fmt` — non-negotiable before commit
- **Linting**: `cargo clippy --lib -- -D clippy::perf` must pass
- **No `unwrap`/`expect`** in library code (only in tests)
- **No `dyn Trait`** on hot paths — monomorphize where possible
- **Zero allocations** in pixel-path code
- **Doc comments** on every public item with a usage example
- **Commit titles** follow [Conventional Commits](https://www.conventionalcommits.org/)

## PR Title Prefixes

| Prefix | When to use |
|--------|-------------|
| `feat:` | New feature or operation |
| `fix:` | Bug fix |
| `perf:` | Performance improvement |
| `refactor:` | Code restructuring (no behavior change) |
| `docs:` | Documentation only |
| `ci:` | CI/CD changes |
| `chore:` | Maintenance (deps, configs) |

## Performance

Every PR that touches operations must show no regression:
- `cargo bench` for microbenchmarks (Criterion)
- `cargo xtask bench <fixture> <op>` for comparison against libvips

A throughput drop > 5% on any standard size (512/2048/8192) blocks the PR.

## Architecture

See [AGENTS.md](AGENTS.md) for the full architectural map. Key rule: `domain/` imports nothing from `ports/` or `adapters/`.

## Questions?

Open a [Discussion](https://github.com/mbertogliati/viprs/discussions) for questions, ideas, or design proposals.
