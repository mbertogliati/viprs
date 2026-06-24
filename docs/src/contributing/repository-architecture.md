# Repository Architecture

`viprs` follows strict module boundaries.

## Domain

`src/domain/` contains pure image-processing concepts and domain traits. It must not
import from `ports/` or `adapters/`.

Important examples include:

- `Image`, `Region`, and `Tile`.
- `BandFormat` and concrete band formats.
- Domain errors.
- Operation traits and operation implementations.
- Reducers and core processing concepts.

## Ports

`src/ports/` contains infrastructure traits only. These traits abstract over external
systems such as codecs, schedulers, sources, and sinks.

Ports should be narrow, typed, and `Send + Sync` where they may cross threads.

## Adapters

`src/adapters/` contains concrete infrastructure implementations. Codec implementations,
schedulers, sources, and sinks live here.

Adapters may depend on `domain/` and `ports/`. They should not cross-import other adapters
directly; shared behavior belongs behind a port or local helper.

## Public re-exports

`src/lib.rs` is for public re-exports. It should not accumulate business logic or
operation implementation details.
