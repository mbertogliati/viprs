//! Port traits — infrastructure capabilities defined as narrow interfaces.
//!
//! Only traits that abstract over external infrastructure live here:
//! codecs, schedulers, I/O sources and sinks.
//!
//! Domain-facing operation traits (`Op`, `DynOperation`, `ColourConvert`,
//! `TileReducer`, `ResampleOp`, etc.) live in `viprs-core`.
//!
//! Concrete implementations live under the `viprs` root crate `adapters/` module.
//! Domain types are imported from `viprs_core`.

pub mod codec;
pub mod scheduler;
pub mod sink;
pub mod source;
