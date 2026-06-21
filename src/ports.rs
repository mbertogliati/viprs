//! Port traits — infrastructure capabilities defined as narrow interfaces.
//!
//! Only traits that abstract over external infrastructure live here:
//! codecs, schedulers, I/O sources and sinks.
//!
//! Domain-facing operation traits (`Op`, `DynOperation`, `ColourConvert`,
//! `TileReducer`, `ResampleOp`, etc.) live in `src/domain/`.
//!
//! Concrete implementations live under `src/adapters/`.
//! Domain types (`src/domain/`) are imported by ports but never the reverse.

pub use viprs_ports::codec;
pub use viprs_ports::scheduler;
pub use viprs_ports::sink;
pub use viprs_ports::source;
