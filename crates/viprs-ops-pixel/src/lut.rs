//! Lookup-table operations that remap samples through precomputed tables.
/// Provides the `map_lut` module for this domain area.
pub mod map_lut;
/// Provides the `recomb` module for this domain area.
pub mod recomb;

pub use map_lut::MapLut;
pub use recomb::Recomb;
