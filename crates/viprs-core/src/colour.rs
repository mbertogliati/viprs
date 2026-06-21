//! `ColourConvert` port — colorspace conversion operations.
//!
//! Design rationale for colorspace conversion operations.

#![allow(clippy::type_complexity)]
// REASON: color-conversion dispatch signatures encode the exact zero-copy callback contracts.

use crate::{
    colorspace::Colorspace,
    format::BandFormat,
    image::{DemandHint, Region, Tile, TileMut},
    stats::ImageStats,
};

/// A colorspace conversion from `From` to `To`.
///
/// # Design
///
/// `ColourConvert` is separate from `Op` for two reasons:
///
/// 1. A colorspace conversion involves **two colorspace type parameters** (`From` and
///    `To`) that have no equivalent in `Op`. Encoding them in `Op`'s `Input`/`Output`
///    associated types would conflate the sample format with the colorspace, producing
///    combinatorial type explosion.
///
/// 2. Conversions are often pixel-local (sRGB→Lab is a per-pixel 3×3 matrix plus
///    non-linear function), matching `Op`'s tile model. But some conversions require
///    image-level statistics (e.g., histogram equalization) — expressing that in `Op`
///    is awkward. `ColourConvert` can optionally produce stats via `pre_stats`.
///
/// # Colorspace type parameters
///
/// `From` and `To` are zero-sized marker types (e.g., `SRgb`, `Lab`). They carry
/// no runtime data — their only role is to make invalid conversions (e.g.,
/// `Lab → Lab`) fail at compile time via an `where From::ID != To::ID` check
/// (enforced by convention, not by the type system — Rust cannot express inequality
/// constraints at the type level).
///
/// # Thread safety
///
/// `ColourConvert` must be `Send + Sync`. State computed during `pre_stats`
/// (e.g., LUT for histogram-based conversions) must be stored externally and passed
/// via `process_region`'s `state` parameter — not inside the struct.
pub trait ColourConvert<From: Colorspace, To: Colorspace>: Send + Sync {
    /// Input sample format (e.g., `U8` for sRGB, `F32` for Lab).
    type InputFormat: BandFormat;
    /// Output sample format. May differ from `InputFormat` (e.g., U8→F32 for sRGB→Lab).
    type OutputFormat: BandFormat;
    /// Per-thread mutable state. Use `()` for stateless conversions.
    type State: Send + 'static;

    /// Returns the tile-demand pattern required by this operation.
    fn demand_hint(&self) -> DemandHint;
    /// Returns the input region required to produce `output`.
    fn required_input_region(&self, output: &Region) -> Region;

    /// Optional image-level statistics pre-computation.
    ///
    /// Called once before tile dispatch. Returns `None` for conversions that do not
    /// need image-level stats (the common case). Return `Some(stats)` only for
    /// histogram-based conversions (e.g., local histogram equalization).
    fn pre_stats(&self) -> Option<Box<dyn Fn(&ImageStats) + Send + Sync>> {
        None
    }

    /// Creates per-thread state before region processing begins.
    fn start(&self) -> Self::State;

    /// Convert one tile from `From` colorspace to `To` colorspace.
    /// No heap allocations per pixel.
    fn convert_region(
        &self,
        state: &mut Self::State,
        input: &Tile<Self::InputFormat>,
        output: &mut TileMut<Self::OutputFormat>,
    );
}

// TODO: re-enable tests when ops crate exists.
// #[cfg(test)]
// mod tests {
//     ...
// }
