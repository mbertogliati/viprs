//! General affine transform for upscaling, rotation, and skew.
//!
//! `Affine` handles the cases that the reduce family (`ReduceH`/`ReduceV`) does
//! not: upscaling (factor < 1.0), arbitrary rotation, and non-axis-aligned
//! transforms. It reads input pixels in non-sequential order, so it requires a
//! `TileCache` upstream when the source is sequential.
//!
//! The matrix maps **output** pixel coordinates to **input** pixel coordinates:
//!   `x_in` = a * `x_out` + b * `y_out` + tx
//!   `y_in` = c * `x_out` + d * `y_out` + ty
//!
//! Out-of-bounds input samples use a libvips-style extend mode. The default is
//! `Background(vec![0.0])`, matching libvips affine.

use std::marker::PhantomData;

use crate::domain::{format::BandFormat, kernel::InterpolationKernel};

pub use crate::domain::ops::conversion::embed::ExtendMode;

/// Affine transform with configurable interpolation kernel.
///
/// The 2×2 transform matrix is stored as `[a, b, c, d]` (row-major), mapping
/// output pixel coordinates to input pixel coordinates:
///
/// ```text
/// x_in = a * x_out + b * y_out + tx
/// y_in = c * x_out + d * y_out + ty
/// ```
///
/// Translation `(tx, ty)` is in input-pixel units.
/// Out-of-bounds input samples use `extend` (default `Background(vec![0.0])`).
///
/// Only the scalar reference path is implemented. SIMD paths are future work.
pub struct Affine<F: BandFormat> {
    /// Row-major 2×2 matrix: [a, b, c, d].
    matrix: [f64; 4],
    tx: f64,
    ty: f64,
    kernel: InterpolationKernel,
    // Used by `AffineBridge` in `adapters/pipeline.rs` to report fixed output dims.
    #[allow(dead_code)]
    output_w: u32,
    #[allow(dead_code)]
    output_h: u32,
    /// libvips-style extend mode for samples that fall outside the source tile.
    extend: ExtendMode,
    premultiplied: bool,
    fast_path: Option<AffineFastPath>,
    _format: PhantomData<F>,
}

struct AxisNearestSample {
    coord: i64,
}

struct AxisLinearSample {
    start: i64,
    weights: [f64; 2],
}

struct AxisCubicSample {
    start: i64,
    weights: [f64; 4],
}

enum AffineFastPath {
    Nearest {
        xs: Box<[AxisNearestSample]>,
        ys: Box<[AxisNearestSample]>,
    },
    Bilinear {
        xs: Box<[AxisLinearSample]>,
        ys: Box<[AxisLinearSample]>,
    },
    Bicubic {
        xs: Box<[AxisCubicSample]>,
        ys: Box<[AxisCubicSample]>,
    },
}

mod core;
mod interpolation;

#[cfg(test)]
mod tests;
