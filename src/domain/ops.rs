//! Image processing operations grouped by domain-specific category.
/// Pixel-wise arithmetic operations re-exported from `viprs-ops-pixel`.
pub mod arithmetic {
    pub use crate::domain::reducers::{ProfileOp, ProfileResult, ProjectOp, ProjectResult};
    pub use viprs_ops_pixel::arithmetic::*;
}
pub use viprs_core::simd_util;
pub use viprs_ops_pixel::boolean;
pub use viprs_ops_pixel::lut;
pub use viprs_ops_pixel::point;
pub use viprs_ops_pixel::relational;

pub use viprs_ops_spatial::convolution;
pub use viprs_ops_spatial::morphology;
pub use viprs_ops_spatial::resample;
pub use viprs_ops_spatial::structural;

/// Histogram operations re-exported from `viprs-ops-colour`.
pub mod histogram {
    pub use crate::domain::reducers::{HistFindNDimReducer as HistFindNDimOp, HistFindNDimResult};
    pub use viprs_ops_colour::histogram::*;
}
pub use viprs_ops_colour::colour;

pub use viprs_ops_composite::conversion;
pub use viprs_ops_composite::create;
pub use viprs_ops_composite::draw;
pub use viprs_ops_composite::freqfilt;
pub use viprs_ops_composite::mosaicing;
