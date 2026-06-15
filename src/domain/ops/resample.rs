//! Resampling operations: reduce, resize, affine, thumbnail.
//!
//! All resampling ops operate on `F: BandFormat` and change image dimensions.
//! They implement `ResampleOp` (a refinement of `Op`) which adds
//! `output_width` / `output_height` declarations used by the pipeline compiler.
//!
//! Design rationale: enum kernels, separate H/V structs,
//! composite thumbnail).

pub(crate) mod reduce_common;
pub(crate) mod reduce_simd;
pub(crate) mod sample_conv;

pub mod affine;
/// Provides the `mapim` module for this domain area.
pub mod mapim;
pub mod quadratic;
/// Provides the `reduce` module for this domain area.
pub mod reduce;
pub mod reduceh;
pub mod reducev;
pub mod resize;
pub mod shrink;
pub mod shrinkh;
pub mod shrinkv;
/// Provides the `similarity` module for this domain area.
pub mod similarity;
pub mod thumbnail;

pub use affine::Affine;
pub use mapim::MapImOp;
pub use quadratic::{Quadratic, QuadraticCoefficients};
pub use reduce::ReduceOp;
pub use reduceh::ReduceH;
pub use reducev::ReduceV;
pub use resize::{Resize, ResizeOp};
pub use shrink::Shrink;
pub use shrinkh::ShrinkH;
pub use shrinkv::ShrinkV;
pub use similarity::{InterpolationKind, SimilarityOp};
pub use thumbnail::Thumbnail;
