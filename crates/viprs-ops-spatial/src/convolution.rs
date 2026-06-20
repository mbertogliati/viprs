//! Convolution-based filters that use neighbourhood kernels around each pixel.
/// Provides the `canny` module for this domain area.
pub mod canny;
mod common;
/// Provides domain support for `compass`.
pub mod compass;
/// Provides the `conv` module for this domain area.
pub mod conv;
/// Provides the `conv2d` module for this domain area.
pub mod conv2d;
/// Provides the `conva` module for this domain area.
pub mod conva;
/// Provides the `convasep` module for this domain area.
pub mod convasep;
/// Provides the `convsep` module for this domain area.
pub mod convsep;
/// Provides the `edge` module for this domain area.
pub mod edge;
/// Provides the `fastcor` module for this domain area.
pub mod fastcor;
pub mod gauss_blur;
/// Provides the `prewitt` module for this domain area.
pub mod prewitt;
/// Provides the `scharr` module for this domain area.
pub mod scharr;
/// Provides the `sharpen` module for this domain area.
pub mod sharpen;
/// Provides the `sobel` module for this domain area.
pub mod sobel;
/// Provides the `spcor` module for this domain area.
pub mod spcor;
pub use crate::resample::sample_conv::ToF64;
pub use canny::Canny;
pub use common::{ConvolutionMask1d, ConvolutionMask2d};
pub use compass::{CompassOp, PREWITT_COMPASS_MASK};
pub use conv::{ApproximatePrecision, ConvOp, ConvPrecision, FloatPrecision, IntegerPrecision};
pub use conva::ConvaOp;
pub use convasep::ConvaSepOp;
pub use convsep::{ConvSep, ConvSepH, ConvSepV};
pub use edge::{EdgeOp, PREWITT_EDGE_MASK, SCHARR_EDGE_MASK, SOBEL_EDGE_MASK};
pub use fastcor::FastCorOp;
pub use gauss_blur::{GaussBlur, GaussBlurH, GaussBlurV, GaussOutput, GaussOutputFormat};
pub use prewitt::Prewitt;
pub use scharr::Scharr;
pub use sharpen::{LabSSharpen, Sharpen};
pub use sobel::Sobel;
pub use spcor::{SpcorOp, SpcorOp as CorrelationOp};
