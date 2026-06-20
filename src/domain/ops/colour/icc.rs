use crate::domain::{
    error::ViprsError,
    format::{F32, U8, U16},
    image::Image,
};

#[cfg(feature = "icc")]
use crate::domain::{error::BuildError, image::Interpretation, op::DynOperation};

#[cfg(not(feature = "icc"))]
const ICC_DETAILS: &str =
    "ICC operations require the deferred CMS adapter boundary and littlecms2 dependency.";

#[cfg(feature = "icc")]
fn icc_error(message: impl Into<String>) -> ViprsError {
    ViprsError::Codec(format!("icc: {}", message.into()))
}

#[cfg(feature = "icc")]
fn lcms_error(err: lcms2::Error) -> ViprsError {
    icc_error(err.to_string())
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
/// Enumerates the available icc intent values.
pub enum IccIntent {
    /// Uses the `Perceptual` variant of `IccIntent`.
    Perceptual,
    #[default]
    /// Uses the `Relative` variant of `IccIntent`.
    Relative,
    /// Uses the `Saturation` variant of `IccIntent`.
    Saturation,
    /// Uses the `Absolute` variant of `IccIntent`.
    Absolute,
    /// Uses the `Auto` variant of `IccIntent`.
    Auto,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
/// Configures icc transform.
pub struct IccTransformOptions<'a> {
    /// Stores the `input_profile` value for this item.
    pub input_profile: Option<&'a [u8]>,
    /// Stores the `intent` value for this item.
    pub intent: IccIntent,
    /// Stores the `black_point_compensation` value for this item.
    pub black_point_compensation: bool,
    /// Stores the `depth` value for this item.
    pub depth: Option<u8>,
}

#[derive(Debug)]
/// Enumerates the available icc image values.
pub enum IccImage {
    /// Uses the `U8` variant of `IccImage`.
    U8(Image<U8>),
    /// Uses the `U16` variant of `IccImage`.
    U16(Image<U16>),
    /// Uses the `F32` variant of `IccImage`.
    F32(Image<F32>),
}

impl IccImage {
    #[must_use]
    /// Returns this value as u8.
    pub const fn as_u8(&self) -> Option<&Image<U8>> {
        match self {
            Self::U8(image) => Some(image),
            Self::U16(_) | Self::F32(_) => None,
        }
    }

    #[must_use]
    /// Returns this value as u16.
    pub const fn as_u16(&self) -> Option<&Image<U16>> {
        match self {
            Self::U16(image) => Some(image),
            Self::U8(_) | Self::F32(_) => None,
        }
    }

    #[must_use]
    /// Returns this value as f32.
    pub const fn as_f32(&self) -> Option<&Image<F32>> {
        match self {
            Self::U8(_) | Self::U16(_) => None,
            Self::F32(image) => Some(image),
        }
    }
}

#[cfg(not(feature = "icc"))]
const fn cms_unimplemented(feature: &'static str) -> ViprsError {
    ViprsError::Unimplemented {
        feature,
        details: ICC_DETAILS,
    }
}

#[cfg(not(feature = "icc"))]
mod stub;

#[cfg(feature = "icc")]
mod normalize;
#[cfg(feature = "icc")]
mod profiles;
#[cfg(all(test, feature = "icc"))]
mod tests;
#[cfg(feature = "icc")]
mod transform;

#[cfg(not(feature = "icc"))]
pub use stub::{icc_export, icc_import, icc_transform, profile_load};

#[cfg(feature = "icc")]
pub(crate) use normalize::{
    build_normalize_to_srgb_op, needs_srgb_normalization, srgb_profile_bytes,
};
#[cfg(feature = "icc")]
pub use profiles::profile_load;
#[cfg(feature = "icc")]
pub use transform::{icc_export, icc_import, icc_transform};
