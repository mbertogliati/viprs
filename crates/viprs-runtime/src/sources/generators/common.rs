//! Common image source adapter.
//!
//! This module exposes concrete source implementations or helpers that feed
//! pixels into compiled pipelines.

use bytemuck::Pod;

use crate::domain::{
    error::ViprsError,
    format::{BandFormat, F32, U8, U16},
    image::Region,
};

/// `validate_output_len` exposes adapter behavior needed by the surrounding module.
/// Call it when you need the concrete operation implemented here.
///
/// # Examples
///
/// ```ignore
/// let _ = viprs::adapters::sources::generators::common::validate_output_len;
/// ```
pub fn validate_output_len(
    region: Region,
    bands: u32,
    sample_size: usize,
    output: &[u8],
    image_width: u32,
    image_height: u32,
) -> Result<(), ViprsError> {
    let expected = region
        .pixel_count()
        .checked_mul(bands as usize)
        .and_then(|count| count.checked_mul(sample_size))
        .ok_or_else(|| ViprsError::RegionOutOfBounds {
            requested: format!(
                "output length overflow for region {region:?} with {bands} bands and sample size {sample_size}"
            ),
            width: image_width,
            height: image_height,
        })?;

    if output.len() != expected {
        return Err(ViprsError::RegionOutOfBounds {
            requested: format!(
                "output length {} != expected {} for region {:?}",
                output.len(),
                expected,
                region
            ),
            width: image_width,
            height: image_height,
        });
    }

    Ok(())
}

#[inline(always)]
/// `clamp_coord` exposes adapter behavior needed by the surrounding module.
/// Call it when you need the concrete operation implemented here.
///
/// # Examples
///
/// ```ignore
/// let _ = viprs::adapters::sources::generators::common::clamp_coord;
/// ```
pub fn clamp_coord(coord: i32, limit: u32) -> u32 {
    if limit == 0 {
        0
    } else {
        coord.clamp(0, limit as i32 - 1) as u32
    }
}

#[inline(always)]
/// `write_sample` exposes adapter behavior needed by the surrounding module.
/// Call it when you need the concrete operation implemented here.
///
/// # Examples
///
/// ```ignore
/// let _ = viprs::adapters::sources::generators::common::write_sample;
/// ```
pub fn write_sample<S: Pod>(output: &mut [u8], sample_index: usize, value: S) {
    let sample_size = std::mem::size_of::<S>();
    let start = sample_index * sample_size;
    output[start..start + sample_size].copy_from_slice(bytemuck::bytes_of(&value));
}

/// The `PointSourceFormat` trait defines behavior used by this adapter module.
/// Implement this trait when a concrete adapter type must participate in the surrounding workflow.
///
/// # Examples
///
/// ```ignore
/// fn accepts_trait<T: viprs::adapters::sources::generators::common::PointSourceFormat>() {}
/// let _ = accepts_trait::<fn()>;
/// ```
pub trait PointSourceFormat: BandFormat {
    fn from_unit_interval(value: f64) -> Self::Sample;
    fn from_signed_unit(value: f64) -> Self::Sample;
}

impl PointSourceFormat for F32 {
    #[inline(always)]
    fn from_unit_interval(value: f64) -> Self::Sample {
        value as f32
    }

    #[inline(always)]
    fn from_signed_unit(value: f64) -> Self::Sample {
        value as f32
    }
}

impl PointSourceFormat for U8 {
    #[inline(always)]
    fn from_unit_interval(value: f64) -> Self::Sample {
        (value.clamp(0.0, 1.0) * 255.0) as u8
    }

    #[inline(always)]
    fn from_signed_unit(value: f64) -> Self::Sample {
        (((value.clamp(-1.0, 1.0) + 1.0) * 0.5) * 255.0) as u8
    }
}

/// The `IdentitySourceFormat` trait defines behavior used by this adapter module.
/// Implement this trait when a concrete adapter type must participate in the surrounding workflow.
///
/// # Examples
///
/// ```ignore
/// fn accepts_trait<T: viprs::adapters::sources::generators::common::IdentitySourceFormat>() {}
/// let _ = accepts_trait::<fn()>;
/// ```
pub trait IdentitySourceFormat: BandFormat {
    fn from_index(index: u32) -> Self::Sample;
}

impl IdentitySourceFormat for U8 {
    #[inline(always)]
    fn from_index(index: u32) -> Self::Sample {
        index.min(u32::from(u8::MAX)) as u8
    }
}

impl IdentitySourceFormat for U16 {
    #[inline(always)]
    fn from_index(index: u32) -> Self::Sample {
        index.min(u32::from(u16::MAX)) as u16
    }
}
