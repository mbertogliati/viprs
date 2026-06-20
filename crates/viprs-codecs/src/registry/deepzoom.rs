use std::path::Path;

use viprs_core::{
    codec_options::{LoadOptions, SaveOptions},
    error::ViprsError,
    format::BandFormat,
    image::Image,
};

#[cfg(feature = "deepzoom")]
use viprs_core::format::{BandFormatId, U8};

#[cfg(feature = "deepzoom")]
use crate::deepzoom::DeepZoomExporter;

use super::ForeignRegistry;

pub fn is_deepzoom_extension(path: &Path) -> bool {
    path.extension()
        .and_then(std::ffi::OsStr::to_str)
        .is_some_and(|ext| matches!(ext.to_ascii_lowercase().as_str(), "dz" | "dzi" | "szi"))
}

#[cfg(feature = "deepzoom")]
trait DeepZoomQuantize {
    fn quantize_to_u8(value: Self, unit_scale: bool) -> u8
    where
        Self: Sized;

    fn uses_unit_scale(_pixels: &[Self]) -> bool
    where
        Self: Sized,
    {
        false
    }
}

#[cfg(feature = "deepzoom")]
impl DeepZoomQuantize for u8 {
    fn quantize_to_u8(value: Self, _unit_scale: bool) -> u8 {
        value
    }
}

#[cfg(feature = "deepzoom")]
impl DeepZoomQuantize for u16 {
    fn quantize_to_u8(value: Self, _unit_scale: bool) -> u8 {
        (value / 257) as u8
    }
}

#[cfg(feature = "deepzoom")]
impl DeepZoomQuantize for i16 {
    fn quantize_to_u8(value: Self, _unit_scale: bool) -> u8 {
        value.clamp(0, i16::from(u8::MAX)) as u8
    }
}

#[cfg(feature = "deepzoom")]
impl DeepZoomQuantize for u32 {
    fn quantize_to_u8(value: Self, _unit_scale: bool) -> u8 {
        (value / 16_843_009) as u8
    }
}

#[cfg(feature = "deepzoom")]
impl DeepZoomQuantize for i32 {
    fn quantize_to_u8(value: Self, _unit_scale: bool) -> u8 {
        value.clamp(0, i32::from(u8::MAX)) as u8
    }
}

#[cfg(feature = "deepzoom")]
impl DeepZoomQuantize for f32 {
    fn quantize_to_u8(value: Self, unit_scale: bool) -> u8 {
        quantize_float_to_u8(f64::from(value), unit_scale)
    }

    fn uses_unit_scale(pixels: &[Self]) -> bool {
        float_pixels_use_unit_scale(pixels.iter().map(|&value| f64::from(value)))
    }
}

#[cfg(feature = "deepzoom")]
impl DeepZoomQuantize for f64 {
    fn quantize_to_u8(value: Self, unit_scale: bool) -> u8 {
        quantize_float_to_u8(value, unit_scale)
    }

    fn uses_unit_scale(pixels: &[Self]) -> bool {
        float_pixels_use_unit_scale(pixels.iter().copied())
    }
}

#[cfg(feature = "deepzoom")]
fn float_pixels_use_unit_scale<I>(values: I) -> bool
where
    I: Iterator<Item = f64>,
{
    let mut saw_finite = false;
    for value in values {
        if !value.is_finite() {
            continue;
        }
        saw_finite = true;
        if !(0.0..=1.0).contains(&value) {
            return false;
        }
    }

    saw_finite
}

#[cfg(feature = "deepzoom")]
fn quantize_float_to_u8(value: f64, unit_scale: bool) -> u8 {
    if !value.is_finite() {
        return 0;
    }

    let scaled = if unit_scale { value * 255.0 } else { value };
    scaled.clamp(0.0, 255.0) as u8
}

#[cfg(feature = "deepzoom")]
fn quantize_pixels_to_u8<T: DeepZoomQuantize + Copy>(pixels: &[T]) -> Vec<u8> {
    let unit_scale = T::uses_unit_scale(pixels);
    let mut quantized = Vec::with_capacity(pixels.len());
    quantized.extend(
        pixels
            .iter()
            .copied()
            .map(|value| T::quantize_to_u8(value, unit_scale)),
    );
    quantized
}

#[cfg(feature = "deepzoom")]
pub(crate) fn to_u8_image<F: BandFormat>(image: &Image<F>) -> Result<Image<U8>, ViprsError> {
    let quantized = match F::ID {
        BandFormatId::U8 => quantize_pixels_to_u8::<u8>(bytemuck::cast_slice(image.pixels())),
        BandFormatId::U16 => quantize_pixels_to_u8::<u16>(bytemuck::cast_slice(image.pixels())),
        BandFormatId::I16 => quantize_pixels_to_u8::<i16>(bytemuck::cast_slice(image.pixels())),
        BandFormatId::U32 => quantize_pixels_to_u8::<u32>(bytemuck::cast_slice(image.pixels())),
        BandFormatId::I32 => quantize_pixels_to_u8::<i32>(bytemuck::cast_slice(image.pixels())),
        BandFormatId::F32 => quantize_pixels_to_u8::<f32>(bytemuck::cast_slice(image.pixels())),
        BandFormatId::F64 => quantize_pixels_to_u8::<f64>(bytemuck::cast_slice(image.pixels())),
    };

    Image::<U8>::from_buffer(image.width(), image.height(), image.bands(), quantized)
        .map(|image_u8| image_u8.with_metadata(image.metadata().clone()))
}

#[cfg(feature = "deepzoom")]
pub(crate) fn save_deepzoom<F: BandFormat>(
    image: &Image<F>,
    path: &Path,
    opts: &SaveOptions,
) -> Result<(), ViprsError> {
    let typed = to_u8_image(image)?;
    let exporter = DeepZoomExporter::from_options(opts)?;
    exporter.export(&typed, path)
}

#[cfg(not(feature = "deepzoom"))]
pub const fn save_deepzoom<F: BandFormat>(
    _image: &Image<F>,
    _path: &Path,
    _opts: &SaveOptions,
) -> Result<(), ViprsError> {
    Err(ViprsError::Unimplemented {
        feature: "foreign encode: deepzoom",
        details: "DeepZoom export is not yet implemented (dzsave tile pyramid backend).",
    })
}

/// Convenience codec I/O methods for [`Image`].
pub trait ImageCodecExt: Sized {
    /// Load an image from disk using the shared foreign registry.
    fn load(path: impl AsRef<Path>) -> Result<Self, ViprsError>;

    /// Load an image using explicit codec options.
    fn load_with_options(path: impl AsRef<Path>, opts: &LoadOptions) -> Result<Self, ViprsError>;

    /// Save an image to disk using the shared foreign registry.
    fn save(&self, path: impl AsRef<Path>) -> Result<(), ViprsError>;

    /// Save an image using explicit codec options.
    fn save_with_options(
        &self,
        path: impl AsRef<Path>,
        opts: &SaveOptions,
    ) -> Result<(), ViprsError>;
}

impl<F: BandFormat> ImageCodecExt for Image<F> {
    /// `load` exposes adapter behavior needed by the surrounding module.
    /// Call it when you need the concrete operation implemented here.
    ///
    /// # Examples
    ///
    /// ```ignore
    /// let _ = viprs::adapters::codecs::registry::load;
    /// ```
    fn load(path: impl AsRef<Path>) -> Result<Self, ViprsError> {
        ForeignRegistry::shared().load_as(path.as_ref())
    }

    /// `load_with_options` exposes adapter behavior needed by the surrounding module.
    /// Call it when you need the concrete operation implemented here.
    ///
    /// # Examples
    ///
    /// ```ignore
    /// let _ = viprs::adapters::codecs::registry::load_with_options;
    /// ```
    fn load_with_options(path: impl AsRef<Path>, opts: &LoadOptions) -> Result<Self, ViprsError> {
        ForeignRegistry::shared().load_as_with_options(path.as_ref(), opts)
    }

    /// `save` exposes adapter behavior needed by the surrounding module.
    /// Call it when you need the concrete operation implemented here.
    ///
    /// # Examples
    ///
    /// ```ignore
    /// let _ = viprs::adapters::codecs::registry::save;
    /// ```
    fn save(&self, path: impl AsRef<Path>) -> Result<(), ViprsError> {
        ForeignRegistry::shared().save_as(self, path.as_ref())
    }

    /// `save_with_options` exposes adapter behavior needed by the surrounding module.
    /// Call it when you need the concrete operation implemented here.
    ///
    /// # Examples
    ///
    /// ```ignore
    /// let _ = viprs::adapters::codecs::registry::save_with_options;
    /// ```
    fn save_with_options(
        &self,
        path: impl AsRef<Path>,
        opts: &SaveOptions,
    ) -> Result<(), ViprsError> {
        ForeignRegistry::shared().save_as_with_options(self, path.as_ref(), opts)
    }
}
