//! Grey image source adapter.
//!
//! This module exposes concrete source implementations or helpers that feed
//! pixels into compiled pipelines.

use std::marker::PhantomData;

use crate::{
    domain::{
        error::ViprsError,
        format::{BandFormatId, F32, U8},
        image::{DemandHint, ImageMetadata, Region},
    },
    ports::source::{DynImageSource, ImageSource, RandomAccessSource},
};

use super::common::{PointSourceFormat, clamp_coord, validate_output_len, write_sample};

struct GreySourceInner<F: PointSourceFormat> {
    width: u32,
    height: u32,
    bands: u32,
    _format: PhantomData<F>,
}

impl<F: PointSourceFormat> GreySourceInner<F> {
    const fn new(width: u32, height: u32, bands: u32) -> Self {
        Self {
            width,
            height,
            bands,
            _format: PhantomData,
        }
    }
}

impl<F: PointSourceFormat> ImageSource for GreySourceInner<F> {
    type Format = F;

    fn width(&self) -> u32 {
        self.width
    }

    fn height(&self) -> u32 {
        self.height
    }

    fn bands(&self) -> u32 {
        self.bands
    }

    fn demand_hint(&self) -> DemandHint {
        DemandHint::Any
    }

    #[inline]
    fn read_region(&self, region: Region, output: &mut [u8]) -> Result<(), ViprsError> {
        validate_output_len(
            region,
            self.bands,
            std::mem::size_of::<F::Sample>(),
            output,
            self.width,
            self.height,
        )?;

        let denom = self.width.saturating_sub(1);
        let region_width = region.width as usize;
        let bands = self.bands as usize;

        for row in 0..region.height as usize {
            for col in 0..region_width {
                let x = clamp_coord(region.x + col as i32, self.width);
                let value = if denom == 0 {
                    F::from_unit_interval(0.0)
                } else {
                    F::from_unit_interval(f64::from(x) / f64::from(denom))
                };
                let pixel_base = (row * region_width + col) * bands;
                for band in 0..bands {
                    write_sample(output, pixel_base + band, value);
                }
            }
        }

        Ok(())
    }
}

impl<F: PointSourceFormat> RandomAccessSource for GreySourceInner<F> {}

/// Synthetic horizontal grey ramp matching libvips output format selection.
pub struct GreySource(GreySourceKind);

enum GreySourceKind {
    F32(GreySourceInner<F32>),
    U8(GreySourceInner<U8>),
}

impl GreySource {
    /// Creates a horizontal grey ramp in either `u8` or `f32` form.
    #[must_use]
    pub const fn new(width: u32, height: u32, bands: u32, uchar: bool) -> Self {
        if uchar {
            Self(GreySourceKind::U8(GreySourceInner::new(
                width, height, bands,
            )))
        } else {
            Self(GreySourceKind::F32(GreySourceInner::new(
                width, height, bands,
            )))
        }
    }

    /// Returns the generated image width.
    #[must_use]
    pub fn width(&self) -> u32 {
        match &self.0 {
            GreySourceKind::F32(source) => ImageSource::width(source),
            GreySourceKind::U8(source) => ImageSource::width(source),
        }
    }

    #[must_use]
    /// `height` exposes adapter behavior needed by the surrounding module.
    /// Call it when you need the concrete operation implemented here.
    ///
    /// # Examples
    ///
    /// ```ignore
    /// let _ = viprs::adapters::sources::generators::grey::height;
    /// ```
    pub fn height(&self) -> u32 {
        match &self.0 {
            GreySourceKind::F32(source) => ImageSource::height(source),
            GreySourceKind::U8(source) => ImageSource::height(source),
        }
    }

    /// Returns the configured band count for the grey ramp.
    #[must_use]
    pub fn bands(&self) -> u32 {
        match &self.0 {
            GreySourceKind::F32(source) => ImageSource::bands(source),
            GreySourceKind::U8(source) => ImageSource::bands(source),
        }
    }

    #[must_use]
    /// `format` exposes adapter behavior needed by the surrounding module.
    /// Call it when you need the concrete operation implemented here.
    ///
    /// # Examples
    ///
    /// ```ignore
    /// let _ = viprs::adapters::sources::generators::grey::format;
    /// ```
    pub const fn format(&self) -> BandFormatId {
        match &self.0 {
            GreySourceKind::F32(_) => BandFormatId::F32,
            GreySourceKind::U8(_) => BandFormatId::U8,
        }
    }

    /// Reads one region from the grey ramp into `output`.
    pub fn read_region(&self, region: Region, output: &mut [u8]) -> Result<(), ViprsError> {
        match &self.0 {
            GreySourceKind::F32(source) => ImageSource::read_region(source, region, output),
            GreySourceKind::U8(source) => ImageSource::read_region(source, region, output),
        }
    }
}

impl DynImageSource for GreySource {
    fn width(&self) -> u32 {
        Self::width(self)
    }

    fn height(&self) -> u32 {
        Self::height(self)
    }

    fn bands(&self) -> u32 {
        Self::bands(self)
    }

    fn format(&self) -> BandFormatId {
        Self::format(self)
    }

    fn demand_hint(&self) -> DemandHint {
        match &self.0 {
            GreySourceKind::F32(source) => ImageSource::demand_hint(source),
            GreySourceKind::U8(source) => ImageSource::demand_hint(source),
        }
    }

    fn metadata(&self) -> ImageMetadata {
        match &self.0 {
            GreySourceKind::F32(source) => ImageSource::metadata(source),
            GreySourceKind::U8(source) => ImageSource::metadata(source),
        }
    }

    fn set_shrink_on_load(&mut self, factor: std::num::NonZeroU8) -> Result<bool, ViprsError> {
        match &mut self.0 {
            GreySourceKind::F32(source) => ImageSource::set_shrink_on_load(source, factor),
            GreySourceKind::U8(source) => ImageSource::set_shrink_on_load(source, factor),
        }
    }

    fn set_thumbnail_shrink_on_load(
        &mut self,
        factor: std::num::NonZeroU8,
    ) -> Result<bool, ViprsError> {
        match &mut self.0 {
            GreySourceKind::F32(source) => {
                ImageSource::set_thumbnail_shrink_on_load(source, factor)
            }
            GreySourceKind::U8(source) => ImageSource::set_thumbnail_shrink_on_load(source, factor),
        }
    }

    fn read_region(&self, region: Region, output: &mut [u8]) -> Result<(), ViprsError> {
        Self::read_region(self, region, output)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sample_size(source: &GreySource) -> usize {
        match source.format() {
            BandFormatId::F32 => std::mem::size_of::<f32>(),
            BandFormatId::U8 => std::mem::size_of::<u8>(),
            _ => unreachable!(),
        }
    }

    fn read_all(source: &GreySource, region: Region) -> Vec<u8> {
        let mut output =
            vec![0u8; region.pixel_count() * source.bands() as usize * sample_size(source)];
        source.read_region(region, &mut output).unwrap();
        output
    }

    #[test]
    fn dimensions_match_declared_values() {
        let source = GreySource::new(7, 9, 2, false);
        assert_eq!(source.width(), 7);
        assert_eq!(source.height(), 9);
        assert_eq!(source.bands(), 2);
        assert_eq!(source.format(), BandFormatId::F32);
    }

    #[test]
    fn read_region_is_deterministic() {
        let source = GreySource::new(5, 3, 1, false);
        let region = Region::new(1, 0, 3, 2);
        let first = read_all(&source, region);
        let second = read_all(&source, region);
        assert_eq!(first, second);
    }

    #[test]
    fn float_mode_spans_zero_to_one() {
        let source = GreySource::new(4, 1, 1, false);
        let output = read_all(&source, Region::new(0, 0, 4, 1));
        let samples: &[f32] = bytemuck::cast_slice(&output);

        assert_eq!(source.format(), BandFormatId::F32);
        assert!((samples[0] - 0.0).abs() < 1e-6);
        assert!((samples[1] - (1.0 / 3.0)).abs() < 1e-6);
        assert!((samples[2] - (2.0 / 3.0)).abs() < 1e-6);
        assert!((samples[3] - 1.0).abs() < 1e-6);
    }

    #[test]
    fn uchar_mode_emits_u8_samples() {
        let source = GreySource::new(4, 1, 1, true);
        let output = read_all(&source, Region::new(0, 0, 4, 1));

        assert_eq!(source.format(), BandFormatId::U8);
        assert_eq!(output, vec![0, 85, 170, 255]);
    }
}
