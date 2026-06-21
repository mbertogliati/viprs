//! Eye image source adapter.
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

struct EyeSourceInner<F: PointSourceFormat> {
    width: u32,
    height: u32,
    factor: f64,
    _format: PhantomData<F>,
}

impl<F: PointSourceFormat> EyeSourceInner<F> {
    fn new(width: u32, height: u32, factor: f64) -> Result<Self, ViprsError> {
        if !(0.0..=1.0).contains(&factor) {
            return Err(ViprsError::Codec(format!(
                "EyeSource: factor must be in [0, 1], got {factor}"
            )));
        }

        Ok(Self {
            width,
            height,
            factor,
            _format: PhantomData,
        })
    }
}

impl<F: PointSourceFormat> ImageSource for EyeSourceInner<F> {
    type Format = F;

    fn width(&self) -> u32 {
        self.width
    }

    fn height(&self) -> u32 {
        self.height
    }

    fn bands(&self) -> u32 {
        1
    }

    fn demand_hint(&self) -> DemandHint {
        DemandHint::Any
    }

    #[inline]
    fn read_region(&self, region: Region, output: &mut [u8]) -> Result<(), ViprsError> {
        validate_output_len(
            region,
            1,
            std::mem::size_of::<F::Sample>(),
            output,
            self.width,
            self.height,
        )?;

        let max_x = self.width.saturating_sub(1).max(1);
        let max_y = self.height.saturating_sub(1).max(1);
        let c = self.factor * std::f64::consts::PI / (2.0 * f64::from(max_x));
        let h = f64::from(max_y * max_y);
        let region_width = region.width as usize;

        for row in 0..region.height as usize {
            for col in 0..region_width {
                let x = f64::from(clamp_coord(region.x + col as i32, self.width));
                let y = f64::from(clamp_coord(region.y + row as i32, self.height));
                let value = F::from_signed_unit((y * y * (c * x * x).cos()) / h);
                write_sample(output, row * region_width + col, value);
            }
        }

        Ok(())
    }
}

impl<F: PointSourceFormat> RandomAccessSource for EyeSourceInner<F> {}

/// Synthetic eye-response pattern matching libvips output format selection.
pub struct EyeSource(EyeSourceKind);

enum EyeSourceKind {
    F32(EyeSourceInner<F32>),
    U8(EyeSourceInner<U8>),
}

impl EyeSource {
    /// Creates an eye-response test pattern with either `f32` or `u8` output.
    pub fn new(width: u32, height: u32, factor: f64, uchar: bool) -> Result<Self, ViprsError> {
        if uchar {
            EyeSourceInner::new(width, height, factor)
                .map(EyeSourceKind::U8)
                .map(Self)
        } else {
            EyeSourceInner::new(width, height, factor)
                .map(EyeSourceKind::F32)
                .map(Self)
        }
    }

    #[must_use]
    /// `width` exposes adapter behavior needed by the surrounding module.
    /// Call it when you need the concrete operation implemented here.
    ///
    /// # Examples
    ///
    /// ```ignore
    /// let _ = viprs_runtime::sources::generators::eye::width;
    /// ```
    pub fn width(&self) -> u32 {
        match &self.0 {
            EyeSourceKind::F32(source) => ImageSource::width(source),
            EyeSourceKind::U8(source) => ImageSource::width(source),
        }
    }

    /// Returns the generated image height.
    #[must_use]
    pub fn height(&self) -> u32 {
        match &self.0 {
            EyeSourceKind::F32(source) => ImageSource::height(source),
            EyeSourceKind::U8(source) => ImageSource::height(source),
        }
    }

    /// Returns the single-band output count used by the eye pattern.
    #[must_use]
    pub const fn bands(&self) -> u32 {
        1
    }

    /// Returns the runtime-selected sample format for this eye source.
    #[must_use]
    pub const fn format(&self) -> BandFormatId {
        match &self.0 {
            EyeSourceKind::F32(_) => BandFormatId::F32,
            EyeSourceKind::U8(_) => BandFormatId::U8,
        }
    }

    /// Reads one region from the generated eye-response pattern into `output`.
    pub fn read_region(&self, region: Region, output: &mut [u8]) -> Result<(), ViprsError> {
        match &self.0 {
            EyeSourceKind::F32(source) => ImageSource::read_region(source, region, output),
            EyeSourceKind::U8(source) => ImageSource::read_region(source, region, output),
        }
    }
}

impl DynImageSource for EyeSource {
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
            EyeSourceKind::F32(source) => ImageSource::demand_hint(source),
            EyeSourceKind::U8(source) => ImageSource::demand_hint(source),
        }
    }

    fn metadata(&self) -> ImageMetadata {
        match &self.0 {
            EyeSourceKind::F32(source) => ImageSource::metadata(source),
            EyeSourceKind::U8(source) => ImageSource::metadata(source),
        }
    }

    fn set_shrink_on_load(&mut self, factor: std::num::NonZeroU8) -> Result<bool, ViprsError> {
        match &mut self.0 {
            EyeSourceKind::F32(source) => ImageSource::set_shrink_on_load(source, factor),
            EyeSourceKind::U8(source) => ImageSource::set_shrink_on_load(source, factor),
        }
    }

    fn set_thumbnail_shrink_on_load(
        &mut self,
        factor: std::num::NonZeroU8,
    ) -> Result<bool, ViprsError> {
        match &mut self.0 {
            EyeSourceKind::F32(source) => ImageSource::set_thumbnail_shrink_on_load(source, factor),
            EyeSourceKind::U8(source) => ImageSource::set_thumbnail_shrink_on_load(source, factor),
        }
    }

    fn read_region(&self, region: Region, output: &mut [u8]) -> Result<(), ViprsError> {
        Self::read_region(self, region, output)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sample_size(source: &EyeSource) -> usize {
        match source.format() {
            BandFormatId::F32 => std::mem::size_of::<f32>(),
            BandFormatId::U8 => std::mem::size_of::<u8>(),
            _ => unreachable!(),
        }
    }

    fn read_all(source: &EyeSource, region: Region) -> Vec<u8> {
        let mut output = vec![0u8; region.pixel_count() * sample_size(source)];
        source.read_region(region, &mut output).unwrap();
        output
    }

    #[test]
    fn dimensions_match_declared_values() {
        let source = EyeSource::new(7, 9, 0.5, false).unwrap();
        assert_eq!(source.width(), 7);
        assert_eq!(source.height(), 9);
        assert_eq!(source.bands(), 1);
        assert_eq!(source.format(), BandFormatId::F32);
    }

    #[test]
    fn read_region_is_deterministic() {
        let source = EyeSource::new(8, 8, 0.5, false).unwrap();
        let region = Region::new(1, 1, 3, 4);
        let first = read_all(&source, region);
        let second = read_all(&source, region);
        assert_eq!(first, second);
    }

    #[test]
    fn factor_zero_produces_vertical_ramp() {
        let source = EyeSource::new(4, 5, 0.0, false).unwrap();
        let output = read_all(&source, Region::new(0, 0, 1, 5));
        let samples: &[f32] = bytemuck::cast_slice(&output);

        assert_eq!(source.format(), BandFormatId::F32);
        assert!((samples[0] - 0.0).abs() < 1e-6);
        assert!((samples[4] - 1.0).abs() < 1e-6);
    }

    #[test]
    fn uchar_mode_emits_u8_samples() {
        let source = EyeSource::new(4, 5, 0.0, true).unwrap();
        let output = read_all(&source, Region::new(0, 0, 1, 5));

        assert_eq!(source.format(), BandFormatId::U8);
        assert_eq!(output, vec![127, 135, 159, 199, 255]);
    }

    #[test]
    fn out_of_range_factor_is_rejected() {
        assert!(matches!(
            EyeSource::new(4, 4, 1.5, false),
            Err(ViprsError::Codec(_))
        ));
    }
}
