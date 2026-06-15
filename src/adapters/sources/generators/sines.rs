//! Sines image source adapter.
//!
//! This module exposes concrete source implementations or helpers that feed
//! pixels into compiled pipelines.

use crate::{
    domain::{
        error::ViprsError,
        format::{BandFormatId, F32, U8},
        image::{DemandHint, ImageMetadata, Region},
    },
    ports::source::{DynImageSource, ImageSource, RandomAccessSource},
};

use super::common::{PointSourceFormat, clamp_coord, validate_output_len, write_sample};

struct SinesSourceInner<F: PointSourceFormat> {
    width: u32,
    height: u32,
    hfreq: f64,
    vfreq: f64,
    _format: std::marker::PhantomData<F>,
}

impl<F: PointSourceFormat> SinesSourceInner<F> {
    const fn new(width: u32, height: u32, hfreq: f64, vfreq: f64) -> Self {
        Self {
            width,
            height,
            hfreq,
            vfreq,
            _format: std::marker::PhantomData,
        }
    }
}

impl<F: PointSourceFormat> ImageSource for SinesSourceInner<F> {
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

        let theta = if self.hfreq == 0.0 {
            std::f64::consts::FRAC_PI_2
        } else {
            (self.vfreq / self.hfreq).atan()
        };
        let factor = self.hfreq.hypot(self.vfreq);
        let c = if self.width == 0 {
            0.0
        } else {
            factor * std::f64::consts::TAU / f64::from(self.width)
        };
        let costheta = theta.cos();
        let sintheta = theta.sin();
        let region_width = region.width as usize;

        for row in 0..region.height as usize {
            for col in 0..region_width {
                let x = f64::from(clamp_coord(region.x + col as i32, self.width));
                let y = f64::from(clamp_coord(region.y + row as i32, self.height));
                let value = F::from_signed_unit((c * y.mul_add(-sintheta, x * costheta)).cos());
                write_sample(output, row * region_width + col, value);
            }
        }

        Ok(())
    }
}

impl<F: PointSourceFormat> RandomAccessSource for SinesSourceInner<F> {}

/// Synthetic 2D sine-wave pattern matching libvips output format selection.
pub struct SinesSource(SinesSourceKind);

enum SinesSourceKind {
    F32(SinesSourceInner<F32>),
    U8(SinesSourceInner<U8>),
}

impl SinesSource {
    /// Default horizontal sine frequency used by [`SinesSource::new`].
    pub const DEFAULT_HFREQ: f64 = 0.5;
    /// Default vertical sine frequency used by [`SinesSource::new`].
    pub const DEFAULT_VFREQ: f64 = 0.5;

    /// Creates a sine-wave source using the default horizontal and vertical frequencies.
    #[must_use]
    pub const fn new(width: u32, height: u32, uchar: bool) -> Self {
        Self::with_frequencies(
            width,
            height,
            Self::DEFAULT_HFREQ,
            Self::DEFAULT_VFREQ,
            uchar,
        )
    }

    /// Creates a sine-wave source with explicit horizontal and vertical frequencies.
    #[must_use]
    pub const fn with_frequencies(
        width: u32,
        height: u32,
        hfreq: f64,
        vfreq: f64,
        uchar: bool,
    ) -> Self {
        if uchar {
            Self(SinesSourceKind::U8(SinesSourceInner::new(
                width, height, hfreq, vfreq,
            )))
        } else {
            Self(SinesSourceKind::F32(SinesSourceInner::new(
                width, height, hfreq, vfreq,
            )))
        }
    }

    /// Returns the generated image width.
    #[must_use]
    pub fn width(&self) -> u32 {
        match &self.0 {
            SinesSourceKind::F32(source) => ImageSource::width(source),
            SinesSourceKind::U8(source) => ImageSource::width(source),
        }
    }

    /// Returns the generated image height.
    #[must_use]
    pub fn height(&self) -> u32 {
        match &self.0 {
            SinesSourceKind::F32(source) => ImageSource::height(source),
            SinesSourceKind::U8(source) => ImageSource::height(source),
        }
    }

    /// Returns the single-band output count for the sine pattern.
    #[must_use]
    pub const fn bands(&self) -> u32 {
        1
    }

    #[must_use]
    /// `format` exposes adapter behavior needed by the surrounding module.
    /// Call it when you need the concrete operation implemented here.
    ///
    /// # Examples
    ///
    /// ```rust
    /// let _ = viprs::adapters::sources::generators::sines::format;
    /// ```
    pub const fn format(&self) -> BandFormatId {
        match &self.0 {
            SinesSourceKind::F32(_) => BandFormatId::F32,
            SinesSourceKind::U8(_) => BandFormatId::U8,
        }
    }

    /// Reads one region from the generated sine-wave pattern into `output`.
    pub fn read_region(&self, region: Region, output: &mut [u8]) -> Result<(), ViprsError> {
        match &self.0 {
            SinesSourceKind::F32(source) => ImageSource::read_region(source, region, output),
            SinesSourceKind::U8(source) => ImageSource::read_region(source, region, output),
        }
    }
}

impl DynImageSource for SinesSource {
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
            SinesSourceKind::F32(source) => ImageSource::demand_hint(source),
            SinesSourceKind::U8(source) => ImageSource::demand_hint(source),
        }
    }

    fn metadata(&self) -> ImageMetadata {
        match &self.0 {
            SinesSourceKind::F32(source) => ImageSource::metadata(source),
            SinesSourceKind::U8(source) => ImageSource::metadata(source),
        }
    }

    fn set_shrink_on_load(&mut self, factor: std::num::NonZeroU8) -> Result<bool, ViprsError> {
        match &mut self.0 {
            SinesSourceKind::F32(source) => ImageSource::set_shrink_on_load(source, factor),
            SinesSourceKind::U8(source) => ImageSource::set_shrink_on_load(source, factor),
        }
    }

    fn set_thumbnail_shrink_on_load(
        &mut self,
        factor: std::num::NonZeroU8,
    ) -> Result<bool, ViprsError> {
        match &mut self.0 {
            SinesSourceKind::F32(source) => {
                ImageSource::set_thumbnail_shrink_on_load(source, factor)
            }
            SinesSourceKind::U8(source) => {
                ImageSource::set_thumbnail_shrink_on_load(source, factor)
            }
        }
    }

    fn read_region(&self, region: Region, output: &mut [u8]) -> Result<(), ViprsError> {
        Self::read_region(self, region, output)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sample_size(source: &SinesSource) -> usize {
        match source.format() {
            BandFormatId::F32 => std::mem::size_of::<f32>(),
            BandFormatId::U8 => std::mem::size_of::<u8>(),
            _ => unreachable!(),
        }
    }

    fn read_all(source: &SinesSource, region: Region) -> Vec<u8> {
        let mut output = vec![0u8; region.pixel_count() * sample_size(source)];
        source.read_region(region, &mut output).unwrap();
        output
    }

    #[test]
    fn dimensions_match_declared_values() {
        let source = SinesSource::with_frequencies(7, 9, 0.5, 0.25, false);
        assert_eq!(source.width(), 7);
        assert_eq!(source.height(), 9);
        assert_eq!(source.bands(), 1);
        assert_eq!(source.format(), BandFormatId::F32);
    }

    #[test]
    fn read_region_is_deterministic() {
        let source = SinesSource::with_frequencies(16, 16, 0.5, 0.25, false);
        let region = Region::new(3, 4, 5, 6);
        let first = read_all(&source, region);
        let second = read_all(&source, region);
        assert_eq!(first, second);
    }

    #[test]
    fn default_frequencies_match_libvips_defaults() {
        let default_source = SinesSource::new(8, 8, false);
        let explicit_source = SinesSource::with_frequencies(8, 8, 0.5, 0.5, false);
        let region = Region::new(0, 0, 8, 8);
        assert_eq!(
            read_all(&default_source, region),
            read_all(&explicit_source, region)
        );
    }

    #[test]
    fn zero_frequency_produces_constant_one() {
        let source = SinesSource::with_frequencies(4, 4, 0.0, 0.0, false);
        let output = read_all(&source, Region::new(0, 0, 4, 4));
        let samples: &[f32] = bytemuck::cast_slice(&output);

        assert_eq!(source.format(), BandFormatId::F32);
        assert!(samples.iter().all(|sample| (*sample - 1.0).abs() < 1e-6));
    }

    #[test]
    fn uchar_mode_emits_u8_samples() {
        let source = SinesSource::with_frequencies(4, 1, 1.0, 0.0, true);
        let output = read_all(&source, Region::new(0, 0, 4, 1));

        assert_eq!(source.format(), BandFormatId::U8);
        assert_eq!(output, vec![255, 127, 0, 127]);
    }
}
