//! Zone image source adapter.
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

struct ZoneSourceInner<F: PointSourceFormat> {
    width: u32,
    height: u32,
    _format: PhantomData<F>,
}

impl<F: PointSourceFormat> ZoneSourceInner<F> {
    const fn new(width: u32, height: u32) -> Self {
        Self {
            width,
            height,
            _format: PhantomData,
        }
    }
}

impl<F: PointSourceFormat> ImageSource for ZoneSourceInner<F> {
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

        let hwidth = f64::from(self.width / 2);
        let hheight = f64::from(self.height / 2);
        let c = std::f64::consts::PI / f64::from(self.width.max(1));
        let region_width = region.width as usize;

        for row in 0..region.height as usize {
            for col in 0..region_width {
                let x = f64::from(clamp_coord(region.x + col as i32, self.width));
                let y = f64::from(clamp_coord(region.y + row as i32, self.height));
                let dx = x - hwidth;
                let dy = y - hheight;
                let value = F::from_signed_unit((c * (dx * dx + dy * dy)).cos());
                write_sample(output, row * region_width + col, value);
            }
        }

        Ok(())
    }
}

impl<F: PointSourceFormat> RandomAccessSource for ZoneSourceInner<F> {}

/// Synthetic cosine zone plate matching libvips output format selection.
pub struct ZoneSource(ZoneSourceKind);

enum ZoneSourceKind {
    F32(ZoneSourceInner<F32>),
    U8(ZoneSourceInner<U8>),
}

impl ZoneSource {
    /// Creates a zone-plate source in either `u8` or `f32` form.
    #[must_use]
    pub const fn new(width: u32, height: u32, uchar: bool) -> Self {
        if uchar {
            Self(ZoneSourceKind::U8(ZoneSourceInner::new(width, height)))
        } else {
            Self(ZoneSourceKind::F32(ZoneSourceInner::new(width, height)))
        }
    }

    #[must_use]
    /// `width` exposes adapter behavior needed by the surrounding module.
    /// Call it when you need the concrete operation implemented here.
    ///
    /// # Examples
    ///
    /// ```ignore
    /// let _ = viprs::adapters::sources::generators::zone::width;
    /// ```
    pub fn width(&self) -> u32 {
        match &self.0 {
            ZoneSourceKind::F32(source) => ImageSource::width(source),
            ZoneSourceKind::U8(source) => ImageSource::width(source),
        }
    }

    /// Returns the generated image height.
    #[must_use]
    pub fn height(&self) -> u32 {
        match &self.0 {
            ZoneSourceKind::F32(source) => ImageSource::height(source),
            ZoneSourceKind::U8(source) => ImageSource::height(source),
        }
    }

    /// Returns the single-band output count for the zone plate.
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
    /// ```ignore
    /// let _ = viprs::adapters::sources::generators::zone::format;
    /// ```
    pub const fn format(&self) -> BandFormatId {
        match &self.0 {
            ZoneSourceKind::F32(_) => BandFormatId::F32,
            ZoneSourceKind::U8(_) => BandFormatId::U8,
        }
    }

    /// Reads one region from the generated zone-plate pattern into `output`.
    pub fn read_region(&self, region: Region, output: &mut [u8]) -> Result<(), ViprsError> {
        match &self.0 {
            ZoneSourceKind::F32(source) => ImageSource::read_region(source, region, output),
            ZoneSourceKind::U8(source) => ImageSource::read_region(source, region, output),
        }
    }
}

impl DynImageSource for ZoneSource {
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
            ZoneSourceKind::F32(source) => ImageSource::demand_hint(source),
            ZoneSourceKind::U8(source) => ImageSource::demand_hint(source),
        }
    }

    fn metadata(&self) -> ImageMetadata {
        match &self.0 {
            ZoneSourceKind::F32(source) => ImageSource::metadata(source),
            ZoneSourceKind::U8(source) => ImageSource::metadata(source),
        }
    }

    fn set_shrink_on_load(&mut self, factor: std::num::NonZeroU8) -> Result<bool, ViprsError> {
        match &mut self.0 {
            ZoneSourceKind::F32(source) => ImageSource::set_shrink_on_load(source, factor),
            ZoneSourceKind::U8(source) => ImageSource::set_shrink_on_load(source, factor),
        }
    }

    fn set_thumbnail_shrink_on_load(
        &mut self,
        factor: std::num::NonZeroU8,
    ) -> Result<bool, ViprsError> {
        match &mut self.0 {
            ZoneSourceKind::F32(source) => {
                ImageSource::set_thumbnail_shrink_on_load(source, factor)
            }
            ZoneSourceKind::U8(source) => ImageSource::set_thumbnail_shrink_on_load(source, factor),
        }
    }

    fn read_region(&self, region: Region, output: &mut [u8]) -> Result<(), ViprsError> {
        Self::read_region(self, region, output)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sample_size(source: &ZoneSource) -> usize {
        match source.format() {
            BandFormatId::F32 => std::mem::size_of::<f32>(),
            BandFormatId::U8 => std::mem::size_of::<u8>(),
            _ => unreachable!(),
        }
    }

    fn read_all(source: &ZoneSource, region: Region) -> Vec<u8> {
        let mut output = vec![0u8; region.pixel_count() * sample_size(source)];
        source.read_region(region, &mut output).unwrap();
        output
    }

    #[test]
    fn dimensions_match_declared_values() {
        let source = ZoneSource::new(7, 9, false);
        assert_eq!(source.width(), 7);
        assert_eq!(source.height(), 9);
        assert_eq!(source.bands(), 1);
        assert_eq!(source.format(), BandFormatId::F32);
    }

    #[test]
    fn read_region_is_deterministic() {
        let source = ZoneSource::new(9, 9, false);
        let region = Region::new(2, 2, 3, 3);
        let first = read_all(&source, region);
        let second = read_all(&source, region);
        assert_eq!(first, second);
    }

    #[test]
    fn zone_centre_is_one() {
        let source = ZoneSource::new(5, 5, false);
        let output = read_all(&source, Region::new(2, 2, 1, 1));
        let sample = bytemuck::cast_slice::<u8, f32>(&output)[0];

        assert_eq!(source.format(), BandFormatId::F32);
        assert!((sample - 1.0).abs() < 1e-6);
    }

    #[test]
    fn uchar_mode_emits_u8_samples() {
        let source = ZoneSource::new(5, 5, true);
        let output = read_all(&source, Region::new(2, 2, 1, 1));

        assert_eq!(source.format(), BandFormatId::U8);
        assert_eq!(output, vec![255]);
    }
}
