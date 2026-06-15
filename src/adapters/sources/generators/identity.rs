//! Identity image source adapter.
//!
//! This module exposes concrete source implementations or helpers that feed
//! pixels into compiled pipelines.

use std::marker::PhantomData;

use crate::{
    domain::{
        error::ViprsError,
        format::{BandFormatId, U8, U16},
        image::{DemandHint, ImageMetadata, Region},
    },
    ports::source::{DynImageSource, ImageSource, RandomAccessSource},
};

use super::common::{IdentitySourceFormat, clamp_coord, validate_output_len, write_sample};

struct IdentitySourceInner<F: IdentitySourceFormat> {
    bands: u32,
    _format: PhantomData<F>,
}

impl<F: IdentitySourceFormat> IdentitySourceInner<F> {
    const fn new(bands: u32) -> Self {
        Self {
            bands,
            _format: PhantomData,
        }
    }
}

impl ImageSource for IdentitySourceInner<U8> {
    type Format = U8;

    fn width(&self) -> u32 {
        256
    }

    fn height(&self) -> u32 {
        1
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
            std::mem::size_of::<u8>(),
            output,
            256,
            1,
        )?;

        let region_width = region.width as usize;
        let bands = self.bands as usize;

        for row in 0..region.height as usize {
            for col in 0..region_width {
                let x = clamp_coord(region.x + col as i32, 256);
                let value = <U8 as IdentitySourceFormat>::from_index(x);
                let pixel_base = (row * region_width + col) * bands;
                for band in 0..bands {
                    write_sample(output, pixel_base + band, value);
                }
            }
        }

        Ok(())
    }
}

impl RandomAccessSource for IdentitySourceInner<U8> {}

impl ImageSource for IdentitySourceInner<U16> {
    type Format = U16;

    fn width(&self) -> u32 {
        65_536
    }

    fn height(&self) -> u32 {
        1
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
            std::mem::size_of::<u16>(),
            output,
            65_536,
            1,
        )?;

        let region_width = region.width as usize;
        let bands = self.bands as usize;

        for row in 0..region.height as usize {
            for col in 0..region_width {
                let x = clamp_coord(region.x + col as i32, 65_536);
                let value = <U16 as IdentitySourceFormat>::from_index(x);
                let pixel_base = (row * region_width + col) * bands;
                for band in 0..bands {
                    write_sample(output, pixel_base + band, value);
                }
            }
        }

        Ok(())
    }
}

impl RandomAccessSource for IdentitySourceInner<U16> {}

/// Synthetic identity LUT image matching libvips output format selection.
pub struct IdentitySource(IdentitySourceKind);

enum IdentitySourceKind {
    U8(IdentitySourceInner<U8>),
    U16(IdentitySourceInner<U16>),
}

impl IdentitySource {
    /// Creates an identity lookup-table image in either `u8` or `u16` form.
    #[must_use]
    pub const fn new(bands: u32, ushort: bool) -> Self {
        if ushort {
            Self(IdentitySourceKind::U16(IdentitySourceInner::new(bands)))
        } else {
            Self(IdentitySourceKind::U8(IdentitySourceInner::new(bands)))
        }
    }

    #[must_use]
    /// `width` exposes adapter behavior needed by the surrounding module.
    /// Call it when you need the concrete operation implemented here.
    ///
    /// # Examples
    ///
    /// ```rust
    /// let _ = viprs::adapters::sources::generators::identity::width;
    /// ```
    pub fn width(&self) -> u32 {
        match &self.0 {
            IdentitySourceKind::U8(source) => ImageSource::width(source),
            IdentitySourceKind::U16(source) => ImageSource::width(source),
        }
    }

    /// Returns the fixed one-row height of the generated lookup table.
    #[must_use]
    pub const fn height(&self) -> u32 {
        1
    }

    /// Returns the configured band count for the lookup table.
    #[must_use]
    pub fn bands(&self) -> u32 {
        match &self.0 {
            IdentitySourceKind::U8(source) => ImageSource::bands(source),
            IdentitySourceKind::U16(source) => ImageSource::bands(source),
        }
    }

    #[must_use]
    /// `format` exposes adapter behavior needed by the surrounding module.
    /// Call it when you need the concrete operation implemented here.
    ///
    /// # Examples
    ///
    /// ```rust
    /// let _ = viprs::adapters::sources::generators::identity::format;
    /// ```
    pub const fn format(&self) -> BandFormatId {
        match &self.0 {
            IdentitySourceKind::U8(_) => BandFormatId::U8,
            IdentitySourceKind::U16(_) => BandFormatId::U16,
        }
    }

    /// Reads one region from the generated identity lookup table into `output`.
    pub fn read_region(&self, region: Region, output: &mut [u8]) -> Result<(), ViprsError> {
        match &self.0 {
            IdentitySourceKind::U8(source) => ImageSource::read_region(source, region, output),
            IdentitySourceKind::U16(source) => ImageSource::read_region(source, region, output),
        }
    }
}

impl DynImageSource for IdentitySource {
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
            IdentitySourceKind::U8(source) => ImageSource::demand_hint(source),
            IdentitySourceKind::U16(source) => ImageSource::demand_hint(source),
        }
    }

    fn metadata(&self) -> ImageMetadata {
        match &self.0 {
            IdentitySourceKind::U8(source) => ImageSource::metadata(source),
            IdentitySourceKind::U16(source) => ImageSource::metadata(source),
        }
    }

    fn set_shrink_on_load(&mut self, factor: std::num::NonZeroU8) -> Result<bool, ViprsError> {
        match &mut self.0 {
            IdentitySourceKind::U8(source) => ImageSource::set_shrink_on_load(source, factor),
            IdentitySourceKind::U16(source) => ImageSource::set_shrink_on_load(source, factor),
        }
    }

    fn set_thumbnail_shrink_on_load(
        &mut self,
        factor: std::num::NonZeroU8,
    ) -> Result<bool, ViprsError> {
        match &mut self.0 {
            IdentitySourceKind::U8(source) => {
                ImageSource::set_thumbnail_shrink_on_load(source, factor)
            }
            IdentitySourceKind::U16(source) => {
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

    fn sample_size(source: &IdentitySource) -> usize {
        match source.format() {
            BandFormatId::U8 => std::mem::size_of::<u8>(),
            BandFormatId::U16 => std::mem::size_of::<u16>(),
            _ => unreachable!(),
        }
    }

    fn read_all(source: &IdentitySource, region: Region) -> Vec<u8> {
        let mut output =
            vec![0u8; region.pixel_count() * source.bands() as usize * sample_size(source)];
        source.read_region(region, &mut output).unwrap();
        output
    }

    #[test]
    fn dimensions_match_declared_values() {
        let source = IdentitySource::new(3, false);
        assert_eq!(source.width(), 256);
        assert_eq!(source.height(), 1);
        assert_eq!(source.bands(), 3);
        assert_eq!(source.format(), BandFormatId::U8);
    }

    #[test]
    fn ushort_mode_uses_65536_entries() {
        let source = IdentitySource::new(1, true);
        assert_eq!(source.width(), 65_536);
        assert_eq!(source.format(), BandFormatId::U16);
    }

    #[test]
    fn read_region_is_deterministic() {
        let source = IdentitySource::new(2, false);
        let region = Region::new(10, 0, 4, 1);
        let first = read_all(&source, region);
        let second = read_all(&source, region);
        assert_eq!(first, second);
    }

    #[test]
    fn identity_pixels_equal_indices_in_byte_mode() {
        let source = IdentitySource::new(2, false);
        let output = read_all(&source, Region::new(0, 0, 4, 1));

        assert_eq!(source.format(), BandFormatId::U8);
        assert_eq!(output, vec![0, 0, 1, 1, 2, 2, 3, 3]);
    }

    #[test]
    fn identity_pixels_equal_indices_in_ushort_mode() {
        let source = IdentitySource::new(1, true);
        let output = read_all(&source, Region::new(65_534, 0, 2, 1));
        let samples: &[u16] = bytemuck::cast_slice(&output);

        assert_eq!(source.format(), BandFormatId::U16);
        assert_eq!(samples, &[65_534, 65_535]);
    }
}
