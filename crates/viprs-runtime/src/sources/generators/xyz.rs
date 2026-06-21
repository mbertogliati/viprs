//! Xyz image source adapter.
//!
//! This module exposes concrete source implementations or helpers that feed
//! pixels into compiled pipelines.

use crate::{
    domain::{
        error::ViprsError,
        format::U16,
        image::{DemandHint, Region},
    },
    ports::source::{ImageSource, RandomAccessSource},
};

use super::common::{clamp_coord, validate_output_len, write_sample};

/// Synthetic coordinate image with `(x, y, 0)` samples.
pub struct XyzSource {
    /// Output width in pixels.
    pub width: u32,
    /// Output height in pixels.
    pub height: u32,
}

impl XyzSource {
    /// Creates a synthetic coordinate image with the requested geometry.
    #[must_use]
    pub const fn new(width: u32, height: u32) -> Self {
        Self { width, height }
    }
}

impl ImageSource for XyzSource {
    type Format = U16;

    fn width(&self) -> u32 {
        self.width
    }

    fn height(&self) -> u32 {
        self.height
    }

    fn bands(&self) -> u32 {
        3
    }

    fn demand_hint(&self) -> DemandHint {
        DemandHint::Any
    }

    #[inline]
    fn read_region(&self, region: Region, output: &mut [u8]) -> Result<(), ViprsError> {
        validate_output_len(
            region,
            self.bands(),
            std::mem::size_of::<u16>(),
            output,
            self.width,
            self.height,
        )?;

        let region_width = region.width as usize;

        for row in 0..region.height as usize {
            for col in 0..region_width {
                let x =
                    clamp_coord(region.x + col as i32, self.width).min(u32::from(u16::MAX)) as u16;
                let y =
                    clamp_coord(region.y + row as i32, self.height).min(u32::from(u16::MAX)) as u16;
                let pixel_base = (row * region_width + col) * 3;
                write_sample(output, pixel_base, x);
                write_sample(output, pixel_base + 1, y);
                write_sample(output, pixel_base + 2, 0u16);
            }
        }

        Ok(())
    }
}

impl RandomAccessSource for XyzSource {}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn dimensions_match_declared_values() {
        let source = XyzSource::new(7, 9);
        assert_eq!(source.width(), 7);
        assert_eq!(source.height(), 9);
        assert_eq!(source.bands(), 3);
    }

    #[test]
    fn read_region_is_deterministic() {
        let source = XyzSource::new(4, 4);
        let region = Region::new(1, 1, 2, 2);
        let mut first = vec![0u8; region.pixel_count() * 3 * std::mem::size_of::<u16>()];
        let mut second = vec![0u8; first.len()];

        source.read_region(region, &mut first).unwrap();
        source.read_region(region, &mut second).unwrap();

        assert_eq!(first, second);
    }

    #[test]
    fn read_region_fills_expected_output_length() {
        let source = XyzSource::new(3, 2);
        let region = Region::new(0, 0, 3, 2);
        let mut output = vec![0u8; region.pixel_count() * 3 * std::mem::size_of::<u16>()];

        source.read_region(region, &mut output).unwrap();

        assert_eq!(
            output.len(),
            region.pixel_count() * 3 * std::mem::size_of::<u16>()
        );
    }

    #[test]
    fn xyz_pixels_encode_coordinates() {
        let source = XyzSource::new(4, 4);
        let mut output = vec![0u8; 2 * 2 * 3 * std::mem::size_of::<u16>()];
        source
            .read_region(Region::new(1, 2, 2, 2), &mut output)
            .unwrap();
        let samples: &[u16] = bytemuck::cast_slice(&output);

        assert_eq!(samples, &[1, 2, 0, 2, 2, 0, 1, 3, 0, 2, 3, 0]);
    }
}
