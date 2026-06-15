//! Black image source adapter.
//!
//! This module exposes concrete source implementations or helpers that feed
//! pixels into compiled pipelines.

use crate::{
    domain::{
        error::ViprsError,
        format::U8,
        image::{DemandHint, Region},
    },
    ports::source::{ImageSource, RandomAccessSource},
};

use super::common::validate_output_len;

/// Synthetic source that fills every sample with zero.
pub struct BlackSource {
    width: u32,
    height: u32,
    bands: u32,
}

impl BlackSource {
    /// Creates a constant-black source with the requested geometry and band count.
    #[must_use]
    pub const fn new(width: u32, height: u32, bands: u32) -> Self {
        Self {
            width,
            height,
            bands,
        }
    }
}

impl ImageSource for BlackSource {
    type Format = U8;

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
            std::mem::size_of::<u8>(),
            output,
            self.width,
            self.height,
        )?;
        output.fill(0);
        Ok(())
    }
}

impl RandomAccessSource for BlackSource {}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn dimensions_match_declared_values() {
        let source = BlackSource::new(7, 9, 3);
        assert_eq!(source.width(), 7);
        assert_eq!(source.height(), 9);
        assert_eq!(source.bands(), 3);
    }

    #[test]
    fn read_region_is_deterministic() {
        let source = BlackSource::new(4, 4, 2);
        let region = Region::new(1, 1, 2, 2);
        let mut first = vec![1u8; region.pixel_count() * source.bands() as usize];
        let mut second = vec![2u8; region.pixel_count() * source.bands() as usize];

        source.read_region(region, &mut first).unwrap();
        source.read_region(region, &mut second).unwrap();

        assert_eq!(first, second);
    }

    #[test]
    fn read_region_fills_expected_output_length() {
        let source = BlackSource::new(3, 2, 4);
        let region = Region::new(0, 0, 3, 2);
        let mut output = vec![99u8; region.pixel_count() * source.bands() as usize];

        source.read_region(region, &mut output).unwrap();

        assert_eq!(output.len(), region.pixel_count() * source.bands() as usize);
    }

    #[test]
    fn black_source_returns_all_zero_bytes() {
        let source = BlackSource::new(4, 4, 2);
        let region = Region::new(0, 0, 4, 4);
        let mut output = vec![255u8; region.pixel_count() * source.bands() as usize];

        source.read_region(region, &mut output).unwrap();

        assert!(output.iter().all(|&sample| sample == 0));
    }
}
