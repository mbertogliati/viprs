//! Gaussnoise image source adapter.
//!
//! This module exposes concrete source implementations or helpers that feed
//! pixels into compiled pipelines.

use crate::{
    domain::{
        error::ViprsError,
        format::F32,
        image::{DemandHint, Region},
    },
    ports::source::{ImageSource, RandomAccessSource},
};

use super::common::{clamp_coord, validate_output_len, write_sample};

/// Synthetic Gaussian noise image.
pub struct GaussnoiseSource {
    /// Output width in pixels.
    pub width: u32,
    /// Output height in pixels.
    pub height: u32,
    /// Number of bands to synthesize per pixel.
    pub bands: u32,
    /// Mean of the generated Gaussian distribution.
    pub mean: f64,
    /// Standard deviation of the generated Gaussian distribution.
    pub sigma: f64,
    /// Seed mixed into the deterministic libvips-style hash generator.
    pub seed: u64,
}

impl GaussnoiseSource {
    /// Creates a deterministic Gaussian noise source.
    #[must_use]
    pub const fn new(
        width: u32,
        height: u32,
        bands: u32,
        mean: f64,
        sigma: f64,
        seed: u64,
    ) -> Self {
        Self {
            width,
            height,
            bands,
            mean,
            sigma,
            seed,
        }
    }
}

#[inline(always)]
fn vips_random_add(mut hash: u32, value: i32) -> u32 {
    for shift in [0, 8, 16, 24] {
        hash = (hash ^ ((value >> shift) as u32 & 0xff)).wrapping_mul(16_777_619);
    }
    hash
}

#[inline(always)]
fn vips_random(seed: u32) -> u32 {
    vips_random_add(2_166_136_261, seed as i32)
}

#[inline(always)]
fn gaussian_sample(seed: u64, x: i32, y: i32, band: i32, mean: f64, sigma: f64) -> f32 {
    let mut mixed = seed as u32;
    mixed = vips_random_add(mixed, x);
    mixed = vips_random_add(mixed, y);
    mixed = vips_random_add(mixed, band);

    let mut sum = 0.0;
    for _ in 0..12 {
        mixed = vips_random(mixed);
        sum += f64::from(mixed) / f64::from(u32::MAX);
    }

    (sum - 6.0).mul_add(sigma, mean) as f32
}

impl ImageSource for GaussnoiseSource {
    type Format = F32;

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
            std::mem::size_of::<f32>(),
            output,
            self.width,
            self.height,
        )?;

        let region_width = region.width as usize;
        let bands = self.bands as usize;

        for row in 0..region.height as usize {
            for col in 0..region_width {
                let x = clamp_coord(region.x + col as i32, self.width) as i32;
                let y = clamp_coord(region.y + row as i32, self.height) as i32;
                let pixel_base = (row * region_width + col) * bands;

                for band in 0..bands {
                    let sample =
                        gaussian_sample(self.seed, x, y, band as i32, self.mean, self.sigma);
                    write_sample(output, pixel_base + band, sample);
                }
            }
        }

        Ok(())
    }
}

impl RandomAccessSource for GaussnoiseSource {}

#[cfg(test)]
mod tests {
    use super::*;
    use proptest::prelude::*;

    #[test]
    fn dimensions_match_declared_values() {
        let source = GaussnoiseSource::new(7, 9, 3, 0.0, 1.0, 42);
        assert_eq!(source.width(), 7);
        assert_eq!(source.height(), 9);
        assert_eq!(source.bands(), 3);
    }

    #[test]
    fn read_region_fills_expected_output_length() {
        let source = GaussnoiseSource::new(3, 2, 2, 0.0, 1.0, 42);
        let region = Region::new(0, 0, 3, 2);
        let mut output =
            vec![0u8; region.pixel_count() * source.bands() as usize * std::mem::size_of::<f32>()];

        source.read_region(region, &mut output).unwrap();

        assert_eq!(
            output.len(),
            region.pixel_count() * source.bands() as usize * std::mem::size_of::<f32>()
        );
    }

    #[test]
    fn repeated_reads_match_exactly() {
        let source = GaussnoiseSource::new(16, 16, 2, 10.0, 3.0, 1234);
        let region = Region::new(2, 3, 5, 7);
        let mut first =
            vec![0u8; region.pixel_count() * source.bands() as usize * std::mem::size_of::<f32>()];
        let mut second = vec![0u8; first.len()];

        source.read_region(region, &mut first).unwrap();
        source.read_region(region, &mut second).unwrap();

        assert_eq!(first, second);
    }

    proptest! {
        #[test]
        fn determinism_holds_across_random_seeds(seed in any::<u64>()) {
            let source = GaussnoiseSource::new(8, 8, 1, 0.0, 1.0, seed);
            let region = Region::new(0, 0, 8, 8);
            let mut first = vec![0u8; region.pixel_count() * std::mem::size_of::<f32>()];
            let mut second = vec![0u8; first.len()];

            source.read_region(region, &mut first).unwrap();
            source.read_region(region, &mut second).unwrap();

            prop_assert_eq!(first, second);
        }
    }
}
