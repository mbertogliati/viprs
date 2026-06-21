//! Gaussmat image source adapter.
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

#[derive(Copy, Clone, Debug, PartialEq, Eq)]
/// Precision modes for the synthetic Gaussian kernel coefficients.
pub enum GaussPrecision {
    /// Emit an integer-style kernel suitable for discrete mask workflows.
    Integer,
    /// Emit full floating-point weights.
    Float,
    /// Emit libvips' approximate Gaussian coefficients.
    Approximate,
}

/// Synthetic Gaussian kernel matrix.
pub struct GaussmatSource {
    /// Standard deviation that controls the kernel spread.
    pub sigma: f64,
    /// Smallest coefficient amplitude that should still appear in the kernel.
    pub min_ampl: f64,
    /// Whether the generated kernel should describe a separable 1D pass.
    pub separable: bool,
    /// Coefficient precision mode used for the generated kernel values.
    pub precision: GaussPrecision,
    radius: u32,
}

impl GaussmatSource {
    /// Builds a Gaussian kernel description using libvips-compatible cutoff rules.
    pub fn new(
        sigma: f64,
        min_ampl: f64,
        separable: bool,
        precision: GaussPrecision,
    ) -> Result<Self, ViprsError> {
        if !sigma.is_finite() || sigma <= 0.0 {
            return Err(ViprsError::Codec(format!(
                "GaussmatSource: sigma must be finite and > 0, got {sigma}"
            )));
        }
        if !min_ampl.is_finite() || min_ampl <= 0.0 {
            return Err(ViprsError::Codec(format!(
                "GaussmatSource: min_ampl must be finite and > 0, got {min_ampl}"
            )));
        }

        let sig2 = 2.0 * sigma * sigma;
        let max_x = (8.0 * sigma).floor().clamp(0.0, 5_000.0) as u32;
        let mut first_below = max_x + 1;
        for x in 0..=max_x {
            let value = (-(f64::from(x * x)) / sig2).exp();
            if value < min_ampl {
                first_below = x;
                break;
            }
        }
        let radius = first_below.saturating_sub(1);

        Ok(Self {
            sigma,
            min_ampl,
            separable,
            precision,
            radius,
        })
    }

    #[must_use]
    fn kernel_value(&self, dx: i32, dy: i32) -> f32 {
        let distance = f64::from(dx * dx + dy * dy);
        let sig2 = 2.0 * self.sigma * self.sigma;
        let mut value = (-distance / sig2).exp();
        if !matches!(self.precision, GaussPrecision::Float) {
            value = (20.0 * value).round();
        }
        value as f32
    }
}

impl ImageSource for GaussmatSource {
    type Format = F32;

    fn width(&self) -> u32 {
        2 * self.radius + 1
    }

    fn height(&self) -> u32 {
        if self.separable { 1 } else { self.width() }
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
            std::mem::size_of::<f32>(),
            output,
            self.width(),
            self.height(),
        )?;

        let width = self.width();
        let height = self.height();
        let center_x = (width / 2) as i32;
        let center_y = (height / 2) as i32;
        let region_width = region.width as usize;

        for row in 0..region.height as usize {
            for col in 0..region_width {
                let x = clamp_coord(region.x + col as i32, width) as i32;
                let y = clamp_coord(region.y + row as i32, height) as i32;
                let value = self.kernel_value(x - center_x, y - center_y);
                write_sample(output, row * region_width + col, value);
            }
        }

        Ok(())
    }
}

impl RandomAccessSource for GaussmatSource {}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn dimensions_are_odd_and_match_extent() {
        let source = GaussmatSource::new(1.0, 0.1, false, GaussPrecision::Float).unwrap();
        assert_eq!(source.width() % 2, 1);
        assert_eq!(source.height(), source.width());
    }

    #[test]
    fn separable_mode_has_single_row() {
        let source = GaussmatSource::new(1.0, 0.1, true, GaussPrecision::Float).unwrap();
        assert_eq!(source.height(), 1);
    }

    #[test]
    fn read_region_is_deterministic() {
        let source = GaussmatSource::new(1.5, 0.05, false, GaussPrecision::Float).unwrap();
        let region = Region::new(0, 0, source.width(), source.height());
        let mut first = vec![0u8; region.pixel_count() * std::mem::size_of::<f32>()];
        let mut second = vec![0u8; first.len()];

        source.read_region(region, &mut first).unwrap();
        source.read_region(region, &mut second).unwrap();

        assert_eq!(first, second);
    }

    #[test]
    fn read_region_fills_expected_output_length() {
        let source = GaussmatSource::new(1.0, 0.1, false, GaussPrecision::Float).unwrap();
        let region = Region::new(0, 0, source.width(), source.height());
        let mut output = vec![0u8; region.pixel_count() * std::mem::size_of::<f32>()];

        source.read_region(region, &mut output).unwrap();

        assert_eq!(
            output.len(),
            region.pixel_count() * std::mem::size_of::<f32>()
        );
    }

    #[test]
    fn centre_is_maximum_for_float_precision() {
        let source = GaussmatSource::new(1.0, 0.1, false, GaussPrecision::Float).unwrap();
        let region = Region::new(0, 0, source.width(), source.height());
        let mut output = vec![0u8; region.pixel_count() * std::mem::size_of::<f32>()];
        source.read_region(region, &mut output).unwrap();
        let samples: &[f32] = bytemuck::cast_slice(&output);
        let center = samples[samples.len() / 2];
        let max = samples.iter().copied().fold(f32::MIN, f32::max);

        assert!((center - 1.0).abs() < 1e-6);
        assert!((center - max).abs() < 1e-6);
    }
}
