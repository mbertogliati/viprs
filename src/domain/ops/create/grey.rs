use std::marker::PhantomData;

use crate::domain::{
    format::BandFormat,
    image::{DemandHint, Region, Tile, TileMut},
    op::{Op, PixelLocalOp},
    ops::resample::sample_conv::FromF64,
};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
/// Enumerates the available grey axis values.
pub enum GreyAxis {
    /// Uses the `Horizontal` variant of `GreyAxis`.
    Horizontal,
    /// Uses the `Vertical` variant of `GreyAxis`.
    Vertical,
}

/// Generate a unit grey ramp along one image axis.
///
/// # Examples
/// ```ignore
/// use viprs::domain::ops::create::grey::GreyOp;
///
/// let op = GreyOp { /* operation parameters */ };
/// // Run `op` through a compiled image pipeline.
/// ```
#[derive(Clone, Copy, Debug)]
pub struct GreyOp<F: BandFormat> {
    width: u32,
    height: u32,
    axis: GreyAxis,
    uchar: bool,
    _format: PhantomData<F>,
}

impl<F: BandFormat> GreyOp<F> {
    #[must_use]
    /// Creates a new `GreyOp`.
    pub const fn new(width: u32, height: u32) -> Self {
        Self {
            width,
            height,
            axis: GreyAxis::Horizontal,
            uchar: false,
            _format: PhantomData,
        }
    }

    #[must_use]
    /// Returns this value configured with axis.
    pub const fn with_axis(mut self, axis: GreyAxis) -> Self {
        self.axis = axis;
        self
    }

    #[must_use]
    /// Returns this value configured with uchar.
    pub const fn with_uchar(mut self, uchar: bool) -> Self {
        self.uchar = uchar;
        self
    }
}

impl<F> Op for GreyOp<F>
where
    F: BandFormat,
    F::Sample: FromF64,
{
    type Input = F;
    type Output = F;
    type State = ();

    fn demand_hint(&self) -> DemandHint {
        DemandHint::Any
    }

    fn required_input_region(&self, output: &Region) -> Region {
        *output
    }

    fn start(&self) {}

    #[inline]
    fn process_region(&self, _state: &mut (), _input: &Tile<F>, output: &mut TileMut<F>) {
        debug_assert!(output.region.x >= 0 && output.region.y >= 0);
        debug_assert!(output.region.x as u32 + output.region.width <= self.width);
        debug_assert!(output.region.y as u32 + output.region.height <= self.height);

        let region_width = output.region.width as usize;
        let bands = output.bands as usize;
        let denom = match self.axis {
            GreyAxis::Horizontal => self.width.saturating_sub(1),
            GreyAxis::Vertical => self.height.saturating_sub(1),
        };

        for row in 0..output.region.height as usize {
            let y = output.region.y as u32 + row as u32;
            for col in 0..region_width {
                let x = output.region.x as u32 + col as u32;
                let normalized = if denom == 0 {
                    0.0
                } else {
                    let coord = match self.axis {
                        GreyAxis::Horizontal => x,
                        GreyAxis::Vertical => y,
                    };
                    f64::from(coord) / f64::from(denom)
                };
                let value = if self.uchar {
                    normalized.clamp(0.0, 1.0) * 255.0
                } else {
                    normalized
                };
                let sample = F::Sample::from_f64(value);
                let pixel_base = (row * region_width + col) * bands;
                for band in 0..bands {
                    output.data[pixel_base + band] = sample;
                }
            }
        }
    }
}

impl<F> PixelLocalOp for GreyOp<F>
where
    F: BandFormat,
    F::Sample: FromF64,
{
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::domain::{
        format::F32,
        image::{Region, Tile, TileMut},
    };
    use proptest::prelude::*;

    fn run_op(op: GreyOp<F32>, width: u32, height: u32, bands: u32) -> Vec<f32> {
        let region = Region::new(0, 0, width, height);
        let input_len = width as usize * height as usize * bands as usize;
        let input_data = vec![0.0f32; input_len];
        let mut output_data = vec![-1.0f32; input_len];
        let input = Tile::<F32>::new(region, bands, &input_data);
        let mut output = TileMut::<F32>::new(region, bands, &mut output_data);
        op.process_region(&mut (), &input, &mut output);
        output_data
    }

    #[test]
    fn horizontal_ramp_spans_zero_to_one() {
        let samples = run_op(GreyOp::<F32>::new(4, 2), 4, 2, 1);
        assert_eq!(samples[0], 0.0);
        assert!((samples[1] - (1.0 / 3.0)).abs() < 1e-6);
        assert!((samples[2] - (2.0 / 3.0)).abs() < 1e-6);
        assert_eq!(samples[3], 1.0);
        assert_eq!(samples[4], 0.0);
        assert_eq!(samples[7], 1.0);
    }

    #[test]
    fn vertical_ramp_uses_row_coordinates() {
        let samples = run_op(
            GreyOp::<F32>::new(3, 4).with_axis(GreyAxis::Vertical),
            3,
            4,
            1,
        );
        assert_eq!(&samples[0..3], &[0.0, 0.0, 0.0]);
        assert!((samples[3] - (1.0 / 3.0)).abs() < 1e-6);
        assert!((samples[6] - (2.0 / 3.0)).abs() < 1e-6);
        assert_eq!(&samples[9..12], &[1.0, 1.0, 1.0]);
    }

    #[test]
    fn uchar_mode_scales_endpoints() {
        let samples = run_op(GreyOp::<F32>::new(2, 1).with_uchar(true), 2, 1, 1);
        assert_eq!(samples, vec![0.0, 255.0]);
    }

    proptest! {
        #[test]
        fn prop_horizontal_ramp_matches_expected_formula(
            width in 1u32..=16,
            height in 1u32..=8,
            bands in 1u32..=4,
            uchar in any::<bool>(),
        ) {
            let op = GreyOp::<F32>::new(width, height).with_uchar(uchar);
            let region = Region::new(0, 0, width, height);
            let len = width as usize * height as usize * bands as usize;
            let input_data = vec![0.0f32; len];
            let mut output_data = vec![0.0f32; len];
            let input = Tile::<F32>::new(region, bands, &input_data);
            let mut output = TileMut::<F32>::new(region, bands, &mut output_data);

            op.process_region(&mut (), &input, &mut output);

            let denom = width.saturating_sub(1);
            for y in 0..height as usize {
                for x in 0..width as usize {
                    let expected = if denom == 0 {
                        0.0
                    } else {
                        x as f32 / denom as f32
                    };
                    let expected = if uchar { ((expected as f64) * 255.0) as f32 } else { expected };
                    let pixel_base = (y * width as usize + x) * bands as usize;
                    for band in 0..bands as usize {
                        prop_assert!((output_data[pixel_base + band] - expected).abs() < 1e-4);
                    }
                }
            }
        }
    }
}
