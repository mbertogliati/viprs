use std::{
    f64::consts::{FRAC_PI_2, TAU},
    marker::PhantomData,
};

use viprs_core::{
    format::BandFormat,
    image::{DemandHint, Region, Tile, TileMut},
    op::{Op, PixelLocalOp},
    shared_ops::sample_conv::FromF64,
};

/// Generate a directional cosine test pattern.
///
/// # Examples
/// ```ignore
/// use viprs::domain::ops::create::sines::SinesOp;
///
/// let op = SinesOp { /* operation parameters */ };
/// // Run `op` through a compiled image pipeline.
/// ```
#[derive(Clone, Copy, Debug)]
pub struct SinesOp<F: BandFormat> {
    width: u32,
    height: u32,
    hfreq: f64,
    vfreq: f64,
    uchar: bool,
    _format: PhantomData<F>,
}

impl<F: BandFormat> SinesOp<F> {
    /// Associated constant for default hfreq.
    pub const DEFAULT_HFREQ: f64 = 0.5;
    /// Associated constant for default vfreq.
    pub const DEFAULT_VFREQ: f64 = 0.5;

    #[must_use]
    /// Creates a new `SinesOp`.
    pub const fn new(width: u32, height: u32) -> Self {
        Self::with_frequencies(width, height, Self::DEFAULT_HFREQ, Self::DEFAULT_VFREQ)
    }

    #[must_use]
    /// Returns this value configured with frequencies.
    pub const fn with_frequencies(width: u32, height: u32, hfreq: f64, vfreq: f64) -> Self {
        Self {
            width,
            height,
            hfreq,
            vfreq,
            uchar: false,
            _format: PhantomData,
        }
    }

    #[must_use]
    /// Returns this value configured with uchar.
    pub const fn with_uchar(mut self, uchar: bool) -> Self {
        self.uchar = uchar;
        self
    }
}

impl<F> Op for SinesOp<F>
where
    F: BandFormat,
    F::Sample: FromF64,
{
    type Input = F;
    type Output = F;
    type State = ();

    const OUTPUT_BANDS: Option<usize> = Some(1);

    fn demand_hint(&self) -> DemandHint {
        DemandHint::Any
    }

    fn required_input_region(&self, output: &Region) -> Region {
        *output
    }

    fn start(&self) {}

    #[inline]
    fn process_region(&self, _state: &mut (), _input: &Tile<F>, output: &mut TileMut<F>) {
        debug_assert_eq!(output.bands, 1, "SinesOp output must be single-band");
        debug_assert!(output.region.x >= 0 && output.region.y >= 0);
        debug_assert!(output.region.x as u32 + output.region.width <= self.width);
        debug_assert!(output.region.y as u32 + output.region.height <= self.height);

        let theta = if self.hfreq == 0.0 {
            FRAC_PI_2
        } else {
            (self.vfreq / self.hfreq).atan()
        };
        let factor = self.hfreq.hypot(self.vfreq);
        let c = if self.width == 0 {
            0.0
        } else {
            factor * TAU / f64::from(self.width)
        };
        let costheta = theta.cos();
        let sintheta = theta.sin();
        let region_width = output.region.width as usize;

        for row in 0..output.region.height as usize {
            let y = f64::from(output.region.y as u32 + row as u32);
            for col in 0..region_width {
                let x = f64::from(output.region.x as u32 + col as u32);
                let value = (c * y.mul_add(-sintheta, x * costheta)).cos();
                let sample = if self.uchar {
                    ((value.clamp(-1.0, 1.0) + 1.0) * 0.5) * 255.0
                } else {
                    value
                };
                output.data[row * region_width + col] = F::Sample::from_f64(sample);
            }
        }
    }
}

impl<F> PixelLocalOp for SinesOp<F>
where
    F: BandFormat,
    F::Sample: FromF64,
{
}

#[cfg(test)]
mod tests {
    use super::*;
    use proptest::prelude::*;
    use viprs_core::{
        format::F32,
        image::{Region, Tile, TileMut},
    };

    fn run_op(op: SinesOp<F32>) -> Vec<f32> {
        let region = Region::new(0, 0, op.width, op.height);
        let input_data = vec![0.0f32; region.pixel_count()];
        let mut output_data = vec![0.0f32; region.pixel_count()];
        let input = Tile::<F32>::new(region, 1, &input_data);
        let mut output = TileMut::<F32>::new(region, 1, &mut output_data);
        op.process_region(&mut (), &input, &mut output);
        output_data
    }

    #[test]
    fn defaults_match_libvips_frequency_defaults() {
        let op = SinesOp::<F32>::new(8, 8);
        let samples = run_op(op);
        let expected = run_op(SinesOp::<F32>::with_frequencies(
            8,
            8,
            SinesOp::<F32>::DEFAULT_HFREQ,
            SinesOp::<F32>::DEFAULT_VFREQ,
        ));
        assert_eq!(samples, expected);
    }

    #[test]
    fn zero_frequency_produces_constant_one() {
        let samples = run_op(SinesOp::<F32>::with_frequencies(4, 4, 0.0, 0.0));
        assert!(samples.iter().all(|sample| (*sample - 1.0).abs() < 1e-6));
    }

    #[test]
    fn uchar_mode_scales_signed_range_to_byte_range() {
        let samples = run_op(SinesOp::<F32>::with_frequencies(4, 1, 1.0, 0.0).with_uchar(true));
        assert!(
            samples
                .iter()
                .all(|sample| *sample >= 0.0 && *sample <= 255.0)
        );
    }

    #[test]
    fn horizontal_wave_matches_expected_cosine() {
        let samples = run_op(SinesOp::<F32>::with_frequencies(4, 1, 1.0, 0.0));
        assert!((samples[0] - 1.0).abs() < 1e-6);
        assert!(samples[1].abs() < 1e-6);
        assert!((samples[2] + 1.0).abs() < 1e-6);
        assert!(samples[3].abs() < 1e-6);
    }

    proptest! {
        #[test]
        fn prop_output_stays_within_expected_range(
            width in 1u32..=16,
            height in 1u32..=16,
            hfreq in 0.0f64..=4.0,
            vfreq in 0.0f64..=4.0,
            uchar in any::<bool>(),
        ) {
            let samples = run_op(
                SinesOp::<F32>::with_frequencies(width, height, hfreq, vfreq).with_uchar(uchar)
            );
            for sample in samples {
                if uchar {
                    prop_assert!((0.0..=255.0).contains(&sample));
                } else {
                    prop_assert!((-1.0 - 1e-6..=1.0 + 1e-6).contains(&sample));
                }
            }
        }
    }
}
