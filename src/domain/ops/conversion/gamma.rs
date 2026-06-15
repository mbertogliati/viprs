use std::marker::PhantomData;

use crate::domain::{
    format::{BandFormat, BandFormatId},
    image::{DemandHint, Region, Tile, TileMut},
    op::{Op, PixelLocalOp},
    ops::resample::sample_conv::{FromF64, ToF64},
};

const DEFAULT_EXPONENT: f64 = 1.0 / 2.4;

/// Sample conversion support for libvips-style gamma correction.
pub trait GammaSample: ToF64 + FromF64 + Copy {
    /// Associated constant for max value.
    const MAX_VALUE: f64;

    /// Creates this value from gamma f64.
    fn from_gamma_f64(v: f64) -> Self;
}

impl GammaSample for u8 {
    const MAX_VALUE: f64 = Self::MAX as f64;

    #[inline]
    fn from_gamma_f64(v: f64) -> Self {
        v.clamp(f64::from(Self::MIN), f64::from(Self::MAX)).trunc() as Self
    }
}

impl GammaSample for u16 {
    const MAX_VALUE: f64 = Self::MAX as f64;

    #[inline]
    fn from_gamma_f64(v: f64) -> Self {
        v.clamp(f64::from(Self::MIN), f64::from(Self::MAX)).trunc() as Self
    }
}

impl GammaSample for i16 {
    const MAX_VALUE: f64 = Self::MAX as f64;

    #[inline]
    fn from_gamma_f64(v: f64) -> Self {
        v.clamp(f64::from(Self::MIN), f64::from(Self::MAX)).trunc() as Self
    }
}

impl GammaSample for u32 {
    const MAX_VALUE: f64 = Self::MAX as f64;

    #[inline]
    fn from_gamma_f64(v: f64) -> Self {
        v.clamp(f64::from(Self::MIN), f64::from(Self::MAX)).trunc() as Self
    }
}

impl GammaSample for i32 {
    const MAX_VALUE: f64 = Self::MAX as f64;

    #[inline]
    fn from_gamma_f64(v: f64) -> Self {
        v.clamp(f64::from(Self::MIN), f64::from(Self::MAX)).trunc() as Self
    }
}

impl GammaSample for f32 {
    const MAX_VALUE: f64 = 1.0;

    #[inline]
    fn from_gamma_f64(v: f64) -> Self {
        v as Self
    }
}

impl GammaSample for f64 {
    const MAX_VALUE: Self = 1.0;

    #[inline]
    fn from_gamma_f64(v: f64) -> Self {
        v
    }
}

/// Raise samples to `1 / exponent`, normalized around the format maximum.
///
/// This follows libvips `gamma`: integer formats are normalized to their
/// format maximum, float formats are normalized around `1.0`, and the default
/// exponent is `1 / 2.4`.
///
/// # Examples
/// ```ignore
/// use viprs::domain::ops::conversion::gamma::GammaOp;
///
/// let op = GammaOp::new(/* operation parameters */);
/// // Run `op` through a compiled image pipeline.
/// ```
pub struct GammaOp<F: BandFormat> {
    exponent: f64,
    u8_lut: [u8; 256],
    _format: PhantomData<F>,
}

impl<F: BandFormat> GammaOp<F>
where
    F::Sample: GammaSample,
{
    #[must_use]
    /// Creates a new `GammaOp`.
    pub fn new(exponent: f64) -> Self {
        debug_assert!(
            exponent > 0.0 && exponent.is_finite(),
            "GammaOp: exponent must be finite and > 0"
        );
        Self {
            exponent,
            u8_lut: build_u8_lut(exponent),
            _format: PhantomData,
        }
    }

    #[must_use]
    /// Returns or performs exponent.
    pub const fn exponent(&self) -> f64 {
        self.exponent
    }
}

impl<F: BandFormat> Default for GammaOp<F>
where
    F::Sample: GammaSample,
{
    fn default() -> Self {
        Self::new(DEFAULT_EXPONENT)
    }
}

impl<F> Op for GammaOp<F>
where
    F: BandFormat,
    F::Sample: GammaSample,
{
    type Input = F;
    type Output = F;
    type State = ();

    fn demand_hint(&self) -> DemandHint {
        DemandHint::ThinStrip
    }

    fn required_input_region(&self, output: &Region) -> Region {
        *output
    }

    fn start(&self) {}

    #[inline]
    fn process_region(&self, _state: &mut (), input: &Tile<F>, output: &mut TileMut<F>) {
        if self.exponent == 1.0 {
            output.data.copy_from_slice(input.data);
            return;
        }

        if F::ID == BandFormatId::U8 {
            apply_u8_gamma_lut(
                &self.u8_lut,
                bytemuck::cast_slice(input.data),
                bytemuck::cast_slice_mut(output.data),
            );
            return;
        }

        let inv_exponent = 1.0 / self.exponent;
        let max_value = F::Sample::MAX_VALUE;

        for (src, dst) in input.data.iter().zip(output.data.iter_mut()) {
            let normalized = src.to_f64() / max_value;
            let corrected = max_value * normalized.powf(inv_exponent);
            *dst = F::Sample::from_gamma_f64(corrected);
        }
    }
}

impl<F> PixelLocalOp for GammaOp<F>
where
    F: BandFormat,
    F::Sample: GammaSample,
{
}

#[inline]
fn build_u8_lut(exponent: f64) -> [u8; 256] {
    if exponent == 1.0 {
        let mut identity = [0u8; 256];
        for (index, value) in identity.iter_mut().enumerate() {
            *value = index as u8;
        }
        return identity;
    }

    let mut lut = [0u8; 256];
    let inv_exponent = 1.0 / exponent;
    let max_value = f64::from(u8::MAX);

    for (index, value) in lut.iter_mut().enumerate() {
        let normalized = index as f64 / max_value;
        let corrected = max_value * normalized.powf(inv_exponent);
        *value = <u8 as GammaSample>::from_gamma_f64(corrected);
    }

    lut
}

#[inline(never)]
fn apply_u8_gamma_lut(lut: &[u8; 256], input: &[u8], output: &mut [u8]) {
    for (src, dst) in input.iter().zip(output.iter_mut()) {
        *dst = lut[*src as usize];
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::domain::{
        format::{F32, U8, U16},
        image::{Region, Tile, TileMut},
    };
    use proptest::prelude::*;

    fn run_gamma_u8(exponent: f64, input_data: &[u8]) -> Vec<u8> {
        let op = GammaOp::<U8>::new(exponent);
        let region = Region::new(0, 0, input_data.len() as u32, 1);
        let input = Tile::<U8>::new(region, 1, input_data);
        let mut output_data = vec![0u8; input_data.len()];
        let mut output = TileMut::<U8>::new(region, 1, &mut output_data);
        let mut state = op.start();
        op.process_region(&mut state, &input, &mut output);
        output_data
    }

    #[test]
    fn exponent_two_takes_square_root_in_normalized_space() {
        let output = run_gamma_u8(2.0, &[0, 64, 255]);
        assert_eq!(output, vec![0, 127, 255]);
    }

    #[test]
    fn default_exponent_matches_libvips_default() {
        let op = GammaOp::<U8>::default();
        assert!((op.exponent() - (1.0 / 2.4)).abs() < f64::EPSILON);
    }

    #[test]
    fn default_curve_matches_libvips_default() {
        let output = run_gamma_u8(GammaOp::<U8>::default().exponent(), &[0, 64, 255]);
        assert_eq!(output, vec![0, 9, 255]);
    }

    #[test]
    fn default_curve_matches_libvips_u8_midtone() {
        let output = run_gamma_u8(GammaOp::<U8>::default().exponent(), &[128]);
        assert_eq!(output, vec![48]);
    }

    #[test]
    fn u16_boundary_values_stay_in_range() {
        let op = GammaOp::<U16>::default();
        let region = Region::new(0, 0, 2, 1);
        let input_data = [0u16, u16::MAX];
        let input = Tile::<U16>::new(region, 1, &input_data);
        let mut output_data = [1u16; 2];
        let mut output = TileMut::<U16>::new(region, 1, &mut output_data);
        let mut state = op.start();
        op.process_region(&mut state, &input, &mut output);
        assert_eq!(output_data, input_data);
    }

    #[test]
    fn f32_uses_one_as_maximum() {
        let op = GammaOp::<F32>::new(2.0);
        let region = Region::new(0, 0, 2, 1);
        let input_data = [0.25f32, 1.0];
        let input = Tile::<F32>::new(region, 1, &input_data);
        let mut output_data = [0.0f32; 2];
        let mut output = TileMut::<F32>::new(region, 1, &mut output_data);
        let mut state = op.start();
        op.process_region(&mut state, &input, &mut output);
        assert!((output_data[0] - 0.5).abs() < 1e-6);
        assert!((output_data[1] - 1.0).abs() < 1e-6);
    }

    #[test]
    fn default_curve_matches_libvips_u16_midtone() {
        let op = GammaOp::<U16>::default();
        let region = Region::new(0, 0, 1, 1);
        let input_data = [32_768u16];
        let input = Tile::<U16>::new(region, 1, &input_data);
        let mut output_data = [0u16; 1];
        let mut output = TileMut::<U16>::new(region, 1, &mut output_data);
        let mut state = op.start();
        op.process_region(&mut state, &input, &mut output);
        assert_eq!(output_data, [12_417u16]);
    }

    #[test]
    fn default_curve_matches_libvips_f32_midtone() {
        let op = GammaOp::<F32>::default();
        let region = Region::new(0, 0, 1, 1);
        let input_data = [0.5f32];
        let input = Tile::<F32>::new(region, 1, &input_data);
        let mut output_data = [0.0f32; 1];
        let mut output = TileMut::<F32>::new(region, 1, &mut output_data);
        let mut state = op.start();
        op.process_region(&mut state, &input, &mut output);
        assert!((output_data[0] - 0.189_464_57).abs() < 1e-6);
    }

    #[test]
    fn metadata_and_helper_functions_cover_identity_and_scalar_paths() {
        let op = GammaOp::<U8>::new(1.0);
        let region = Region::new(2, 3, 4, 5);
        assert_eq!(op.demand_hint(), DemandHint::ThinStrip);
        assert_eq!(op.required_input_region(&region), region);
        op.start();

        let identity = build_u8_lut(1.0);
        assert_eq!(identity[0], 0);
        assert_eq!(identity[255], 255);

        let mut output = [0u8; 3];
        apply_u8_gamma_lut(&identity, &[0, 127, 255], &mut output);
        assert_eq!(output, [0, 127, 255]);
    }

    #[test]
    fn signed_and_wide_integer_formats_clamp_like_libvips() {
        assert_eq!(<i16 as GammaSample>::from_gamma_f64(-1.2), -1);
        assert_eq!(<i32 as GammaSample>::from_gamma_f64(1234.9), 1234);
        assert_eq!(<u32 as GammaSample>::from_gamma_f64(7.9), 7);
        assert_eq!(<f64 as GammaSample>::from_gamma_f64(0.25), 0.25);
    }

    proptest! {
    #[test]
    fn exponent_one_is_identity(samples in proptest::collection::vec(any::<u8>(), 1..=128)) {
        prop_assert_eq!(run_gamma_u8(1.0, &samples), samples);
    }

    #[test]
    fn output_stays_inside_u8_bounds(samples in proptest::collection::vec(any::<u8>(), 1..=128), exponent in 0.1f64..=10.0) {
        let output = run_gamma_u8(exponent, &samples);
        prop_assert_eq!(output.len(), samples.len());
    }

        #[test]
        fn u8_lut_matches_scalar_curve(exponent in 0.1f64..=10.0, sample in any::<u8>()) {
            let lut = build_u8_lut(exponent);
            let normalized = f64::from(sample) / f64::from(u8::MAX);
            let corrected = f64::from(u8::MAX) * normalized.powf(1.0 / exponent);
            let expected = <u8 as GammaSample>::from_gamma_f64(corrected);
            prop_assert_eq!(lut[sample as usize], expected);
        }
    }
}
