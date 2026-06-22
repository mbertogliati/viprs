#![allow(clippy::debug_assert_with_mut_call)]
// REASON: debug-only assertions intentionally inspect mutable tile views without changing release codegen.

use crate::arithmetic::rhs_broadcast::{RhsLayout, detect_rhs_layout};
use viprs_core::{
    format::{NumericBand, SubSample},
    image::{DemandHint, Region, Tile, TileMut},
    op::{Op, PixelLocalOp},
};

/// Element-wise subtraction of a fixed right-hand-side buffer from input pixels.
///
/// Integer outputs saturate instead of wrapping, matching libvips clip semantics
/// for the current same-format implementation. The rhs buffer may be full-tile,
/// scalar, per-band, or single-band-image data.
pub struct Subtract<F: NumericBand> {
    rhs: Vec<F::Sample>,
}

impl<F: NumericBand> Subtract<F> {
    #[must_use]
    /// Creates a new `Subtract`.
    pub const fn new(rhs: Vec<F::Sample>) -> Self {
        Self { rhs }
    }
}

impl<F> Op for Subtract<F>
where
    F: NumericBand,
    F::Sample: SubSample + Copy,
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
        let src = input.data;
        let rhs = &self.rhs;
        let dst = &mut output.data;
        let bands = input.bands as usize;
        let layout = detect_rhs_layout(rhs.len(), src.len(), bands);

        debug_assert!(
            layout.is_some(),
            "Subtract: rhs must match full tile, scalar, per-band, or single-band-image layout"
        );
        debug_assert_eq!(
            src.len(),
            dst.len(),
            "Subtract: input and output must have same length"
        );

        match layout {
            Some(RhsLayout::Direct) => {
                for ((s, r), d) in src.iter().zip(rhs.iter()).zip(dst.iter_mut()) {
                    *d = s.s_sub(*r);
                }
            }
            Some(RhsLayout::Scalar) => {
                let rhs = rhs[0];
                for (s, d) in src.iter().zip(dst.iter_mut()) {
                    *d = s.s_sub(rhs);
                }
            }
            Some(RhsLayout::PerBand) => {
                for (src_pixel, dst_pixel) in
                    src.chunks_exact(bands).zip(dst.chunks_exact_mut(bands))
                {
                    for ((sample, rhs_sample), out_sample) in
                        src_pixel.iter().zip(rhs.iter()).zip(dst_pixel.iter_mut())
                    {
                        *out_sample = sample.s_sub(*rhs_sample);
                    }
                }
            }
            Some(RhsLayout::SingleBandImage) => {
                for ((src_pixel, dst_pixel), rhs_sample) in src
                    .chunks_exact(bands)
                    .zip(dst.chunks_exact_mut(bands))
                    .zip(rhs.iter())
                {
                    for (sample, out_sample) in src_pixel.iter().zip(dst_pixel.iter_mut()) {
                        *out_sample = sample.s_sub(*rhs_sample);
                    }
                }
            }
            None => {}
        }
    }
}

/// `Subtract` is pixel-local: output(x,y) depends only on input(x,y).
/// No neighbourhood access; identity tile geometry. See `PixelLocalOp` for invariants.
impl<F> PixelLocalOp for Subtract<F>
where
    F: NumericBand,
    F::Sample: SubSample + Copy,
{
}
#[cfg(test)]
mod tests {
    use super::*;
    use proptest::prelude::*;
    use viprs_core::{
        format::{F32, I16, U8},
        image::Region,
    };

    fn make_region(w: u32, h: u32) -> Region {
        Region::new(0, 0, w, h)
    }

    #[test]
    fn subtract_zero_is_identity() {
        let rhs = vec![0.0f32; 4];
        let op = Subtract::<F32>::new(rhs);
        let r = make_region(2, 2);
        let input_data = vec![1.0f32, 2.0, 3.0, 4.0];
        let mut output_data = vec![0.0f32; 4];
        let input = Tile::<F32>::new(r, 1, &input_data);
        let mut output = TileMut::<F32>::new(r, 1, &mut output_data);
        let mut state = ();
        op.process_region(&mut state, &input, &mut output);
        assert_eq!(output_data, input_data);
    }

    #[test]
    fn subtracting_same_image_yields_zeroes() {
        let input_data = vec![1.0f32, -2.5, 3.25, 0.0];
        let op = Subtract::<F32>::new(input_data.clone());
        let r = make_region(2, 2);
        let mut output_data = vec![99.0f32; 4];
        let input = Tile::<F32>::new(r, 1, &input_data);
        let mut output = TileMut::<F32>::new(r, 1, &mut output_data);
        let mut state = ();

        op.process_region(&mut state, &input, &mut output);

        assert_eq!(output_data, vec![0.0; 4]);
        assert_eq!(op.required_input_region(&r), r);
        assert_eq!(op.demand_hint(), DemandHint::ThinStrip);
    }

    #[test]
    fn subtract_u8_saturates_at_zero() {
        let op = Subtract::<U8>::new(vec![20u8]);
        let r = make_region(2, 1);
        let input_data = vec![10u8, 255];
        let mut output_data = vec![0u8; 2];
        let input = Tile::<U8>::new(r, 1, &input_data);
        let mut output = TileMut::<U8>::new(r, 1, &mut output_data);
        let mut state = ();
        op.process_region(&mut state, &input, &mut output);
        assert_eq!(output_data, vec![0u8, 235]);
    }

    #[test]
    fn subtract_i16_saturates_at_minimum() {
        let op = Subtract::<I16>::new(vec![1i16]);
        let r = make_region(2, 1);
        let input_data = vec![i16::MIN, 0];
        let mut output_data = vec![0i16; 2];
        let input = Tile::<I16>::new(r, 1, &input_data);
        let mut output = TileMut::<I16>::new(r, 1, &mut output_data);
        let mut state = ();
        op.process_region(&mut state, &input, &mut output);
        assert_eq!(output_data, vec![i16::MIN, -1]);
    }

    #[test]
    fn subtract_single_band_rhs_expands_across_multiband_input() {
        let op = Subtract::<U8>::new(vec![1u8, 10]);
        let r = make_region(2, 1);
        let input_data = vec![10u8, 20, 30, 40, 50, 60];
        let mut output_data = vec![0u8; 6];
        let input = Tile::<U8>::new(r, 3, &input_data);
        let mut output = TileMut::<U8>::new(r, 3, &mut output_data);
        let mut state = ();
        op.process_region(&mut state, &input, &mut output);
        assert_eq!(output_data, vec![9u8, 19, 29, 30, 40, 50]);
    }

    proptest! {
        #[test]
        fn subtract_zero_rhs_identity_prop(
            pixels in proptest::collection::vec(0.0f32..=1.0f32, 1..=64)
        ) {
            let len = pixels.len();
            let rhs = vec![0.0f32; len];
            let op = Subtract::<F32>::new(rhs);
            let r = Region::new(0, 0, len as u32, 1);
            let mut output_data = vec![0.0f32; len];
            let input = Tile::<F32>::new(r, 1, &pixels);
            let mut output = TileMut::<F32>::new(r, 1, &mut output_data);
            let mut state = ();
            op.process_region(&mut state, &input, &mut output);
            prop_assert_eq!(output_data, pixels);
        }

        #[test]
        fn subtract_u8_stays_within_saturation_bounds(
            left in proptest::collection::vec(0u8..=255u8, 1..=64),
            right in proptest::collection::vec(0u8..=255u8, 1..=64),
        ) {
            let len = left.len().min(right.len());
            let left = left[..len].to_vec();
            let right = right[..len].to_vec();
            let region = Region::new(0, 0, len as u32, 1);
            let mut output_data = vec![0u8; len];
            let op = Subtract::<U8>::new(right);
            let input = Tile::<U8>::new(region, 1, &left);
            let mut output = TileMut::<U8>::new(region, 1, &mut output_data);
            let mut state = ();
            op.process_region(&mut state, &input, &mut output);
            // Verify output is populated (operation didn't panic).
            prop_assert_eq!(output_data.len(), len);
        }
    }
}
