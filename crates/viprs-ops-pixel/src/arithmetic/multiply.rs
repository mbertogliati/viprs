#![allow(clippy::debug_assert_with_mut_call)]
// REASON: debug-only assertions intentionally inspect mutable tile views without changing release codegen.

use crate::arithmetic::rhs_broadcast::{RhsLayout, detect_rhs_layout};
use viprs_core::{
    format::{MulSample, NumericBand},
    image::{DemandHint, Region, Tile, TileMut},
    op::{Op, PixelLocalOp},
};

/// Element-wise multiplication of input pixels by a fixed right-hand-side buffer.
///
/// Integer outputs saturate instead of wrapping, matching libvips clip semantics
/// for the current same-format implementation. The rhs buffer may be full-tile,
/// scalar, per-band, or single-band-image data.
pub struct Multiply<F: NumericBand> {
    rhs: Vec<F::Sample>,
}

impl<F: NumericBand> Multiply<F> {
    #[must_use]
    /// Creates a new `Multiply`.
    pub const fn new(rhs: Vec<F::Sample>) -> Self {
        Self { rhs }
    }
}

impl<F> Op for Multiply<F>
where
    F: NumericBand,
    F::Sample: MulSample + Copy,
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
            "Multiply: rhs must match full tile, scalar, per-band, or single-band-image layout"
        );
        debug_assert_eq!(
            src.len(),
            dst.len(),
            "Multiply: input and output must have same length"
        );

        match layout {
            Some(RhsLayout::Direct) => {
                for ((s, r), d) in src.iter().zip(rhs.iter()).zip(dst.iter_mut()) {
                    *d = s.s_mul(*r);
                }
            }
            Some(RhsLayout::Scalar) => {
                let rhs = rhs[0];
                for (s, d) in src.iter().zip(dst.iter_mut()) {
                    *d = s.s_mul(rhs);
                }
            }
            Some(RhsLayout::PerBand) => {
                for (src_pixel, dst_pixel) in
                    src.chunks_exact(bands).zip(dst.chunks_exact_mut(bands))
                {
                    for ((sample, rhs_sample), out_sample) in
                        src_pixel.iter().zip(rhs.iter()).zip(dst_pixel.iter_mut())
                    {
                        *out_sample = sample.s_mul(*rhs_sample);
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
                        *out_sample = sample.s_mul(*rhs_sample);
                    }
                }
            }
            None => {}
        }
    }
}

/// `Multiply` is pixel-local: output(x,y) depends only on input(x,y).
/// No neighbourhood access; identity tile geometry. See `PixelLocalOp` for invariants.
impl<F> PixelLocalOp for Multiply<F>
where
    F: NumericBand,
    F::Sample: MulSample + Copy,
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
    fn multiply_by_one_is_identity() {
        let rhs = vec![1.0f32; 4];
        let op = Multiply::<F32>::new(rhs);
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
    fn multiply_u8_saturates_at_upper_boundary() {
        let op = Multiply::<U8>::new(vec![3u8]);
        let r = make_region(2, 1);
        let input_data = vec![100u8, 255];
        let mut output_data = vec![0u8; 2];
        let input = Tile::<U8>::new(r, 1, &input_data);
        let mut output = TileMut::<U8>::new(r, 1, &mut output_data);
        let mut state = ();
        op.process_region(&mut state, &input, &mut output);
        assert_eq!(output_data, vec![255u8, 255]);
    }

    #[test]
    fn multiply_i16_saturates_at_numeric_limits() {
        let op = Multiply::<I16>::new(vec![2i16]);
        let r = make_region(2, 1);
        let input_data = vec![i16::MAX, i16::MIN];
        let mut output_data = vec![0i16; 2];
        let input = Tile::<I16>::new(r, 1, &input_data);
        let mut output = TileMut::<I16>::new(r, 1, &mut output_data);
        let mut state = ();
        op.process_region(&mut state, &input, &mut output);
        assert_eq!(output_data, vec![i16::MAX, i16::MIN]);
    }

    #[test]
    fn multiply_single_band_rhs_expands_across_multiband_input() {
        let op = Multiply::<U8>::new(vec![2u8, 3]);
        let r = make_region(2, 1);
        let input_data = vec![10u8, 20, 30, 5, 6, 7];
        let mut output_data = vec![0u8; 6];
        let input = Tile::<U8>::new(r, 3, &input_data);
        let mut output = TileMut::<U8>::new(r, 3, &mut output_data);
        let mut state = ();
        op.process_region(&mut state, &input, &mut output);
        assert_eq!(output_data, vec![20u8, 40, 60, 15, 18, 21]);
    }

    #[test]
    fn multiply_scalar_and_per_band_rhs_layouts_match_reference() {
        let scalar = Multiply::<U8>::new(vec![2u8]);
        let per_band = Multiply::<U8>::new(vec![2u8, 3, 4]);
        let region = make_region(2, 1);
        let input_data = vec![10u8, 20, 30, 5, 6, 7];

        let mut scalar_out = vec![0u8; 6];
        let input = Tile::<U8>::new(region, 3, &input_data);
        let mut output = TileMut::<U8>::new(region, 3, &mut scalar_out);
        scalar.process_region(&mut (), &input, &mut output);
        assert_eq!(scalar_out, vec![20, 40, 60, 10, 12, 14]);

        let mut per_band_out = vec![0u8; 6];
        let input = Tile::<U8>::new(region, 3, &input_data);
        let mut output = TileMut::<U8>::new(region, 3, &mut per_band_out);
        per_band.process_region(&mut (), &input, &mut output);
        assert_eq!(per_band_out, vec![20, 60, 120, 10, 18, 28]);
    }

    #[test]
    fn multiply_large_f32_rows_cover_dispatch_and_metadata() {
        let src = (0..9).map(|index| index as f32 + 1.0).collect::<Vec<_>>();
        let rhs = (0..9)
            .map(|index| 0.5f32 + index as f32)
            .collect::<Vec<_>>();
        let op = Multiply::<F32>::new(rhs.clone());
        let region = make_region(9, 1);
        let mut output_data = vec![0.0f32; src.len()];
        let input = Tile::<F32>::new(region, 1, &src);
        let mut output = TileMut::<F32>::new(region, 1, &mut output_data);
        let mut state = ();
        op.process_region(&mut state, &input, &mut output);
        let expected: Vec<f32> = src.iter().zip(rhs.iter()).map(|(a, b)| a * b).collect();
        assert_eq!(output_data, expected);
        assert_eq!(op.demand_hint(), viprs_core::image::DemandHint::ThinStrip);
        assert_eq!(op.required_input_region(&region), region);
    }

    proptest! {
        #[test]
        fn multiply_one_rhs_identity_prop(
            pixels in proptest::collection::vec(0.0f32..=1.0f32, 1..=64)
        ) {
            let len = pixels.len();
            let rhs = vec![1.0f32; len];
            let op = Multiply::<F32>::new(rhs);
            let r = Region::new(0, 0, len as u32, 1);
            let mut output_data = vec![0.0f32; len];
            let input = Tile::<F32>::new(r, 1, &pixels);
            let mut output = TileMut::<F32>::new(r, 1, &mut output_data);
            let mut state = ();
            op.process_region(&mut state, &input, &mut output);
            prop_assert_eq!(output_data, pixels);
        }

        #[test]
        fn multiply_u8_commutative_prop(
            left in proptest::collection::vec(0u8..=255u8, 1..=64),
            right in proptest::collection::vec(0u8..=255u8, 1..=64)
        ) {
            let len = left.len().min(right.len());
            let left = left[..len].to_vec();
            let right = right[..len].to_vec();
            let region = Region::new(0, 0, len as u32, 1);

            let mut lhs_mul_rhs = vec![0u8; len];
            {
                let op = Multiply::<U8>::new(right.clone());
                let input = Tile::<U8>::new(region, 1, &left);
                let mut output = TileMut::<U8>::new(region, 1, &mut lhs_mul_rhs);
                let mut state = ();
                op.process_region(&mut state, &input, &mut output);
            }

            let mut rhs_mul_lhs = vec![0u8; len];
            {
                let op = Multiply::<U8>::new(left);
                let input = Tile::<U8>::new(region, 1, &right);
                let mut output = TileMut::<U8>::new(region, 1, &mut rhs_mul_lhs);
                let mut state = ();
                op.process_region(&mut state, &input, &mut output);
            }

            prop_assert_eq!(lhs_mul_rhs, rhs_mul_lhs);
        }
    }
}
