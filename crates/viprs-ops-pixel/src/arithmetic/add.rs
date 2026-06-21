#![allow(clippy::debug_assert_with_mut_call)]
// REASON: debug-only assertions intentionally inspect mutable tile views without changing release codegen.

use crate::arithmetic::rhs_broadcast::{RhsLayout, detect_rhs_layout};
use viprs_core::{
    format::{AddSample, NumericBand},
    image::{DemandHint, Region, Tile, TileMut},
    op::{Op, PixelLocalOp},
};

/// Element-wise addition of a tile with a fixed right-hand-side buffer.
///
/// The rhs buffer is allocated once at construction time and has the same
/// length as the expected tile pixel buffer, one sample, one value per band,
/// or one value per pixel from a single-band image. Integer outputs saturate
/// instead of wrapping, matching libvips clip semantics for the current
/// same-format implementation. `process_region` performs zero heap allocations.
pub struct Add<F: NumericBand> {
    rhs: Vec<F::Sample>,
}

impl<F: NumericBand> Add<F> {
    /// Construct an Add operation.
    ///
    /// rhs must have exactly width * height * bands elements matching the
    /// tiles that will be passed to `process_region`.
    #[must_use]
    pub const fn new(rhs: Vec<F::Sample>) -> Self {
        Self { rhs }
    }
}

impl<F> Op for Add<F>
where
    F: NumericBand,
    F::Sample: AddSample + Copy,
{
    type Input = F;
    type Output = F;
    type State = ();

    fn demand_hint(&self) -> DemandHint {
        // Addition is pixel-local: no neighbourhood required.
        DemandHint::ThinStrip
    }

    fn required_input_region(&self, output: &Region) -> Region {
        // Pixel-local operation: input region equals output region.
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
            "Add: rhs must match full tile, scalar, per-band, or single-band-image layout"
        );
        debug_assert_eq!(
            src.len(),
            dst.len(),
            "Add: input and output must have same length"
        );

        match layout {
            Some(RhsLayout::Direct) => {
                for ((s, r), d) in src.iter().zip(rhs.iter()).zip(dst.iter_mut()) {
                    *d = s.s_add(*r);
                }
            }
            Some(RhsLayout::Scalar) => {
                let rhs = rhs[0];
                for (s, d) in src.iter().zip(dst.iter_mut()) {
                    *d = s.s_add(rhs);
                }
            }
            Some(RhsLayout::PerBand) => {
                for (src_pixel, dst_pixel) in
                    src.chunks_exact(bands).zip(dst.chunks_exact_mut(bands))
                {
                    for ((sample, rhs_sample), out_sample) in
                        src_pixel.iter().zip(rhs.iter()).zip(dst_pixel.iter_mut())
                    {
                        *out_sample = sample.s_add(*rhs_sample);
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
                        *out_sample = sample.s_add(*rhs_sample);
                    }
                }
            }
            None => {}
        }
    }
}

/// `Add` is pixel-local: it accesses only the current pixel and produces the same
/// geometry. See `PixelLocalOp` in `ports/op.rs` for the invariants.
impl<F> PixelLocalOp for Add<F>
where
    F: NumericBand,
    F::Sample: AddSample + Copy,
{
}

#[cfg(test)]
mod tests {
    use super::*;
    use proptest::prelude::*;
    use viprs_core::format::{F32, I16, U8};
    use viprs_core::image::Region;

    fn make_region(w: u32, h: u32) -> Region {
        Region::new(0, 0, w, h)
    }

    #[test]
    fn add_zero_is_identity() {
        let rhs = vec![0u8; 4];
        let op = Add::<U8>::new(rhs);
        let r = make_region(2, 2);
        let input_data = vec![10u8, 20, 30, 40];
        let mut output_data = vec![0u8; 4];
        let input = Tile::<U8>::new(r, 1, &input_data);
        let mut output = TileMut::<U8>::new(r, 1, &mut output_data);
        let mut state = ();
        op.process_region(&mut state, &input, &mut output);
        assert_eq!(output_data, input_data);
    }

    #[test]
    fn add_constant_increments_each_pixel() {
        let rhs = vec![1u8; 4];
        let op = Add::<U8>::new(rhs);
        let r = make_region(2, 2);
        let input_data = vec![0u8, 1, 2, 3];
        let mut output_data = vec![0u8; 4];
        let input = Tile::<U8>::new(r, 1, &input_data);
        let mut output = TileMut::<U8>::new(r, 1, &mut output_data);
        let mut state = ();
        op.process_region(&mut state, &input, &mut output);
        assert_eq!(output_data, &[1u8, 2, 3, 4]);
    }

    #[test]
    fn add_u8_saturates_at_upper_boundary() {
        let op = Add::<U8>::new(vec![10u8]);
        let r = make_region(2, 1);
        let input_data = vec![250u8, 255];
        let mut output_data = vec![0u8; 2];
        let input = Tile::<U8>::new(r, 1, &input_data);
        let mut output = TileMut::<U8>::new(r, 1, &mut output_data);
        let mut state = ();
        op.process_region(&mut state, &input, &mut output);
        assert_eq!(output_data, vec![255u8, 255]);
    }

    #[test]
    fn add_i16_saturates_at_numeric_limits() {
        let op = Add::<I16>::new(vec![1i16]);
        let r = make_region(2, 1);
        let input_data = vec![i16::MAX, -1];
        let mut output_data = vec![0i16; 2];
        let input = Tile::<I16>::new(r, 1, &input_data);
        let mut output = TileMut::<I16>::new(r, 1, &mut output_data);
        let mut state = ();
        op.process_region(&mut state, &input, &mut output);
        assert_eq!(output_data, vec![i16::MAX, 0]);
    }

    #[test]
    fn add_single_band_rhs_expands_across_multiband_input() {
        let op = Add::<U8>::new(vec![1u8, 10]);
        let r = make_region(2, 1);
        let input_data = vec![10u8, 20, 30, 40, 50, 60];
        let mut output_data = vec![0u8; 6];
        let input = Tile::<U8>::new(r, 3, &input_data);
        let mut output = TileMut::<U8>::new(r, 3, &mut output_data);
        let mut state = ();
        op.process_region(&mut state, &input, &mut output);
        assert_eq!(output_data, vec![11u8, 21, 31, 50, 60, 70]);
    }

    #[test]
    fn add_per_band_rhs_applies_bandwise() {
        let op = Add::<U8>::new(vec![1u8, 2, 3]);
        let r = make_region(2, 1);
        let input_data = vec![10u8, 20, 30, 40, 50, 60];
        let mut output_data = vec![0u8; 6];
        let input = Tile::<U8>::new(r, 3, &input_data);
        let mut output = TileMut::<U8>::new(r, 3, &mut output_data);
        let mut state = ();

        op.process_region(&mut state, &input, &mut output);

        assert_eq!(output_data, vec![11u8, 22, 33, 41, 52, 63]);
    }

    #[test]
    fn add_direct_rhs_matches_elementwise_f32_sum() {
        let op = Add::<F32>::new(vec![0.5f32, -0.5, 1.0, 3.0]);
        let region = make_region(4, 1);
        let input_data = vec![1.0f32, 2.0, -1.0, 4.0];
        let mut output_data = vec![0.0f32; 4];
        let input = Tile::<F32>::new(region, 1, &input_data);
        let mut output = TileMut::<F32>::new(region, 1, &mut output_data);
        let mut state = ();

        op.process_region(&mut state, &input, &mut output);

        assert_eq!(output_data, vec![1.5f32, 1.5, 0.0, 7.0]);
    }

    #[test]
    fn add_reports_pixel_local_geometry_contract() {
        let op = Add::<U8>::new(vec![0u8]);
        let region = Region::new(3, 4, 5, 6);

        assert_eq!(op.demand_hint(), DemandHint::ThinStrip);
        assert_eq!(op.required_input_region(&region), region);
        op.start();
    }

    proptest! {
        #[test]
        fn add_zero_rhs_identity_prop(pixels in proptest::collection::vec(0u8..=255u8, 1..=64)) {
            let len = pixels.len();
            let rhs = vec![0u8; len];
            let op = Add::<U8>::new(rhs);
            let r = Region::new(0, 0, len as u32, 1);
            let mut output_data = vec![0u8; len];
            let input = Tile::<U8>::new(r, 1, &pixels);
            let mut output = TileMut::<U8>::new(r, 1, &mut output_data);
            let mut state = ();
            op.process_region(&mut state, &input, &mut output);
            prop_assert_eq!(output_data, pixels);
        }

        #[test]
        fn add_u8_commutative_prop(
            left in proptest::collection::vec(0u8..=255u8, 1..=64),
            right in proptest::collection::vec(0u8..=255u8, 1..=64),
        ) {
            let len = left.len().min(right.len());
            let left = left[..len].to_vec();
            let right = right[..len].to_vec();
            let region = Region::new(0, 0, len as u32, 1);

            let mut lhs_plus_rhs = vec![0u8; len];
            {
                let op = Add::<U8>::new(right.clone());
                let input = Tile::<U8>::new(region, 1, &left);
                let mut output = TileMut::<U8>::new(region, 1, &mut lhs_plus_rhs);
                let mut state = ();
                op.process_region(&mut state, &input, &mut output);
            }

            let mut rhs_plus_lhs = vec![0u8; len];
            {
                let op = Add::<U8>::new(left.clone());
                let input = Tile::<U8>::new(region, 1, &right);
                let mut output = TileMut::<U8>::new(region, 1, &mut rhs_plus_lhs);
                let mut state = ();
                op.process_region(&mut state, &input, &mut output);
            }

            prop_assert_eq!(lhs_plus_rhs, rhs_plus_lhs);
        }
    }
}
