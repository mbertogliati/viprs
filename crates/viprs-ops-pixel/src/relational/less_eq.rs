use std::marker::PhantomData;

use crate::relational::CmpSample;

use viprs_core::{
    format::{NumericBand, U8},
    image::{DemandHint, Region, Tile, TileMut},
    op::{Op, PixelLocalOp},
};

/// Element-wise less-than-or-equal comparison against a scalar.
///
/// Each sample is compared to `rhs`. The output is `255` if the sample
/// is less than or equal to `rhs`, `0` otherwise, matching libvips.
pub struct LessEq<F: NumericBand>
where
    F::Sample: CmpSample,
{
    rhs: F::Sample,
    _format: PhantomData<F>,
}

impl<F: NumericBand> LessEq<F>
where
    F::Sample: CmpSample,
{
    /// Creates a new `LessEq`.
    pub const fn new(rhs: F::Sample) -> Self {
        Self {
            rhs,
            _format: PhantomData,
        }
    }
}

impl<F> Op for LessEq<F>
where
    F: NumericBand,
    F::Sample: CmpSample,
{
    type Input = F;
    type Output = U8;
    type State = ();

    fn demand_hint(&self) -> DemandHint {
        DemandHint::ThinStrip
    }

    fn required_input_region(&self, output: &Region) -> Region {
        *output
    }

    fn start(&self) {}

    #[inline]
    fn process_region(&self, _state: &mut (), input: &Tile<F>, output: &mut TileMut<U8>) {
        let rhs = self.rhs;
        for (s, d) in input.data.iter().zip(output.data.iter_mut()) {
            *d = if *s <= rhs { u8::MAX } else { 0 };
        }
    }
}

/// `LessEq` is pixel-local: each output sample depends only on the corresponding
/// input sample. See `PixelLocalOp` for invariants.
impl<F> PixelLocalOp for LessEq<F>
where
    F: NumericBand,
    F::Sample: CmpSample,
{
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::relational::CmpSample;

    use proptest::prelude::*;
    use viprs_core::{
        format::{F32, U8},
        image::{DemandHint, Region},
    };

    fn make_region(w: u32, h: u32) -> Region {
        Region::new(0, 0, w, h)
    }

    #[test]
    fn less_eq_f32_known_values() {
        let op = LessEq::<F32>::new(5.0);
        let r = make_region(3, 1);
        // 3.0 <= 5.0 → true, 5.0 <= 5.0 → true, 7.0 <= 5.0 → false
        let input_data = vec![3.0f32, 5.0, 7.0];
        let mut output_data = vec![0u8; 3];
        let input = Tile::<F32>::new(r, 1, &input_data);
        let mut output = TileMut::<U8>::new(r, 1, &mut output_data);
        let mut state = ();
        op.process_region(&mut state, &input, &mut output);
        assert_eq!(output_data, vec![255u8, 255, 0]);
    }

    #[test]
    fn less_eq_boundary_equal_is_true() {
        let op = LessEq::<F32>::new(5.0);
        let r = make_region(1, 1);
        let input_data = vec![5.0f32];
        let mut output_data = vec![0u8; 1];
        let input = Tile::<F32>::new(r, 1, &input_data);
        let mut output = TileMut::<U8>::new(r, 1, &mut output_data);
        let mut state = ();
        op.process_region(&mut state, &input, &mut output);
        assert_eq!(output_data[0], 255);
    }

    #[test]
    fn less_eq_reports_thin_strip_and_passthrough_region() {
        let op = LessEq::<F32>::new(5.0);
        let region = make_region(4, 2);

        assert_eq!(op.demand_hint(), DemandHint::ThinStrip);
        assert_eq!(op.required_input_region(&region), region);

        op.start();
    }

    #[test]
    fn less_eq_multiband_processes_each_band_independently() {
        let op = LessEq::<F32>::new(5.0);
        let region = make_region(2, 1);
        let input_data = vec![4.0f32, 5.0, 6.0, 2.0, 7.0, 5.0];
        let mut output_data = vec![0u8; input_data.len()];
        let input = Tile::<F32>::new(region, 3, &input_data);
        let mut output = TileMut::<U8>::new(region, 3, &mut output_data);
        let mut state = ();

        op.process_region(&mut state, &input, &mut output);

        assert_eq!(output_data, vec![255u8, 255, 0, 255, 0, 255]);
    }

    proptest! {
        #[test]
        fn less_eq_output_is_always_true_or_false(
            pixels in proptest::collection::vec(0.0f32..=100.0f32, 1..=64),
            rhs in 0.0f32..=100.0f32,
        ) {
            let len = pixels.len();
            let op = LessEq::<F32>::new(rhs);
            let r = Region::new(0, 0, len as u32, 1);
            let mut output_data = vec![0u8; len];
            let input = Tile::<F32>::new(r, 1, &pixels);
            let mut output = TileMut::<U8>::new(r, 1, &mut output_data);
            let mut state = ();
            op.process_region(&mut state, &input, &mut output);
            for v in &output_data {
                prop_assert!(*v == 0 || *v == 255, "output must be 0 or 255, got {}", v);
            }
        }

        #[test]
        fn less_eq_matches_scalar_comparison(
            pixels in proptest::collection::vec(-1000.0f32..=1000.0f32, 1..=64),
            rhs in -1000.0f32..=1000.0f32,
        ) {
            let len = pixels.len();
            let op = LessEq::<F32>::new(rhs);
            let region = Region::new(0, 0, len as u32, 1);
            let mut output_data = vec![0u8; len];
            let input = Tile::<F32>::new(region, 1, &pixels);
            let mut output = TileMut::<U8>::new(region, 1, &mut output_data);
            let mut state = ();

            op.process_region(&mut state, &input, &mut output);

            for (sample, actual) in pixels.iter().zip(&output_data) {
                let expected = if *sample <= rhs { u8::MAX } else { 0 };
                prop_assert_eq!(*actual, expected);
            }
        }
    }
}
