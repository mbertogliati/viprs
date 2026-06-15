use crate::{
    domain::op::{NodeSpec, Op, PixelLocalOp},
    domain::{
        format::{AbsSample, BandFormat},
        image::{DemandHint, Region, Tile, TileMut},
    },
};

/// Pixel-wise absolute value. For unsigned types: identity. For signed and float: abs.
///
/// # Examples
/// ```ignore
/// use viprs::domain::ops::arithmetic::abs::Abs;
///
/// let op = Abs::new(/* operation parameters */);
/// // Run `op` through a compiled image pipeline.
/// ```
pub struct Abs<F: BandFormat>(std::marker::PhantomData<F>)
where
    F::Sample: AbsSample;

#[allow(dead_code)]
impl<F: BandFormat> Abs<F>
where
    F::Sample: AbsSample,
{
    #[must_use]
    /// Creates a new `Abs`.
    pub const fn new() -> Self {
        Self(std::marker::PhantomData)
    }
}

impl<F: BandFormat> Default for Abs<F>
where
    F::Sample: AbsSample,
{
    fn default() -> Self {
        Self::new()
    }
}

impl<F: BandFormat> Op for Abs<F>
where
    F::Sample: AbsSample,
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
    fn node_spec(&self, w: u32, h: u32) -> NodeSpec {
        NodeSpec::identity(w, h)
    }
    fn start(&self) {}
    #[inline]
    fn process_region(&self, (): &mut (), input: &Tile<F>, output: &mut TileMut<F>) {
        for (s, d) in input.data.iter().zip(output.data.iter_mut()) {
            *d = s.s_abs();
        }
    }
}

/// `Abs` is pixel-local: output(x,y) depends only on input(x,y).
/// No neighbourhood access; identity tile geometry. See `PixelLocalOp` for invariants.
impl<F: BandFormat> PixelLocalOp for Abs<F> where F::Sample: AbsSample {}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::domain::{
        format::{F32, I16, I32, U8},
        image::Region,
    };
    use proptest::prelude::*;

    fn make_region(w: u32, h: u32) -> Region {
        Region::new(0, 0, w, h)
    }

    #[test]
    fn abs_f32_known_values() {
        let op = Abs::<F32>::new();
        let r = make_region(3, 1);
        let input_data = vec![-1.0f32, 0.0, 2.5];
        let mut output_data = vec![0.0f32; 3];
        let input = Tile::<F32>::new(r, 1, &input_data);
        let mut output = TileMut::<F32>::new(r, 1, &mut output_data);
        let mut state = ();
        op.process_region(&mut state, &input, &mut output);
        assert!((output_data[0] - 1.0).abs() < f32::EPSILON);
        assert!((output_data[1] - 0.0).abs() < f32::EPSILON);
        assert!((output_data[2] - 2.5).abs() < f32::EPSILON);
    }

    #[test]
    fn abs_metadata_matches_identity_geometry() {
        let op = Abs::<F32>::default();
        let region = make_region(4, 2);
        assert_eq!(op.demand_hint(), crate::domain::image::DemandHint::Any);
        assert_eq!(op.required_input_region(&region), region);
        assert_eq!(
            op.node_spec(4, 2),
            crate::domain::op::NodeSpec::identity(4, 2)
        );
    }

    #[test]
    fn abs_i16_min_saturates_to_max() {
        let op = Abs::<I16>::new();
        let r = make_region(1, 1);
        let input_data = vec![i16::MIN];
        let mut output_data = vec![0i16; 1];
        let input = Tile::<I16>::new(r, 1, &input_data);
        let mut output = TileMut::<I16>::new(r, 1, &mut output_data);
        let mut state = ();
        op.process_region(&mut state, &input, &mut output);
        assert_eq!(output_data[0], i16::MAX);
    }

    #[test]
    fn abs_unsigned_values_are_identity() {
        let op = Abs::<U8>::new();
        let r = make_region(3, 1);
        let input_data = vec![0u8, 127, 255];
        let mut output_data = vec![0u8; 3];
        let input = Tile::<U8>::new(r, 1, &input_data);
        let mut output = TileMut::<U8>::new(r, 1, &mut output_data);
        let mut state = ();
        op.process_region(&mut state, &input, &mut output);
        assert_eq!(output_data, input_data);
    }

    #[test]
    fn abs_i32_min_saturates_and_large_rows_match_reference() {
        let op = Abs::<I32>::new();
        let r = make_region(5, 1);
        let input_data = vec![i32::MIN, -9, -1, 0, 12];
        let mut output_data = vec![0i32; 5];
        let input = Tile::<I32>::new(r, 1, &input_data);
        let mut output = TileMut::<I32>::new(r, 1, &mut output_data);
        let mut state = ();
        op.process_region(&mut state, &input, &mut output);
        assert_eq!(output_data, vec![i32::MAX, 9, 1, 0, 12]);
    }

    proptest! {
        #[test]
        fn abs_f32_nonnegative(
            pixels in proptest::collection::vec(-1000.0f32..=1000.0f32, 1..=64)
        ) {
            let len = pixels.len();
            let op = Abs::<F32>::new();
            let r = Region::new(0, 0, len as u32, 1);
            let mut result = vec![0.0f32; len];
            {
                let input = Tile::<F32>::new(r, 1, &pixels);
                let mut output = TileMut::<F32>::new(r, 1, &mut result);
                let mut state = ();
                op.process_region(&mut state, &input, &mut output);
            }
            for v in &result {
                prop_assert!(*v >= 0.0, "abs output negative: {}", v);
            }
        }

        #[test]
        fn abs_f32_idempotent(
            pixels in proptest::collection::vec(-1000.0f32..=1000.0f32, 1..=64)
        ) {
            let len = pixels.len();
            let op = Abs::<F32>::new();
            let r = Region::new(0, 0, len as u32, 1);
            let mut first = vec![0.0f32; len];
            {
                let input = Tile::<F32>::new(r, 1, &pixels);
                let mut output = TileMut::<F32>::new(r, 1, &mut first);
                let mut state = ();
                op.process_region(&mut state, &input, &mut output);
            }
            let mut second = vec![0.0f32; len];
            {
                let input = Tile::<F32>::new(r, 1, &first);
                let mut output = TileMut::<F32>::new(r, 1, &mut second);
                let mut state = ();
                op.process_region(&mut state, &input, &mut output);
            }
            for (a, b) in first.iter().zip(second.iter()) {
                prop_assert!((a - b).abs() < f32::EPSILON, "abs(abs(x))!=abs(x): {} vs {}", a, b);
            }
        }

        #[test]
        fn abs_f32_is_identity_for_nonnegative_values(
            pixels in proptest::collection::vec(0.0f32..=1000.0f32, 1..=64)
        ) {
            let len = pixels.len();
            let op = Abs::<F32>::new();
            let r = Region::new(0, 0, len as u32, 1);
            let mut result = vec![0.0f32; len];
            {
                let input = Tile::<F32>::new(r, 1, &pixels);
                let mut output = TileMut::<F32>::new(r, 1, &mut result);
                let mut state = ();
                op.process_region(&mut state, &input, &mut output);
            }
            for (expected, actual) in pixels.iter().zip(result.iter()) {
                prop_assert!((expected - actual).abs() < f32::EPSILON);
            }
        }
    }
}
