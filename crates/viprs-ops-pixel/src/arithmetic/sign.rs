use viprs_core::{
    format::{AbsSample, BandFormat},
    image::{DemandHint, Region, Tile, TileMut},
    op::{NodeSpec, Op, PixelLocalOp},
};

/// Sign of each pixel sample.
/// Returns -1/0/1 for signed and float types, and 0/1 for unsigned types.
///
/// # Examples
/// ```ignore
/// use viprs::domain::ops::arithmetic::sign::Sign;
///
/// let op = Sign::new(/* operation parameters */);
/// // Run `op` through a compiled image pipeline.
/// ```
pub struct Sign<F: BandFormat>(std::marker::PhantomData<F>)
where
    F::Sample: AbsSample;

#[allow(dead_code)]
impl<F: BandFormat> Sign<F>
where
    F::Sample: AbsSample,
{
    #[must_use]
    /// Creates a new `Sign`.
    pub const fn new() -> Self {
        Self(std::marker::PhantomData)
    }
}

impl<F: BandFormat> Default for Sign<F>
where
    F::Sample: AbsSample,
{
    fn default() -> Self {
        Self::new()
    }
}

impl<F: BandFormat> Op for Sign<F>
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
            *d = s.s_sign();
        }
    }
}

/// `Sign` is pixel-local: output(x,y) depends only on input(x,y).
/// No neighbourhood access; identity tile geometry. See `PixelLocalOp` for invariants.
impl<F: BandFormat> PixelLocalOp for Sign<F> where F::Sample: viprs_core::format::AbsSample {}

#[cfg(test)]
mod tests {
    use super::*;
    use proptest::prelude::*;
    use viprs_core::{
        format::{F32, U8},
        image::Region,
    };

    fn make_region(w: u32, h: u32) -> Region {
        Region::new(0, 0, w, h)
    }

    #[test]
    fn sign_f32_known_values() {
        let op = Sign::<F32>::new();
        let r = make_region(3, 1);
        let input_data = vec![-5.0f32, -0.0f32, 3.0];
        let mut output_data = vec![0.0f32; 3];
        let input = Tile::<F32>::new(r, 1, &input_data);
        let mut output = TileMut::<F32>::new(r, 1, &mut output_data);
        let mut state = ();
        op.process_region(&mut state, &input, &mut output);
        assert!((output_data[0] - (-1.0)).abs() < f32::EPSILON);
        assert!((output_data[1] - 0.0).abs() < f32::EPSILON);
        assert!((output_data[2] - 1.0).abs() < f32::EPSILON);
    }

    #[test]
    fn sign_metadata_matches_identity_geometry() {
        let op = Sign::<F32>::default();
        let region = make_region(4, 2);
        assert_eq!(op.demand_hint(), viprs_core::image::DemandHint::Any);
        assert_eq!(op.required_input_region(&region), region);
        assert_eq!(op.node_spec(4, 2), viprs_core::op::NodeSpec::identity(4, 2));
    }

    #[test]
    fn sign_u8_is_zero_or_one() {
        let op = Sign::<U8>::new();
        let r = make_region(3, 1);
        let input_data = vec![0u8, 1, 255];
        let mut output_data = vec![0u8; 3];
        let input = Tile::<U8>::new(r, 1, &input_data);
        let mut output = TileMut::<U8>::new(r, 1, &mut output_data);
        let mut state = ();
        op.process_region(&mut state, &input, &mut output);
        assert_eq!(output_data, vec![0u8, 1, 1]);
    }

    proptest! {
        #[test]
        fn sign_f32_result_is_minus_zero_or_plus_one(
            pixels in proptest::collection::vec(-1000.0f32..=1000.0f32, 1..=32)
        ) {
            let len = pixels.len();
            let op = Sign::<F32>::new();
            let r = Region::new(0, 0, len as u32, 1);
            let mut result = vec![0.0f32; len];
            {
                let input = Tile::<F32>::new(r, 1, &pixels);
                let mut output = TileMut::<F32>::new(r, 1, &mut result);
                let mut state = ();
                op.process_region(&mut state, &input, &mut output);
            }
            for v in &result {
                prop_assert!(
                    *v == -1.0 || *v == 0.0 || *v == 1.0,
                    "sign output not in {{-1, 0, 1}}: {}",
                    v
                );
            }
        }

        #[test]
        fn sign_idempotent(
            pixels in proptest::collection::vec(-1000.0f32..=1000.0f32, 1..=32)
        ) {
            let len = pixels.len();
            let op = Sign::<F32>::new();
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
                prop_assert!((a - b).abs() < f32::EPSILON, "sign not idempotent: {} vs {}", a, b);
            }
        }

        #[test]
        fn sign_is_identity_for_sign_outputs(
            pixels in proptest::collection::vec(prop_oneof![Just(-1.0f32), Just(0.0f32), Just(1.0f32)], 1..=32)
        ) {
            let len = pixels.len();
            let op = Sign::<F32>::new();
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
