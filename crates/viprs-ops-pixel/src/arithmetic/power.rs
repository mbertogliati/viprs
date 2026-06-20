use viprs_core::{
    format::{FloatFormat, FloatSample},
    image::{DemandHint, Region, Tile, TileMut},
    op::{NodeSpec, Op, PixelLocalOp},
};

/// Raises each pixel sample to a constant exponent: `out = in ^ exponent`.
/// Matches libvips special cases for zero bases and common exponents.
///
/// # Examples
/// ```ignore
/// use viprs::domain::ops::arithmetic::power::Power;
///
/// let op = Power::new(/* operation parameters */);
/// // Run `op` through a compiled image pipeline.
/// ```
#[allow(dead_code)]
pub struct Power<F: FloatFormat>
where
    F::Sample: FloatSample,
{
    exponent: F::Sample,
}

#[allow(dead_code)]
impl<F: FloatFormat> Power<F>
where
    F::Sample: FloatSample,
{
    /// Creates a new `Power`.
    pub const fn new(exponent: F::Sample) -> Self {
        Self { exponent }
    }
}

impl<F: FloatFormat> Op for Power<F>
where
    F::Sample: FloatSample,
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
            *d = s.s_pow(self.exponent);
        }
    }
}

/// `Power` is pixel-local: output(x,y) depends only on input(x,y).
/// No neighbourhood access; identity tile geometry. See `PixelLocalOp` for invariants.
impl<F: FloatFormat> PixelLocalOp for Power<F> where F::Sample: FloatSample {}
#[cfg(test)]
mod tests {
    use super::*;
    use proptest::prelude::*;
    use viprs_core::{format::F32, image::Region};

    fn make_region(w: u32, h: u32) -> Region {
        Region::new(0, 0, w, h)
    }

    #[test]
    fn power_exponent_one_is_identity() {
        let op = Power::<F32>::new(1.0);
        let r = make_region(3, 1);
        let input_data = vec![2.0f32, 3.0, 4.0];
        let mut output_data = vec![0.0f32; 3];
        let input = Tile::<F32>::new(r, 1, &input_data);
        let mut output = TileMut::<F32>::new(r, 1, &mut output_data);
        let mut state = ();
        op.process_region(&mut state, &input, &mut output);
        for (a, b) in input_data.iter().zip(output_data.iter()) {
            assert!((a - b).abs() < f32::EPSILON);
        }
    }

    #[test]
    fn power_exponent_two_is_square() {
        let op = Power::<F32>::new(2.0);
        let r = make_region(3, 1);
        let input_data = vec![2.0f32, 3.0, 4.0];
        let mut output_data = vec![0.0f32; 3];
        let input = Tile::<F32>::new(r, 1, &input_data);
        let mut output = TileMut::<F32>::new(r, 1, &mut output_data);
        let mut state = ();
        op.process_region(&mut state, &input, &mut output);
        assert!((output_data[0] - 4.0).abs() < f32::EPSILON);
        assert!((output_data[1] - 9.0).abs() < f32::EPSILON);
        assert!((output_data[2] - 16.0).abs() < f32::EPSILON);
    }

    #[test]
    fn power_zero_base_stays_zero_for_negative_exponent() {
        let op = Power::<F32>::new(-1.0);
        let r = make_region(1, 1);
        let input_data = vec![0.0f32];
        let mut output_data = vec![1.0f32; 1];
        let input = Tile::<F32>::new(r, 1, &input_data);
        let mut output = TileMut::<F32>::new(r, 1, &mut output_data);
        let mut state = ();
        op.process_region(&mut state, &input, &mut output);
        assert_eq!(output_data[0], 0.0);
    }

    #[test]
    fn power_metadata_matches_identity_geometry() {
        let op = Power::<F32>::new(2.0);
        let region = make_region(4, 2);
        assert_eq!(op.demand_hint(), viprs_core::image::DemandHint::Any);
        assert_eq!(op.required_input_region(&region), region);
        assert_eq!(op.node_spec(4, 2), viprs_core::op::NodeSpec::identity(4, 2));
    }

    proptest! {
        #[test]
        fn power_exponent_zero_is_one(
            pixels in proptest::collection::vec(1u32..=10000u32, 1..=32)
                .prop_map(|v| v.into_iter().map(|x| x as f32 / 100.0).collect::<Vec<_>>())
        ) {
            let len = pixels.len();
            let op = Power::<F32>::new(0.0);
            let r = Region::new(0, 0, len as u32, 1);
            let mut result = vec![0.0f32; len];
            {
                let input = Tile::<F32>::new(r, 1, &pixels);
                let mut output = TileMut::<F32>::new(r, 1, &mut result);
                let mut state = ();
                op.process_region(&mut state, &input, &mut output);
            }
            for v in &result {
                prop_assert!((v - 1.0).abs() < f32::EPSILON, "x^0 != 1: {}", v);
            }
        }
    }
}
