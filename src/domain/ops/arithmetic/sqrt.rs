use crate::{
    domain::op::{NodeSpec, Op, PixelLocalOp},
    domain::{
        format::{FloatFormat, FloatSample},
        image::{DemandHint, Region, Tile, TileMut},
    },
};

/// Square root of each pixel sample. Only valid for float formats.
/// Negative inputs yield NaN (IEEE behavior).
///
/// # Examples
/// ```ignore
/// use viprs::domain::ops::arithmetic::sqrt::Sqrt;
///
/// let op = Sqrt::new(/* operation parameters */);
/// // Run `op` through a compiled image pipeline.
/// ```
pub struct Sqrt<F: FloatFormat>(std::marker::PhantomData<F>)
where
    F::Sample: FloatSample;

#[allow(dead_code)]
impl<F: FloatFormat> Sqrt<F>
where
    F::Sample: FloatSample,
{
    #[must_use]
    /// Creates a new `Sqrt`.
    pub const fn new() -> Self {
        Self(std::marker::PhantomData)
    }
}

impl<F: FloatFormat> Default for Sqrt<F>
where
    F::Sample: FloatSample,
{
    fn default() -> Self {
        Self::new()
    }
}

impl<F: FloatFormat> Op for Sqrt<F>
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
            *d = s.s_sqrt();
        }
    }
}

/// `Sqrt` is pixel-local: output(x,y) depends only on input(x,y).
/// No neighbourhood access; identity tile geometry. See `PixelLocalOp` for invariants.
impl<F: FloatFormat> PixelLocalOp for Sqrt<F> where F::Sample: FloatSample {}
#[cfg(test)]
mod tests {
    use super::*;
    use crate::domain::{format::F32, image::Region};
    use proptest::prelude::*;

    fn make_region(w: u32, h: u32) -> Region {
        Region::new(0, 0, w, h)
    }

    #[test]
    fn sqrt_f32_known_values() {
        let op = Sqrt::<F32>::new();
        let r = make_region(3, 1);
        // sqrt(0) = 0, sqrt(1) = 1, sqrt(4) = 2
        let input_data = vec![0.0f32, 1.0, 4.0];
        let mut output_data = vec![0.0f32; 3];
        let input = Tile::<F32>::new(r, 1, &input_data);
        let mut output = TileMut::<F32>::new(r, 1, &mut output_data);
        let mut state = ();
        op.process_region(&mut state, &input, &mut output);
        assert!((output_data[0] - 0.0).abs() < 1e-6);
        assert!((output_data[1] - 1.0).abs() < 1e-6);
        assert!((output_data[2] - 2.0).abs() < 1e-6);
    }

    #[test]
    fn sqrt_metadata_matches_identity_geometry() {
        let op = Sqrt::<F32>::default();
        let region = make_region(4, 2);
        assert_eq!(op.demand_hint(), crate::domain::image::DemandHint::Any);
        assert_eq!(op.required_input_region(&region), region);
        assert_eq!(
            op.node_spec(4, 2),
            crate::domain::op::NodeSpec::identity(4, 2)
        );
    }

    proptest! {
        #[test]
        fn sqrt_non_negative_for_positive_inputs(
            pixels in proptest::collection::vec(0.0f32..=10000.0f32, 1..=32)
        ) {
            let len = pixels.len();
            let op = Sqrt::<F32>::new();
            let r = Region::new(0, 0, len as u32, 1);
            let mut output = vec![0.0f32; len];
            {
                let input = Tile::<F32>::new(r, 1, &pixels);
                let mut out = TileMut::<F32>::new(r, 1, &mut output);
                let mut state = ();
                op.process_region(&mut state, &input, &mut out);
            }
            for v in &output {
                prop_assert!(*v >= 0.0, "sqrt of non-negative input is negative: {}", v);
            }
        }
    }
}
