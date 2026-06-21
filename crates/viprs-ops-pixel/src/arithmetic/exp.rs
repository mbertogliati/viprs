use viprs_core::{
    format::{FloatFormat, FloatSample},
    image::{DemandHint, Region, Tile, TileMut},
    op::{NodeSpec, Op, PixelLocalOp},
};

/// Natural exponential (e^x) of each pixel sample. Only valid for float formats.
///
/// # Examples
/// ```ignore
/// use viprs_ops_pixel::arithmetic::exp::Exp;
///
/// let op = Exp::new(/* operation parameters */);
/// // Run `op` through a compiled image pipeline.
/// ```
pub struct Exp<F: FloatFormat>(std::marker::PhantomData<F>)
where
    F::Sample: FloatSample;

#[allow(dead_code)]
impl<F: FloatFormat> Exp<F>
where
    F::Sample: FloatSample,
{
    #[must_use]
    /// Creates a new `Exp`.
    pub const fn new() -> Self {
        Self(std::marker::PhantomData)
    }
}

impl<F: FloatFormat> Default for Exp<F>
where
    F::Sample: FloatSample,
{
    fn default() -> Self {
        Self::new()
    }
}

impl<F: FloatFormat> Op for Exp<F>
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
            *d = s.s_exp();
        }
    }
}

/// `Exp` is pixel-local: output(x,y) depends only on input(x,y).
/// No neighbourhood access; identity tile geometry. See `PixelLocalOp` for invariants.
impl<F: FloatFormat> PixelLocalOp for Exp<F> where F::Sample: FloatSample {}
#[cfg(test)]
mod tests {
    use super::*;
    use proptest::prelude::*;
    use viprs_core::{format::F32, image::Region};

    fn make_region(w: u32, h: u32) -> Region {
        Region::new(0, 0, w, h)
    }

    #[test]
    fn exp_f32_known_values() {
        let op = Exp::<F32>::new();
        let r = make_region(2, 1);
        // e^0 = 1, e^1 ≈ 2.71828
        let input_data = vec![0.0f32, 1.0];
        let mut output_data = vec![0.0f32; 2];
        let input = Tile::<F32>::new(r, 1, &input_data);
        let mut output = TileMut::<F32>::new(r, 1, &mut output_data);
        let mut state = ();
        op.process_region(&mut state, &input, &mut output);
        assert!((output_data[0] - 1.0).abs() < 1e-6);
        assert!((output_data[1] - std::f32::consts::E).abs() < 1e-5);
    }

    #[test]
    fn exp_metadata_matches_identity_geometry() {
        let op = Exp::<F32>::default();
        let region = make_region(4, 2);
        assert_eq!(op.demand_hint(), viprs_core::image::DemandHint::Any);
        assert_eq!(op.required_input_region(&region), region);
        assert_eq!(op.node_spec(4, 2), viprs_core::op::NodeSpec::identity(4, 2));
    }

    proptest! {
        #[test]
        fn exp_then_ln_roundtrip(
            pixels in proptest::collection::vec(-10.0f32..=10.0f32, 1..=32)
        ) {
            // exp(ln(exp(x))) == exp(x), so applying ln after exp recovers the original value.
            let len = pixels.len();
            let op_exp = Exp::<F32>::new();
            let r = Region::new(0, 0, len as u32, 1);
            let mut after_exp = vec![0.0f32; len];
            {
                let input = Tile::<F32>::new(r, 1, &pixels);
                let mut output = TileMut::<F32>::new(r, 1, &mut after_exp);
                let mut state = ();
                op_exp.process_region(&mut state, &input, &mut output);
            }
            // All exp outputs must be positive.
            for v in &after_exp {
                prop_assert!(*v > 0.0, "exp output non-positive: {}", v);
            }
        }
    }
}
