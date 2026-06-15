use crate::{
    domain::op::{NodeSpec, Op, PixelLocalOp},
    domain::{
        format::{FloatFormat, FloatSample},
        image::{DemandHint, Region, Tile, TileMut},
    },
};

/// Natural logarithm of each pixel sample. Only valid for float formats.
/// Matches libvips zero-avoiding semantics: `log(0) == 0`.
///
/// # Examples
/// ```ignore
/// use viprs::domain::ops::arithmetic::log::Log;
///
/// let op = Log::new(/* operation parameters */);
/// // Run `op` through a compiled image pipeline.
/// ```
pub struct Log<F: FloatFormat>(std::marker::PhantomData<F>)
where
    F::Sample: FloatSample;

#[allow(dead_code)]
impl<F: FloatFormat> Log<F>
where
    F::Sample: FloatSample,
{
    #[must_use]
    /// Creates a new `Log`.
    pub const fn new() -> Self {
        Self(std::marker::PhantomData)
    }
}

impl<F: FloatFormat> Default for Log<F>
where
    F::Sample: FloatSample,
{
    fn default() -> Self {
        Self::new()
    }
}

impl<F: FloatFormat> Op for Log<F>
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
            *d = s.s_ln();
        }
    }
}

/// `Log` is pixel-local: output(x,y) depends only on input(x,y).
/// No neighbourhood access; identity tile geometry. See `PixelLocalOp` for invariants.
impl<F: FloatFormat> PixelLocalOp for Log<F> where F::Sample: FloatSample {}
#[cfg(test)]
mod tests {
    use super::*;
    use crate::domain::{format::F32, image::Region};
    use proptest::prelude::*;

    fn make_region(w: u32, h: u32) -> Region {
        Region::new(0, 0, w, h)
    }

    #[test]
    fn log_f32_known_values() {
        let op = Log::<F32>::new();
        let r = make_region(2, 1);
        // ln(1) = 0, ln(e) = 1
        let input_data = vec![1.0f32, std::f32::consts::E];
        let mut output_data = vec![0.0f32; 2];
        let input = Tile::<F32>::new(r, 1, &input_data);
        let mut output = TileMut::<F32>::new(r, 1, &mut output_data);
        let mut state = ();
        op.process_region(&mut state, &input, &mut output);
        assert!((output_data[0] - 0.0).abs() < 1e-6);
        assert!((output_data[1] - 1.0).abs() < 1e-6);
    }

    #[test]
    fn log_of_zero_is_zero() {
        let op = Log::<F32>::new();
        let r = make_region(1, 1);
        let input_data = vec![0.0f32];
        let mut output_data = vec![0.0f32; 1];
        let input = Tile::<F32>::new(r, 1, &input_data);
        let mut output = TileMut::<F32>::new(r, 1, &mut output_data);
        let mut state = ();
        op.process_region(&mut state, &input, &mut output);
        assert_eq!(output_data[0], 0.0);
    }

    #[test]
    fn log_metadata_matches_identity_geometry() {
        let op = Log::<F32>::default();
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
        fn log_exp_roundtrip(
            pixels in proptest::collection::vec(0.001f32..=1000.0f32, 1..=32)
        ) {
            let len = pixels.len();
            let op_log = Log::<F32>::new();
            let op_exp = crate::domain::ops::arithmetic::exp::Exp::<F32>::new();
            let r = Region::new(0, 0, len as u32, 1);
            let mut after_log = vec![0.0f32; len];
            {
                let input = Tile::<F32>::new(r, 1, &pixels);
                let mut output = TileMut::<F32>::new(r, 1, &mut after_log);
                let mut state = ();
                op_log.process_region(&mut state, &input, &mut output);
            }
            let mut after_exp = vec![0.0f32; len];
            {
                let input = Tile::<F32>::new(r, 1, &after_log);
                let mut output = TileMut::<F32>::new(r, 1, &mut after_exp);
                let mut state = ();
                op_exp.process_region(&mut state, &input, &mut output);
            }
            for (a, b) in pixels.iter().zip(after_exp.iter()) {
                prop_assert!((a - b).abs() / a.max(1.0) < 1e-5, "ln(exp(x)) roundtrip: {} vs {}", a, b);
            }
        }
    }
}
