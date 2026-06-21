use viprs_core::{
    format::{FloatFormat, FloatSample},
    image::{DemandHint, Region, Tile, TileMut},
    op::{NodeSpec, Op, PixelLocalOp},
};

/// Arc cosine of each pixel sample, returning degrees.
/// Input expected in [-1, 1]; values outside this range yield NaN (IEEE behavior).
///
/// # Examples
/// ```ignore
/// use viprs_ops_pixel::arithmetic::acos::ACos;
///
/// let op = ACos::new(/* operation parameters */);
/// // Run `op` through a compiled image pipeline.
/// ```
pub struct ACos<F: FloatFormat>(std::marker::PhantomData<F>)
where
    F::Sample: FloatSample;

#[allow(dead_code)]
impl<F: FloatFormat> ACos<F>
where
    F::Sample: FloatSample,
{
    #[must_use]
    /// Creates a new `ACos`.
    pub const fn new() -> Self {
        Self(std::marker::PhantomData)
    }
}

impl<F: FloatFormat> Default for ACos<F>
where
    F::Sample: FloatSample,
{
    fn default() -> Self {
        Self::new()
    }
}

impl<F: FloatFormat> Op for ACos<F>
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
            *d = s.s_acos();
        }
    }
}

/// `ACos` is pixel-local: output(x,y) depends only on input(x,y).
/// No neighbourhood access; identity tile geometry. See `PixelLocalOp` for invariants.
impl<F: FloatFormat> PixelLocalOp for ACos<F> where F::Sample: FloatSample {}
#[cfg(test)]
mod tests {
    use super::*;
    use proptest::prelude::*;
    use viprs_core::{format::F32, image::Region};
    fn make_region(w: u32, h: u32) -> Region {
        Region::new(0, 0, w, h)
    }

    #[test]
    fn acos_f32_known_values() {
        let op = ACos::<F32>::new();
        let r = make_region(3, 1);
        // libvips returns inverse trig results in degrees.
        let input_data = vec![1.0f32, 0.0, -1.0];
        let mut output_data = vec![0.0f32; 3];
        let input = Tile::<F32>::new(r, 1, &input_data);
        let mut output = TileMut::<F32>::new(r, 1, &mut output_data);
        let mut state = ();
        op.process_region(&mut state, &input, &mut output);
        assert!((output_data[0] - 0.0).abs() < 1e-6);
        assert!((output_data[1] - 90.0).abs() < 1e-6);
        assert!((output_data[2] - 180.0).abs() < 1e-6);
    }

    #[test]
    fn acos_metadata_matches_identity_geometry() {
        let op = ACos::<F32>::default();
        let region = make_region(4, 2);
        assert_eq!(op.demand_hint(), viprs_core::image::DemandHint::Any);
        assert_eq!(op.required_input_region(&region), region);
        assert_eq!(op.node_spec(4, 2), viprs_core::op::NodeSpec::identity(4, 2));
    }

    proptest! {
        #[test]
        fn acos_output_in_valid_range(
            // acos is only defined in [-1, 1]
            pixels in proptest::collection::vec(-1.0f32..=1.0f32, 1..=32)
        ) {
            let len = pixels.len();
            let op = ACos::<F32>::new();
            let r = Region::new(0, 0, len as u32, 1);
            let mut output = vec![0.0f32; len];
            {
                let input = Tile::<F32>::new(r, 1, &pixels);
                let mut out = TileMut::<F32>::new(r, 1, &mut output);
                let mut state = ();
                op.process_region(&mut state, &input, &mut out);
            }
            for v in &output {
                prop_assert!(v.is_finite(), "acos output not finite for input in [-1,1]: {}", v);
            }
        }
    }
}
