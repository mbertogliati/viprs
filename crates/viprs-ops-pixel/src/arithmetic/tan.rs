use viprs_core::{
    format::{FloatFormat, FloatSample},
    image::{DemandHint, Region, Tile, TileMut},
    op::{NodeSpec, Op, PixelLocalOp},
};

/// Tangent of each pixel sample, with angles expressed in degrees.
///
/// # Examples
/// ```ignore
/// use viprs_ops_pixel::arithmetic::tan::Tan;
///
/// let op = Tan::new(/* operation parameters */);
/// // Run `op` through a compiled image pipeline.
/// ```
pub struct Tan<F: FloatFormat>(std::marker::PhantomData<F>)
where
    F::Sample: FloatSample;

#[allow(dead_code)]
impl<F: FloatFormat> Tan<F>
where
    F::Sample: FloatSample,
{
    #[must_use]
    /// Creates a new `Tan`.
    pub const fn new() -> Self {
        Self(std::marker::PhantomData)
    }
}

impl<F: FloatFormat> Default for Tan<F>
where
    F::Sample: FloatSample,
{
    fn default() -> Self {
        Self::new()
    }
}

impl<F: FloatFormat> Op for Tan<F>
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
            *d = s.s_tan();
        }
    }
}

/// `Tan` is pixel-local: output(x,y) depends only on input(x,y).
/// No neighbourhood access; identity tile geometry. See `PixelLocalOp` for invariants.
impl<F: FloatFormat> PixelLocalOp for Tan<F> where F::Sample: FloatSample {}
#[cfg(test)]
mod tests {
    use super::*;
    use proptest::prelude::*;
    use viprs_core::{format::F32, image::Region};

    fn make_region(w: u32, h: u32) -> Region {
        Region::new(0, 0, w, h)
    }

    #[test]
    fn tan_f32_known_values() {
        let op = Tan::<F32>::new();
        let r = make_region(2, 1);
        // libvips interprets angles in degrees.
        let input_data = vec![0.0f32, 45.0];
        let mut output_data = vec![0.0f32; 2];
        let input = Tile::<F32>::new(r, 1, &input_data);
        let mut output = TileMut::<F32>::new(r, 1, &mut output_data);
        let mut state = ();
        op.process_region(&mut state, &input, &mut output);
        assert!((output_data[0] - 0.0).abs() < 1e-6);
        assert!((output_data[1] - 1.0).abs() < 1e-6);
    }

    #[test]
    fn tan_metadata_matches_identity_geometry() {
        let op = Tan::<F32>::default();
        let region = make_region(4, 2);
        assert_eq!(op.demand_hint(), viprs_core::image::DemandHint::Any);
        assert_eq!(op.required_input_region(&region), region);
        assert_eq!(op.node_spec(4, 2), viprs_core::op::NodeSpec::identity(4, 2));
    }

    proptest! {
        #[test]
        fn tan_finite_for_small_inputs(
            // Avoid near 90 degrees where tan diverges.
            pixels in proptest::collection::vec(-45.0f32..=45.0f32, 1..=32)
        ) {
            let len = pixels.len();
            let op = Tan::<F32>::new();
            let r = Region::new(0, 0, len as u32, 1);
            let mut output = vec![0.0f32; len];
            {
                let input = Tile::<F32>::new(r, 1, &pixels);
                let mut out = TileMut::<F32>::new(r, 1, &mut output);
                let mut state = ();
                op.process_region(&mut state, &input, &mut out);
            }
            for v in &output {
                prop_assert!(v.is_finite(), "tan output not finite for small input: {}", v);
            }
        }
    }
}
