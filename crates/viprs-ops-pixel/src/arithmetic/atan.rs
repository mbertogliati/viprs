use viprs_core::{
    format::{FloatFormat, FloatSample},
    image::{DemandHint, Region, Tile, TileMut},
    op::{NodeSpec, Op, PixelLocalOp},
};

/// Arc tangent of each pixel sample, returning degrees in `[-90, 90]`.
/// Only valid for float formats. Defined for all real inputs.
///
/// # Examples
/// ```ignore
/// use viprs_ops_pixel::arithmetic::atan::ATan;
///
/// let op = ATan::new(/* operation parameters */);
/// // Run `op` through a compiled image pipeline.
/// ```
pub struct ATan<F: FloatFormat>(std::marker::PhantomData<F>)
where
    F::Sample: FloatSample;

#[allow(dead_code)]
impl<F: FloatFormat> ATan<F>
where
    F::Sample: FloatSample,
{
    #[must_use]
    /// Creates a new `ATan`.
    pub const fn new() -> Self {
        Self(std::marker::PhantomData)
    }
}

impl<F: FloatFormat> Default for ATan<F>
where
    F::Sample: FloatSample,
{
    fn default() -> Self {
        Self::new()
    }
}

impl<F: FloatFormat> Op for ATan<F>
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
            *d = s.s_atan();
        }
    }
}

/// `ATan` is pixel-local: output(x,y) depends only on input(x,y).
/// No neighbourhood access; identity tile geometry. See `PixelLocalOp` for invariants.
impl<F: FloatFormat> PixelLocalOp for ATan<F> where F::Sample: FloatSample {}
#[cfg(test)]
mod tests {
    use super::*;
    use proptest::prelude::*;
    use viprs_core::{format::F32, image::Region};
    fn make_region(w: u32, h: u32) -> Region {
        Region::new(0, 0, w, h)
    }

    #[test]
    fn atan_f32_known_values() {
        let op = ATan::<F32>::new();
        let r = make_region(2, 1);
        // libvips returns inverse trig results in degrees.
        let input_data = vec![0.0f32, 1.0];
        let mut output_data = vec![0.0f32; 2];
        let input = Tile::<F32>::new(r, 1, &input_data);
        let mut output = TileMut::<F32>::new(r, 1, &mut output_data);
        let mut state = ();
        op.process_region(&mut state, &input, &mut output);
        assert!((output_data[0] - 0.0).abs() < 1e-6);
        assert!((output_data[1] - 45.0).abs() < 1e-6);
    }

    #[test]
    fn atan_metadata_matches_identity_geometry() {
        let op = ATan::<F32>::default();
        let region = make_region(4, 2);
        assert_eq!(op.demand_hint(), viprs_core::image::DemandHint::Any);
        assert_eq!(op.required_input_region(&region), region);
        assert_eq!(op.node_spec(4, 2), viprs_core::op::NodeSpec::identity(4, 2));
    }

    proptest! {
        #[test]
        fn atan_always_finite(
            pixels in proptest::collection::vec(-1000.0f32..=1000.0f32, 1..=32)
        ) {
            let len = pixels.len();
            let op = ATan::<F32>::new();
            let r = Region::new(0, 0, len as u32, 1);
            let mut output = vec![0.0f32; len];
            {
                let input = Tile::<F32>::new(r, 1, &pixels);
                let mut out = TileMut::<F32>::new(r, 1, &mut output);
                let mut state = ();
                op.process_region(&mut state, &input, &mut out);
            }
            // atan is defined on all reals: output always finite
            for v in &output {
                prop_assert!(v.is_finite(), "atan output not finite: {}", v);
            }
        }
    }
}
