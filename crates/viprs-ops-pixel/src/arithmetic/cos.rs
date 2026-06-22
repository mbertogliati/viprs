use viprs_core::{
    format::{FloatFormat, FloatSample},
    image::{DemandHint, Region, Tile, TileMut},
    op::{NodeSpec, Op, PixelLocalOp},
};

/// Cosine of each pixel sample, with angles expressed in degrees.
///
/// # Examples
/// ```ignore
/// use viprs_ops_pixel::arithmetic::cos::Cos;
///
/// let op = Cos::new(/* operation parameters */);
/// // Run `op` through a compiled image pipeline.
/// ```
pub struct Cos<F: FloatFormat>(std::marker::PhantomData<F>)
where
    F::Sample: FloatSample;

#[allow(dead_code)]
impl<F: FloatFormat> Cos<F>
where
    F::Sample: FloatSample,
{
    #[must_use]
    /// Creates a new `Cos`.
    pub const fn new() -> Self {
        Self(std::marker::PhantomData)
    }
}

impl<F: FloatFormat> Default for Cos<F>
where
    F::Sample: FloatSample,
{
    fn default() -> Self {
        Self::new()
    }
}

impl<F: FloatFormat> Op for Cos<F>
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
            *d = s.s_cos();
        }
    }
}

/// `Cos` is pixel-local: output(x,y) depends only on input(x,y).
/// No neighbourhood access; identity tile geometry. See `PixelLocalOp` for invariants.
impl<F: FloatFormat> PixelLocalOp for Cos<F> where F::Sample: FloatSample {}
#[cfg(test)]
mod tests {
    use super::*;
    use proptest::prelude::*;
    use viprs_core::{format::F32, image::Region};
    fn make_region(w: u32, h: u32) -> Region {
        Region::new(0, 0, w, h)
    }

    #[test]
    fn cos_f32_known_values() {
        let op = Cos::<F32>::new();
        let r = make_region(3, 1);
        // libvips interprets angles in degrees.
        let input_data = vec![0.0f32, 90.0, 180.0];
        let mut output_data = vec![0.0f32; 3];
        let input = Tile::<F32>::new(r, 1, &input_data);
        let mut output = TileMut::<F32>::new(r, 1, &mut output_data);
        let mut state = ();
        op.process_region(&mut state, &input, &mut output);
        assert!((output_data[0] - 1.0).abs() < 1e-6);
        assert!((output_data[1] - 0.0).abs() < 1e-6);
        assert!((output_data[2] - (-1.0)).abs() < 1e-6);
    }

    #[test]
    fn cos_metadata_matches_identity_geometry() {
        let op = Cos::<F32>::default();
        let region = make_region(4, 2);
        assert_eq!(op.demand_hint(), viprs_core::image::DemandHint::Any);
        assert_eq!(op.required_input_region(&region), region);
        assert_eq!(op.node_spec(4, 2), viprs_core::op::NodeSpec::identity(4, 2));
    }

    proptest! {
        #[test]
        fn cos_output_in_range(
            pixels in proptest::collection::vec(-100.0f32..=100.0f32, 1..=32)
        ) {
            let len = pixels.len();
            let op = Cos::<F32>::new();
            let r = Region::new(0, 0, len as u32, 1);
            let mut output = vec![0.0f32; len];
            {
                let input = Tile::<F32>::new(r, 1, &pixels);
                let mut out = TileMut::<F32>::new(r, 1, &mut output);
                let mut state = ();
                op.process_region(&mut state, &input, &mut out);
            }
            for v in &output {
                prop_assert!(
                    v.is_nan() || (*v >= -1.0 && *v <= 1.0),
                    "cos output out of [-1, 1]: {}",
                    v
                );
            }
        }
    }

    /// Ported from libvips `test_arithmetic.py::test_cos`.
    ///
    /// libvips test: `my_cos(x) = math.cos(math.radians(x))` applied per-pixel.
    /// Key reference values: cos(60°) = 0.5, cos(45°) ≈ √2/2, cos(180°) = -1.
    #[test]
    fn cos_libvips_reference_values_degrees() {
        let op = Cos::<F32>::new();
        let r = make_region(4, 1);
        // Input in degrees, matching libvips convention.
        let input_data = vec![60.0f32, 45.0, 180.0, 360.0];
        let mut output_data = vec![0.0f32; 4];
        let input = Tile::<F32>::new(r, 1, &input_data);
        let mut output = TileMut::<F32>::new(r, 1, &mut output_data);
        let mut state = ();
        op.process_region(&mut state, &input, &mut output);

        // cos(60°) = 0.5
        assert!(
            (output_data[0] - 0.5).abs() < 1e-6,
            "cos(60°)={}",
            output_data[0]
        );
        // cos(45°) = √2/2 ≈ 0.7071067
        assert!(
            (output_data[1] - std::f32::consts::FRAC_1_SQRT_2).abs() < 1e-6,
            "cos(45°)={}",
            output_data[1]
        );
        // cos(180°) = -1
        assert!(
            (output_data[2] - (-1.0)).abs() < 1e-6,
            "cos(180°)={}",
            output_data[2]
        );
        // cos(360°) = 1
        assert!(
            (output_data[3] - 1.0).abs() < 1e-6,
            "cos(360°)={}",
            output_data[3]
        );
    }
}
