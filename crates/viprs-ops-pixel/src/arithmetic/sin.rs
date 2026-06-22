use viprs_core::{
    format::{FloatFormat, FloatSample},
    image::{DemandHint, Region, Tile, TileMut},
    op::{NodeSpec, Op, PixelLocalOp},
};

/// Sine of each pixel sample, with angles expressed in degrees.
///
/// # Examples
/// ```ignore
/// use viprs_ops_pixel::arithmetic::sin::Sin;
///
/// let op = Sin::new(/* operation parameters */);
/// // Run `op` through a compiled image pipeline.
/// ```
pub struct Sin<F: FloatFormat>(std::marker::PhantomData<F>)
where
    F::Sample: FloatSample;

#[allow(dead_code)]
impl<F: FloatFormat> Sin<F>
where
    F::Sample: FloatSample,
{
    #[must_use]
    /// Creates a new `Sin`.
    pub const fn new() -> Self {
        Self(std::marker::PhantomData)
    }
}

impl<F: FloatFormat> Default for Sin<F>
where
    F::Sample: FloatSample,
{
    fn default() -> Self {
        Self::new()
    }
}

impl<F: FloatFormat> Op for Sin<F>
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
            *d = s.s_sin();
        }
    }
}

/// `Sin` is pixel-local: output(x,y) depends only on input(x,y).
/// No neighbourhood access; identity tile geometry. See `PixelLocalOp` for invariants.
impl<F: FloatFormat> PixelLocalOp for Sin<F> where F::Sample: FloatSample {}
#[cfg(test)]
mod tests {
    use super::*;
    use proptest::prelude::*;
    use viprs_core::{format::F32, image::Region};
    fn make_region(w: u32, h: u32) -> Region {
        Region::new(0, 0, w, h)
    }

    #[test]
    fn sin_f32_known_values() {
        let op = Sin::<F32>::new();
        let r = make_region(3, 1);
        // libvips interprets angles in degrees.
        let input_data = vec![0.0f32, 90.0, 180.0];
        let mut output_data = vec![0.0f32; 3];
        let input = Tile::<F32>::new(r, 1, &input_data);
        let mut output = TileMut::<F32>::new(r, 1, &mut output_data);
        let mut state = ();
        op.process_region(&mut state, &input, &mut output);
        assert!((output_data[0] - 0.0).abs() < 1e-6);
        assert!((output_data[1] - 1.0).abs() < 1e-6);
        assert!((output_data[2] - 0.0).abs() < 1e-6);
    }

    #[test]
    fn sin_metadata_matches_identity_geometry() {
        let op = Sin::<F32>::default();
        let region = make_region(4, 2);
        assert_eq!(op.demand_hint(), viprs_core::image::DemandHint::Any);
        assert_eq!(op.required_input_region(&region), region);
        assert_eq!(op.node_spec(4, 2), viprs_core::op::NodeSpec::identity(4, 2));
    }

    proptest! {
        #[test]
        fn sin_output_in_range(
            pixels in proptest::collection::vec(-100.0f32..=100.0f32, 1..=32)
        ) {
            let len = pixels.len();
            let op = Sin::<F32>::new();
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
                    "sin output out of [-1, 1]: {}",
                    v
                );
            }
        }
    }

    /// Ported from libvips `test_arithmetic.py::test_sin`.
    ///
    /// libvips test: `my_sin(x) = math.sin(math.radians(x))` applied per-pixel.
    /// Key reference values: sin(30°) = 0.5, sin(45°) ≈ √2/2, sin(270°) = -1.
    #[test]
    fn sin_libvips_reference_values_degrees() {
        let op = Sin::<F32>::new();
        let r = make_region(4, 1);
        // Input in degrees, matching libvips convention.
        let input_data = vec![30.0f32, 45.0, 270.0, 360.0];
        let mut output_data = vec![0.0f32; 4];
        let input = Tile::<F32>::new(r, 1, &input_data);
        let mut output = TileMut::<F32>::new(r, 1, &mut output_data);
        let mut state = ();
        op.process_region(&mut state, &input, &mut output);

        // sin(30°) = 0.5 exactly (IEEE 754 representable)
        assert!(
            (output_data[0] - 0.5).abs() < 1e-6,
            "sin(30°)={}",
            output_data[0]
        );
        // sin(45°) = √2/2 ≈ 0.7071067
        assert!(
            (output_data[1] - std::f32::consts::FRAC_1_SQRT_2).abs() < 1e-6,
            "sin(45°)={}",
            output_data[1]
        );
        // sin(270°) = -1
        assert!(
            (output_data[2] - (-1.0)).abs() < 1e-6,
            "sin(270°)={}",
            output_data[2]
        );
        // sin(360°) ≈ 0
        assert!(output_data[3].abs() < 1e-6, "sin(360°)={}", output_data[3]);
    }

    /// Ported from libvips `test_arithmetic.py::test_sin` + `test_cos`.
    ///
    /// Pythagorean identity: sin²(x) + cos²(x) = 1 for all x (in degrees).
    /// This exercises the degrees-to-radians conversion in both ops.
    #[test]
    fn sin_squared_plus_cos_squared_equals_one() {
        use super::super::cos::Cos;

        let angles = [0.0f32, 30.0, 45.0, 60.0, 90.0, 135.0, 180.0, 270.0, 360.0];
        let r = make_region(angles.len() as u32, 1);
        let input = angles.to_vec();

        let sin_op = Sin::<F32>::new();
        let cos_op = Cos::<F32>::new();

        let mut sin_out = vec![0.0f32; angles.len()];
        let mut cos_out = vec![0.0f32; angles.len()];

        {
            let inp = Tile::<F32>::new(r, 1, &input);
            let mut out = TileMut::<F32>::new(r, 1, &mut sin_out);
            let mut state = ();
            sin_op.process_region(&mut state, &inp, &mut out);
        }
        {
            let inp = Tile::<F32>::new(r, 1, &input);
            let mut out = TileMut::<F32>::new(r, 1, &mut cos_out);
            let mut state = ();
            cos_op.process_region(&mut state, &inp, &mut out);
        }

        for (i, (s, c)) in sin_out.iter().zip(cos_out.iter()).enumerate() {
            let identity = s * s + c * c;
            assert!(
                (identity - 1.0).abs() < 1e-5,
                "sin²({})°+cos²({})°={} ≠ 1",
                angles[i],
                angles[i],
                identity
            );
        }
    }
}
