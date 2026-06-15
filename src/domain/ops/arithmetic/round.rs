use crate::{
    domain::op::{NodeSpec, Op, PixelLocalOp},
    domain::{
        format::{FloatFormat, FloatSample},
        image::{DemandHint, Region, Tile, TileMut},
    },
};

/// Round each float sample to the nearest integer (ties round to even, like libvips `rint`).
///
/// # Examples
/// ```ignore
/// use viprs::domain::ops::arithmetic::round::Round;
///
/// let op = Round::new(/* operation parameters */);
/// // Run `op` through a compiled image pipeline.
/// ```
pub struct Round<F: FloatFormat>(std::marker::PhantomData<F>)
where
    F::Sample: FloatSample;

#[allow(dead_code)]
impl<F: FloatFormat> Round<F>
where
    F::Sample: FloatSample,
{
    #[must_use]
    /// Creates a new `Round`.
    pub const fn new() -> Self {
        Self(std::marker::PhantomData)
    }
}

impl<F: FloatFormat> Default for Round<F>
where
    F::Sample: FloatSample,
{
    fn default() -> Self {
        Self::new()
    }
}

impl<F: FloatFormat> Op for Round<F>
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
            *d = s.s_round();
        }
    }
}

/// Floor of each float sample (round towards −∞).
///
/// # Examples
/// ```ignore
/// use viprs::domain::ops::arithmetic::round::Floor;
///
/// let op = Floor::new(/* operation parameters */);
/// // Run `op` through a compiled image pipeline.
/// ```
pub struct Floor<F: FloatFormat>(std::marker::PhantomData<F>)
where
    F::Sample: FloatSample;

#[allow(dead_code)]
impl<F: FloatFormat> Floor<F>
where
    F::Sample: FloatSample,
{
    #[must_use]
    /// Creates a new `Floor`.
    pub const fn new() -> Self {
        Self(std::marker::PhantomData)
    }
}

impl<F: FloatFormat> Default for Floor<F>
where
    F::Sample: FloatSample,
{
    fn default() -> Self {
        Self::new()
    }
}

impl<F: FloatFormat> Op for Floor<F>
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
            *d = s.s_floor();
        }
    }
}

/// Ceiling of each float sample (round towards +∞).
///
/// # Examples
/// ```ignore
/// use viprs::domain::ops::arithmetic::round::Ceil;
///
/// let op = Ceil::new(/* operation parameters */);
/// // Run `op` through a compiled image pipeline.
/// ```
pub struct Ceil<F: FloatFormat>(std::marker::PhantomData<F>)
where
    F::Sample: FloatSample;

#[allow(dead_code)]
impl<F: FloatFormat> Ceil<F>
where
    F::Sample: FloatSample,
{
    #[must_use]
    /// Creates a new `Ceil`.
    pub const fn new() -> Self {
        Self(std::marker::PhantomData)
    }
}

impl<F: FloatFormat> Default for Ceil<F>
where
    F::Sample: FloatSample,
{
    fn default() -> Self {
        Self::new()
    }
}

impl<F: FloatFormat> Op for Ceil<F>
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
            *d = s.s_ceil();
        }
    }
}

impl<F: FloatFormat> PixelLocalOp for Round<F> where F::Sample: FloatSample {}
impl<F: FloatFormat> PixelLocalOp for Floor<F> where F::Sample: FloatSample {}
impl<F: FloatFormat> PixelLocalOp for Ceil<F> where F::Sample: FloatSample {}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::domain::{format::F32, image::Region};
    use proptest::prelude::*;

    fn make_region(w: u32, h: u32) -> Region {
        Region::new(0, 0, w, h)
    }

    #[test]
    fn round_f32_known_values() {
        let op = Round::<F32>::new();
        let r = make_region(7, 1);
        let input_data = vec![0.4f32, 0.5, 1.5, 2.5, -0.5, -1.5, -2.5];
        let mut output_data = vec![0.0f32; 7];
        let input = Tile::<F32>::new(r, 1, &input_data);
        let mut output = TileMut::<F32>::new(r, 1, &mut output_data);
        let mut state = ();
        op.process_region(&mut state, &input, &mut output);
        assert!((output_data[0] - 0.0).abs() < f32::EPSILON);
        assert!((output_data[1] - 0.0).abs() < f32::EPSILON);
        assert!((output_data[2] - 2.0).abs() < f32::EPSILON);
        assert!((output_data[3] - 2.0).abs() < f32::EPSILON);
        assert!((output_data[4] - 0.0).abs() < f32::EPSILON);
        assert!((output_data[5] - (-2.0)).abs() < f32::EPSILON);
        assert!((output_data[6] - (-2.0)).abs() < f32::EPSILON);
    }

    #[test]
    fn round_floor_and_ceil_report_identity_metadata() {
        let round = Round::<F32>::default();
        let floor = Floor::<F32>::default();
        let ceil = Ceil::<F32>::default();
        let region = Region::new(2, 3, 4, 5);

        round.start();
        floor.start();
        ceil.start();

        assert_eq!(round.demand_hint(), DemandHint::Any);
        assert_eq!(floor.demand_hint(), DemandHint::Any);
        assert_eq!(ceil.demand_hint(), DemandHint::Any);
        assert_eq!(round.required_input_region(&region), region);
        assert_eq!(floor.required_input_region(&region), region);
        assert_eq!(ceil.required_input_region(&region), region);
        assert_eq!(round.node_spec(64, 32), NodeSpec::identity(64, 32));
        assert_eq!(floor.node_spec(64, 32), NodeSpec::identity(64, 32));
        assert_eq!(ceil.node_spec(64, 32), NodeSpec::identity(64, 32));
    }

    #[test]
    fn floor_f32_known_values() {
        let op = Floor::<F32>::new();
        let r = make_region(4, 1);
        let input_data = vec![0.9f32, -0.1, 2.0, -2.9];
        let mut output_data = vec![0.0f32; 4];
        let input = Tile::<F32>::new(r, 1, &input_data);
        let mut output = TileMut::<F32>::new(r, 1, &mut output_data);
        let mut state = ();
        op.process_region(&mut state, &input, &mut output);
        assert!((output_data[0] - 0.0).abs() < f32::EPSILON);
        assert!((output_data[1] - (-1.0)).abs() < f32::EPSILON);
        assert!((output_data[2] - 2.0).abs() < f32::EPSILON);
        assert!((output_data[3] - (-3.0)).abs() < f32::EPSILON);
    }

    #[test]
    fn ceil_f32_known_values() {
        let op = Ceil::<F32>::new();
        let r = make_region(4, 1);
        let input_data = vec![0.1f32, -0.9, 2.0, -2.1];
        let mut output_data = vec![0.0f32; 4];
        let input = Tile::<F32>::new(r, 1, &input_data);
        let mut output = TileMut::<F32>::new(r, 1, &mut output_data);
        let mut state = ();
        op.process_region(&mut state, &input, &mut output);
        assert!((output_data[0] - 1.0).abs() < f32::EPSILON);
        assert!((output_data[1] - 0.0).abs() < f32::EPSILON);
        assert!((output_data[2] - 2.0).abs() < f32::EPSILON);
        assert!((output_data[3] - (-2.0)).abs() < f32::EPSILON);
    }

    proptest! {
        #[test]
        fn floor_le_value_le_ceil(
            pixels in proptest::collection::vec(-1000.0f32..=1000.0f32, 1..=32)
        ) {
            let len = pixels.len();
            let op_floor = Floor::<F32>::new();
            let op_ceil = Ceil::<F32>::new();
            let r = Region::new(0, 0, len as u32, 1);
            let mut floors = vec![0.0f32; len];
            let mut ceils = vec![0.0f32; len];
            {
                let input = Tile::<F32>::new(r, 1, &pixels);
                let mut output = TileMut::<F32>::new(r, 1, &mut floors);
                let mut state = ();
                op_floor.process_region(&mut state, &input, &mut output);
            }
            {
                let input = Tile::<F32>::new(r, 1, &pixels);
                let mut output = TileMut::<F32>::new(r, 1, &mut ceils);
                let mut state = ();
                op_ceil.process_region(&mut state, &input, &mut output);
            }
            for ((f, c), p) in floors.iter().zip(ceils.iter()).zip(pixels.iter()) {
                prop_assert!(*f <= *p, "floor({}) > {}", p, f);
                prop_assert!(*c >= *p, "ceil({}) < {}", p, c);
            }
        }

        #[test]
        fn round_idempotent_on_integers(
            pixels in proptest::collection::vec(-100.0f32..=100.0f32, 1..=32)
        ) {
            let pixels: Vec<f32> = pixels.iter().map(|p| p.round_ties_even()).collect();
            let len = pixels.len();
            let op = Round::<F32>::new();
            let r = Region::new(0, 0, len as u32, 1);
            let mut result = vec![0.0f32; len];
            {
                let input = Tile::<F32>::new(r, 1, &pixels);
                let mut output = TileMut::<F32>::new(r, 1, &mut result);
                let mut state = ();
                op.process_region(&mut state, &input, &mut output);
            }
            for (a, b) in pixels.iter().zip(result.iter()) {
                prop_assert!((a - b).abs() < f32::EPSILON, "round({}) != {}", a, b);
            }
        }

        #[test]
        fn round_matches_ties_even_reference(
            pixels in proptest::collection::vec(-1000.0f32..=1000.0f32, 1..=32)
        ) {
            let len = pixels.len();
            let op = Round::<F32>::new();
            let r = Region::new(0, 0, len as u32, 1);
            let expected: Vec<f32> = pixels.iter().map(|p| p.round_ties_even()).collect();
            let mut result = vec![0.0f32; len];
            let input = Tile::<F32>::new(r, 1, &pixels);
            let mut output = TileMut::<F32>::new(r, 1, &mut result);
            let mut state = ();

            op.process_region(&mut state, &input, &mut output);

            for (actual, expected) in result.iter().zip(expected.iter()) {
                prop_assert!((actual - expected).abs() < f32::EPSILON);
            }
        }

        #[test]
        fn floor_is_identity_for_integer_values(
            pixels in proptest::collection::vec(-100.0f32..=100.0f32, 1..=32)
        ) {
            let pixels: Vec<f32> = pixels.iter().map(|p| p.round_ties_even()).collect();
            let len = pixels.len();
            let op = Floor::<F32>::new();
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

        #[test]
        fn ceil_is_identity_for_integer_values(
            pixels in proptest::collection::vec(-100.0f32..=100.0f32, 1..=32)
        ) {
            let pixels: Vec<f32> = pixels.iter().map(|p| p.round_ties_even()).collect();
            let len = pixels.len();
            let op = Ceil::<F32>::new();
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
