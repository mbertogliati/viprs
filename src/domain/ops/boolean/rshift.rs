use std::marker::PhantomData;

use crate::{
    domain::op::{NodeSpec, Op},
    domain::{
        error::BooleanError,
        format::{BandFormat, I32},
        image::{DemandHint, Region, Tile, TileMut},
    },
};

use super::common::{
    BooleanOperand, BooleanOutput, BooleanOutputSample, BooleanResultSample, cast_rhs_constants,
    cast_rhs_vec, process_boolean_region,
};

/// Right-shift each pixel sample by a pre-cast rhs buffer.
///
/// # Examples
/// ```ignore
/// use viprs::domain::ops::boolean::rshift::RShift;
///
/// let op = RShift::new(/* operation parameters */);
/// // Run `op` through a compiled image pipeline.
/// ```
#[allow(dead_code)]
pub struct RShift<L: BandFormat + BooleanOperand<R>, R: BandFormat = I32> {
    rhs: Vec<BooleanOutputSample<L, R>>,
    _formats: PhantomData<(L, R)>,
}

#[allow(dead_code)]
impl<L: BandFormat + BooleanOperand<I32>> RShift<L, I32> {
    #[must_use]
    /// Creates a new `RShift`.
    pub fn new(shift: u32) -> Self {
        Self::from_vec(vec![shift as i32])
    }
}

#[allow(dead_code)]
impl<L: BandFormat + BooleanOperand<R>, R: BandFormat> RShift<L, R> {
    #[must_use]
    /// Creates this value from vec.
    pub fn from_vec(rhs: Vec<R::Sample>) -> Self {
        Self {
            rhs: cast_rhs_vec::<L, R>(rhs),
            _formats: PhantomData,
        }
    }

    /// Creates this value from constants.
    pub fn from_constants(rhs: Vec<R::Sample>, bands: u32) -> Result<Self, BooleanError> {
        Ok(Self {
            rhs: cast_rhs_constants::<L, R>(rhs, bands)?,
            _formats: PhantomData,
        })
    }
}

impl<L, R> Op for RShift<L, R>
where
    L: BooleanOperand<R>,
    R: BandFormat,
    BooleanOutputSample<L, R>: BooleanResultSample,
{
    type Input = L;
    type Output = BooleanOutput<L, R>;
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
    fn process_region(
        &self,
        (): &mut (),
        input: &Tile<L>,
        output: &mut TileMut<BooleanOutput<L, R>>,
    ) {
        process_boolean_region::<L, R>(
            input,
            &self.rhs,
            output,
            super::common::BooleanResultSample::bool_rshift,
        );
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::domain::{
        format::{F32, I32, U8},
        image::{Region, Tile, TileMut},
    };
    use proptest::prelude::*;

    fn run_rshift<L, R>(
        op: &RShift<L, R>,
        input_data: &[L::Sample],
        bands: u32,
        output_data: &mut [BooleanOutputSample<L, R>],
    ) where
        L: BooleanOperand<R>,
        R: BandFormat,
        BooleanOutputSample<L, R>: BooleanResultSample,
    {
        let pixels = input_data.len() / bands as usize;
        let region = Region::new(0, 0, pixels as u32, 1);
        let input = Tile::<L>::new(region, bands, input_data);
        let mut output = TileMut::<BooleanOutput<L, R>>::new(region, bands, output_data);
        let mut state = ();
        op.process_region(&mut state, &input, &mut output);
    }

    #[test]
    fn mixed_float_right_shift_promotes_to_i32() {
        let op = RShift::<F32, U8>::from_vec(vec![1u8]);
        let input = [3.9f32, 8.1, -4.0];
        let mut output = [0i32; 3];
        run_rshift(&op, &input, 1, &mut output);
        assert_eq!(output, [1, 4, -2]);
    }

    #[test]
    fn rshift_metadata_matches_identity_geometry() {
        let op = RShift::<U8>::new(3);
        let region = Region::new(0, 0, 4, 2);
        assert_eq!(op.demand_hint(), crate::domain::image::DemandHint::Any);
        assert_eq!(op.required_input_region(&region), region);
        assert_eq!(
            op.node_spec(4, 2),
            crate::domain::op::NodeSpec::identity(4, 2)
        );
    }

    proptest! {
        #[test]
        fn rshift_u8_boundaries_match_checked_semantics(
            pixels in 1usize..=32,
            bands in 1u32..=4,
            input in proptest::collection::vec(any::<u8>(), 1..=128),
        ) {
            let len = pixels * bands as usize;
            let input = input.into_iter().take(len).collect::<Vec<_>>();
            prop_assume!(input.len() == len);

            let identity = RShift::<U8>::from_vec(vec![0i32; pixels]);
            let highest = RShift::<U8>::from_vec(vec![(u8::BITS - 1) as i32; pixels]);
            let overflow = RShift::<U8>::from_vec(vec![u8::BITS as i32; pixels]);

            let mut identity_output = vec![0u8; len];
            let mut highest_output = vec![0u8; len];
            let mut overflow_output = vec![u8::MAX; len];

            run_rshift(&identity, &input, bands, &mut identity_output);
            run_rshift(&highest, &input, bands, &mut highest_output);
            run_rshift(&overflow, &input, bands, &mut overflow_output);

            let highest_expected = input.iter().map(|&sample| sample.checked_shr(u8::BITS - 1).unwrap_or(0)).collect::<Vec<_>>();
            prop_assert_eq!(identity_output, input);
            prop_assert_eq!(highest_output, highest_expected);
            prop_assert!(overflow_output.iter().all(|&sample| sample == 0));
        }

        #[test]
        fn rshift_i32_boundary_extends_sign(
            input in proptest::collection::vec(any::<i32>(), 1..=64),
        ) {
            let highest = RShift::<I32>::from_vec(vec![(i32::BITS - 1) as i32; input.len()]);
            let overflow = RShift::<I32>::from_vec(vec![i32::BITS as i32; input.len()]);
            let mut highest_output = vec![0i32; input.len()];
            let mut overflow_output = vec![0i32; input.len()];
            run_rshift(&highest, &input, 1, &mut highest_output);
            run_rshift(&overflow, &input, 1, &mut overflow_output);
            let highest_expected = input.iter().map(|&sample| sample >> (i32::BITS - 1)).collect::<Vec<_>>();
            let overflow_expected = input.iter().map(|&sample| if sample < 0 { -1 } else { 0 }).collect::<Vec<_>>();
            prop_assert_eq!(highest_output, highest_expected);
            prop_assert_eq!(overflow_output, overflow_expected);
        }
    }
}
