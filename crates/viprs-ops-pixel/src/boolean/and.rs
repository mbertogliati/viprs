use std::marker::PhantomData;

use viprs_core::{
    error::BooleanError,
    format::BandFormat,
    image::{DemandHint, Region, Tile, TileMut},
    op::{NodeSpec, Op},
};

use super::common::{
    BooleanOperand, BooleanOutput, BooleanOutputSample, BooleanResultSample, cast_rhs_constants,
    cast_rhs_vec, process_boolean_region,
};

/// Bitwise AND of each pixel sample with a pre-cast rhs buffer.
///
/// # Examples
/// ```ignore
/// use viprs::domain::ops::boolean::and::And;
///
/// let op = And::new(/* operation parameters */);
/// // Run `op` through a compiled image pipeline.
/// ```
#[allow(dead_code)]
pub struct And<L: BandFormat + BooleanOperand<R>, R: BandFormat = L> {
    rhs: Vec<BooleanOutputSample<L, R>>,
    _formats: PhantomData<(L, R)>,
}

#[allow(dead_code)]
impl<L: BandFormat + BooleanOperand<R>, R: BandFormat> And<L, R> {
    /// Creates a new `And`.
    pub fn new(mask: R::Sample) -> Self {
        Self::from_vec(vec![mask])
    }

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

impl<L, R> Op for And<L, R>
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
            super::common::BooleanResultSample::bool_and,
        );
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use proptest::prelude::*;
    use viprs_core::{
        format::{F32, I16, U8, U16},
        image::{Region, Tile, TileMut},
    };

    fn run_and<L, R>(
        op: &And<L, R>,
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
    fn mixed_float_and_u16_promotes_to_i32() {
        let op = And::<F32, U16>::new(0b0110);
        let input = [7.9f32, 3.4, 12.0];
        let mut output = [0i32; 3];
        run_and(&op, &input, 1, &mut output);
        assert_eq!(output, [6, 2, 4]);
    }

    #[test]
    fn and_metadata_matches_identity_geometry() {
        let op = And::<U8>::new(0x0F);
        let region = Region::new(0, 0, 4, 2);
        assert_eq!(op.demand_hint(), viprs_core::image::DemandHint::Any);
        assert_eq!(op.required_input_region(&region), region);
        assert_eq!(op.node_spec(4, 2), viprs_core::op::NodeSpec::identity(4, 2));
    }

    proptest! {
        #[test]
        fn and_identity_with_single_band_rhs_broadcast(
            pixels in 1usize..=32,
            bands in 1u32..=4,
            input in proptest::collection::vec(any::<u8>(), 1..=128),
        ) {
            let len = pixels * bands as usize;
            let input = input.into_iter().take(len).collect::<Vec<_>>();
            prop_assume!(input.len() == len);
            let op = And::<U8>::from_vec(vec![u8::MAX; pixels]);
            let mut output = vec![0u8; len];
            run_and(&op, &input, bands, &mut output);
            prop_assert_eq!(output, input);
        }

        #[test]
        fn and_zero_annihilates_with_single_band_rhs_broadcast(
            pixels in 1usize..=32,
            bands in 1u32..=4,
            input in proptest::collection::vec(any::<u16>(), 1..=128),
        ) {
            let len = pixels * bands as usize;
            let input = input.into_iter().take(len).collect::<Vec<_>>();
            prop_assume!(input.len() == len);
            let op = And::<U16>::from_vec(vec![0u16; pixels]);
            let mut output = vec![u16::MAX; len];
            run_and(&op, &input, bands, &mut output);
            prop_assert!(output.iter().all(|&sample| sample == 0));
        }

        #[test]
        fn and_mixed_i16_rhs_matches_common_i32_cast(
            input in proptest::collection::vec(any::<i16>(), 1..=64),
            rhs in proptest::collection::vec(any::<u16>(), 1..=64),
        ) {
            let len = input.len().min(rhs.len());
            prop_assume!(len > 0);
            let input = input[..len].to_vec();
            let rhs = rhs[..len].to_vec();
            let op = And::<I16, U16>::from_vec(rhs.clone());
            let mut output = vec![0i16; len];
            run_and(&op, &input, 1, &mut output);
            let expected = input.iter().zip(rhs.iter()).map(|(&left, &right)| left & (right as i16)).collect::<Vec<_>>();
            prop_assert_eq!(output, expected);
        }
    }
}
