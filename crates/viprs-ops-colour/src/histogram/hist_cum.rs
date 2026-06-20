use std::marker::PhantomData;

use bytemuck::Pod;

use viprs_core::{
    format::{BandFormat, F32, F64, I16, I32, U8, U16, U32},
    image::{DemandHint, Region, Tile, TileMut},
    op::Op,
    shared_ops::sample_conv::{FromF64, ToF64},
};

/// Compute the cumulative histogram band-by-band.
///
/// # Examples
/// ```ignore
/// use viprs::domain::ops::histogram::hist_cum::HistCumOp;
///
/// let op = HistCumOp::new(/* operation parameters */);
/// // Run `op` through a compiled image pipeline.
/// ```
pub struct HistCumOp<F: BandFormat> {
    _phantom: PhantomData<F>,
}

/// Defines the contract for hist cum output format.
pub trait HistCumOutputFormat: BandFormat {
    /// Associated type for output.
    type Output: BandFormat;
}

impl HistCumOutputFormat for U8 {
    type Output = U32;
}

impl HistCumOutputFormat for U16 {
    type Output = U32;
}

impl HistCumOutputFormat for U32 {
    type Output = Self;
}

impl HistCumOutputFormat for I16 {
    type Output = I32;
}

impl HistCumOutputFormat for I32 {
    type Output = Self;
}

impl HistCumOutputFormat for F32 {
    type Output = Self;
}

impl HistCumOutputFormat for F64 {
    type Output = Self;
}

impl<F: BandFormat> HistCumOp<F> {
    #[must_use]
    /// Creates a new `HistCumOp`.
    pub const fn new() -> Self {
        Self {
            _phantom: PhantomData,
        }
    }
}

impl<F: BandFormat> Default for HistCumOp<F> {
    fn default() -> Self {
        Self::new()
    }
}

impl<F> Op for HistCumOp<F>
where
    F: HistCumOutputFormat,
    F::Sample: ToF64 + FromF64 + Pod,
    <F::Output as BandFormat>::Sample: FromF64 + Pod,
{
    type Input = F;
    type Output = F::Output;
    type State = ();

    fn demand_hint(&self) -> DemandHint {
        DemandHint::Any
    }

    fn required_input_region(&self, output: &Region) -> Region {
        *output
    }

    fn start(&self) {}

    #[inline]
    fn process_region(&self, _state: &mut (), input: &Tile<F>, output: &mut TileMut<Self::Output>) {
        let bands = input.bands as usize;
        for band in 0..bands {
            let mut total = 0.0f64;
            for idx in (band..input.data.len()).step_by(bands) {
                total += input.data[idx].to_f64();
                output.data[idx] = <Self::Output as BandFormat>::Sample::from_f64(total);
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use proptest::prelude::*;
    use viprs_core::{
        format::{BandFormatId, F32, I16, U8, U16, U32},
        image::Region,
        op::{DynOperation, OperationBridge},
    };

    fn run_op<Input, Output>(input_data: &[Input::Sample]) -> Vec<Output::Sample>
    where
        Input: HistCumOutputFormat<Output = Output>,
        Output: BandFormat,
        Input::Sample: ToF64 + FromF64 + Pod + Copy + Default,
        Output::Sample: FromF64 + Pod + Copy + Default,
    {
        let region = Region::new(0, 0, input_data.len() as u32, 1);
        let input = Tile::<Input>::new(region, 1, input_data);
        let mut output_data = vec![Output::Sample::default(); input_data.len()];
        let mut output = TileMut::<Output>::new(region, 1, &mut output_data);
        let mut state = ();
        HistCumOp::<Input>::new().process_region(&mut state, &input, &mut output);
        output_data
    }

    #[test]
    fn hist_cum_zero_histogram_stays_zero() {
        let output = run_op::<F32, F32>(&[0.0, 0.0, 0.0, 0.0]);
        assert_eq!(output, vec![0.0, 0.0, 0.0, 0.0]);
    }

    #[test]
    fn hist_cum_known_values() {
        let output = run_op::<F32, F32>(&[1.0, 2.0, 3.0, 4.0]);
        assert_eq!(output, vec![1.0, 3.0, 6.0, 10.0]);
    }

    #[test]
    fn hist_cum_accumulates_each_band_independently() {
        let region = Region::new(0, 0, 3, 1);
        let input_data = vec![1.0f32, 10.0, 2.0, 20.0, 3.0, 30.0];
        let input = Tile::<F32>::new(region, 2, &input_data);
        let mut output_data = vec![0.0f32; input_data.len()];
        let mut output = TileMut::<F32>::new(region, 2, &mut output_data);
        let mut state = ();

        HistCumOp::<F32>::default().process_region(&mut state, &input, &mut output);

        assert_eq!(output_data, vec![1.0, 10.0, 3.0, 30.0, 6.0, 60.0]);
    }

    #[test]
    fn hist_cum_promotes_u8_output_to_u32() {
        let bridge = OperationBridge::new(HistCumOp::<U8>::new(), 1);
        assert_eq!(bridge.input_format(), BandFormatId::U8);
        assert_eq!(bridge.output_format(), BandFormatId::U32);

        let output = run_op::<U8, U32>(&[1, 2, 3, 4]);
        assert_eq!(output, vec![1, 3, 6, 10]);
    }

    #[test]
    fn hist_cum_promotes_u16_output_to_u32() {
        let bridge = OperationBridge::new(HistCumOp::<U16>::new(), 1);
        assert_eq!(bridge.input_format(), BandFormatId::U16);
        assert_eq!(bridge.output_format(), BandFormatId::U32);
    }

    #[test]
    fn hist_cum_promotes_i16_output_to_i32() {
        let bridge = OperationBridge::new(HistCumOp::<I16>::new(), 1);
        assert_eq!(bridge.input_format(), BandFormatId::I16);
        assert_eq!(bridge.output_format(), BandFormatId::I32);
    }

    #[test]
    fn hist_cum_region_contract_is_identity() {
        let op = HistCumOp::<F32>::new();
        let region = Region::new(-2, 7, 13, 5);
        assert_eq!(op.demand_hint(), DemandHint::Any);
        assert_eq!(op.required_input_region(&region), region);
        op.start();
    }

    proptest! {
        #[test]
        fn hist_cum_matches_running_sum(values in proptest::collection::vec(0u16..=256u16, 1..=64)) {
            let input: Vec<f32> = values.iter().map(|&v| f32::from(v)).collect();
            let output = run_op::<F32, F32>(&input);
            let mut total = 0.0f32;
            let mut previous = 0.0f32;
            for (i, value) in input.iter().enumerate() {
                total += *value;
                prop_assert!((output[i] - total).abs() < f32::EPSILON);
                prop_assert!(output[i] >= previous);
                previous = output[i];
            }
        }
    }
}
