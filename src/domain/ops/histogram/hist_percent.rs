use std::marker::PhantomData;

use bytemuck::Pod;

use crate::domain::{
    error::ViprsError,
    format::{BandFormat, U32},
    image::{DemandHint, Region, Tile, TileMut},
    op::Op,
    ops::resample::sample_conv::ToF64,
};

/// Find the first cumulative-histogram bin meeting a percentile threshold.
///
/// # Examples
/// ```ignore
/// use viprs::domain::ops::histogram::hist_percent::HistPercentOp;
///
/// let op = HistPercentOp::new(/* operation parameters */);
/// // Run `op` through a compiled image pipeline.
/// ```
pub struct HistPercentOp<F: BandFormat> {
    percent: f64,
    _phantom: PhantomData<F>,
}

impl<F: BandFormat> HistPercentOp<F> {
    /// Creates a new `HistPercentOp`.
    pub fn new(percent: f64) -> Result<Self, ViprsError> {
        if !percent.is_finite() || !(0.0..=1.0).contains(&percent) {
            return Err(ViprsError::Scheduler(
                "HistPercentOp percent must be finite and between 0.0 and 1.0".into(),
            ));
        }

        Ok(Self {
            percent,
            _phantom: PhantomData,
        })
    }
}

impl<F> Op for HistPercentOp<F>
where
    F: BandFormat,
    F::Sample: ToF64 + Pod,
{
    type Input = F;
    type Output = U32;
    type State = ();

    fn demand_hint(&self) -> DemandHint {
        DemandHint::Any
    }

    fn required_input_region(&self, output: &Region) -> Region {
        *output
    }

    fn start(&self) {}

    #[inline]
    fn process_region(&self, _state: &mut (), input: &Tile<F>, output: &mut TileMut<U32>) {
        let total = input
            .data
            .last()
            .map_or(0.0, |sample| sample.to_f64().max(0.0));
        let threshold = total * self.percent;
        let index = input
            .data
            .iter()
            .map(|sample| sample.to_f64().max(0.0))
            .position(|value| value >= threshold)
            .unwrap_or_else(|| input.data.len().saturating_sub(1));

        output.data.fill(index as u32);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::domain::{format::F32, image::Region};
    use proptest::prelude::*;

    fn run_op(input_data: &[f32], percent: f64) -> u32 {
        let input_region = Region::new(0, 0, input_data.len() as u32, 1);
        let output_region = Region::new(0, 0, 1, 1);
        let input = Tile::<F32>::new(input_region, 1, input_data);
        let mut output_data = vec![0u32; 1];
        let mut output = TileMut::<U32>::new(output_region, 1, &mut output_data);
        let mut state = ();
        let op = HistPercentOp::<F32>::new(percent).unwrap();
        op.process_region(&mut state, &input, &mut output);
        output_data[0]
    }

    #[test]
    fn hist_percent_finds_requested_bin() {
        assert_eq!(run_op(&[10.0, 20.0, 30.0, 40.0], 0.5), 1);
    }

    #[test]
    fn hist_percent_returns_last_bin_at_full_percent() {
        assert_eq!(run_op(&[10.0, 20.0, 30.0, 40.0], 1.0), 3);
    }

    #[test]
    fn hist_percent_rejects_invalid_percent() {
        assert!(HistPercentOp::<F32>::new(f64::NAN).is_err());
        assert!(HistPercentOp::<F32>::new(-0.1).is_err());
        assert!(HistPercentOp::<F32>::new(1.1).is_err());
    }

    #[test]
    fn hist_percent_empty_histogram_returns_zero() {
        let input_region = Region::new(0, 0, 0, 1);
        let output_region = Region::new(0, 0, 1, 1);
        let input_data: Vec<f32> = Vec::new();
        let input = Tile::<F32>::new(input_region, 1, &input_data);
        let mut output_data = vec![99u32; 1];
        let mut output = TileMut::<U32>::new(output_region, 1, &mut output_data);
        let mut state = ();
        let op = HistPercentOp::<F32>::new(0.5).unwrap();

        op.process_region(&mut state, &input, &mut output);

        assert_eq!(output_data, vec![0]);
    }

    #[test]
    fn hist_percent_region_contract_is_identity() {
        let op = HistPercentOp::<F32>::new(0.5).unwrap();
        let region = Region::new(-8, 3, 21, 1);
        assert_eq!(op.demand_hint(), DemandHint::Any);
        assert_eq!(op.required_input_region(&region), region);
        op.start();
    }

    proptest! {
        #[test]
        fn hist_percent_output_is_first_bin_meeting_threshold(
            increments in proptest::collection::vec(1u16..=32u16, 1..=32),
            percent in 0.0f64..=1.0f64,
        ) {
            let mut running = 0.0f32;
            let cumulative = increments
                .into_iter()
                .map(|value| {
                    running += f32::from(value);
                    running
                })
                .collect::<Vec<_>>();
            let idx = run_op(&cumulative, percent) as usize;
            let threshold = f64::from(*cumulative.last().unwrap_or(&0.0)) * percent;
            prop_assert!(idx < cumulative.len());
            prop_assert!(f64::from(cumulative[idx]) >= threshold);
            if idx > 0 {
                prop_assert!(f64::from(cumulative[idx - 1]) < threshold);
            }
        }
    }
}
