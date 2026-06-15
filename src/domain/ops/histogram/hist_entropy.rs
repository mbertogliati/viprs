use std::marker::PhantomData;

use bytemuck::Pod;

use crate::domain::{
    format::{BandFormat, F64},
    image::{DemandHint, Region, Tile, TileMut},
    op::Op,
    ops::resample::sample_conv::ToF64,
};

/// Estimate histogram entropy as `-sum(p * log2(p))`.
///
/// # Examples
/// ```ignore
/// use viprs::domain::ops::histogram::hist_entropy::HistEntropyOp;
///
/// let op = HistEntropyOp::new(/* operation parameters */);
/// // Run `op` through a compiled image pipeline.
/// ```
pub struct HistEntropyOp<F: BandFormat> {
    _phantom: PhantomData<F>,
}

impl<F: BandFormat> HistEntropyOp<F> {
    #[must_use]
    /// Creates a new `HistEntropyOp`.
    pub const fn new() -> Self {
        Self {
            _phantom: PhantomData,
        }
    }
}

impl<F: BandFormat> Default for HistEntropyOp<F> {
    fn default() -> Self {
        Self::new()
    }
}

impl<F> Op for HistEntropyOp<F>
where
    F: BandFormat,
    F::Sample: ToF64 + Pod,
{
    type Input = F;
    type Output = F64;
    type State = ();

    fn demand_hint(&self) -> DemandHint {
        DemandHint::Any
    }

    fn required_input_region(&self, output: &Region) -> Region {
        *output
    }

    fn start(&self) {}

    #[inline]
    fn process_region(&self, _state: &mut (), input: &Tile<F>, output: &mut TileMut<F64>) {
        let total = input
            .data
            .iter()
            .map(|sample| sample.to_f64().max(0.0))
            .sum::<f64>();

        let entropy = if total > 0.0 {
            -input
                .data
                .iter()
                .map(|sample| sample.to_f64().max(0.0))
                .filter(|&value| value > 0.0)
                .map(|value| {
                    let probability = value / total;
                    probability * probability.log2()
                })
                .sum::<f64>()
        } else {
            0.0
        };

        output.data.fill(entropy);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::domain::{format::F32, image::Region};
    use proptest::prelude::*;

    fn run_op(input_data: &[f32]) -> f64 {
        let input_region = Region::new(0, 0, input_data.len() as u32, 1);
        let output_region = Region::new(0, 0, 1, 1);
        let input = Tile::<F32>::new(input_region, 1, input_data);
        let mut output_data = vec![0.0f64; 1];
        let mut output = TileMut::<F64>::new(output_region, 1, &mut output_data);
        let mut state = ();
        HistEntropyOp::<F32>::new().process_region(&mut state, &input, &mut output);
        output_data[0]
    }

    #[test]
    fn hist_entropy_uniform_histogram_is_log2_bin_count() {
        let entropy = run_op(&vec![1.0f32; 256]);
        assert!((entropy - 8.0).abs() < 1e-12);
    }

    #[test]
    fn hist_entropy_degenerate_histogram_is_zero() {
        let mut bins = vec![0.0f32; 256];
        bins[17] = 42.0;
        let entropy = run_op(&bins);
        assert_eq!(entropy, 0.0);
    }

    #[test]
    fn hist_entropy_ignores_negative_bins() {
        let entropy = run_op(&[-100.0, 1.0, 1.0, 0.0]);
        assert!((entropy - 1.0).abs() < 1e-12);
    }

    #[test]
    fn hist_entropy_region_contract_is_identity() {
        let op = HistEntropyOp::<F32>::new();
        let region = Region::new(2, -4, 17, 1);
        assert_eq!(op.demand_hint(), DemandHint::Any);
        assert_eq!(op.required_input_region(&region), region);
        op.start();
    }

    proptest! {
        #[test]
        fn hist_entropy_uniform_histogram_matches_log2(len in 1usize..=64usize, count in 1u16..=32u16) {
            let input = vec![f32::from(count); len];
            let entropy = run_op(&input);
            prop_assert!((entropy - (len as f64).log2()).abs() < 1e-12);
        }
    }
}
