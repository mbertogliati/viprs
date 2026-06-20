use std::marker::PhantomData;

use bytemuck::Pod;

use viprs_core::{
    format::{BandFormat, U8},
    image::{DemandHint, Region, Tile, TileMut},
    op::Op,
    shared_ops::sample_conv::ToF64,
};

/// Test whether histogram bins are monotonically non-decreasing.
///
/// # Examples
/// ```ignore
/// use viprs::domain::ops::histogram::hist_ismonotonic::HistIsMonotonicOp;
///
/// let op = HistIsMonotonicOp::new(/* operation parameters */);
/// // Run `op` through a compiled image pipeline.
/// ```
pub struct HistIsMonotonicOp<F: BandFormat> {
    _phantom: PhantomData<F>,
}

impl<F: BandFormat> HistIsMonotonicOp<F> {
    #[must_use]
    /// Creates a new `HistIsMonotonicOp`.
    pub const fn new() -> Self {
        Self {
            _phantom: PhantomData,
        }
    }
}

impl<F: BandFormat> Default for HistIsMonotonicOp<F> {
    fn default() -> Self {
        Self::new()
    }
}

impl<F> Op for HistIsMonotonicOp<F>
where
    F: BandFormat,
    F::Sample: ToF64 + Pod,
{
    type Input = F;
    type Output = U8;
    type State = ();

    fn demand_hint(&self) -> DemandHint {
        DemandHint::Any
    }

    fn required_input_region(&self, output: &Region) -> Region {
        *output
    }

    fn start(&self) {}

    #[inline]
    fn process_region(&self, _state: &mut (), input: &Tile<F>, output: &mut TileMut<U8>) {
        let mut previous = None;
        let mut monotonic = true;

        for value in input.data.iter().map(|sample| sample.to_f64()) {
            if let Some(prev) = previous
                && value < prev
            {
                monotonic = false;
                break;
            }
            previous = Some(value);
        }

        output.data.fill(u8::from(monotonic));
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use proptest::prelude::*;
    use viprs_core::{format::F32, image::Region};

    fn run_op(input_data: &[f32]) -> u8 {
        let input_region = Region::new(0, 0, input_data.len() as u32, 1);
        let output_region = Region::new(0, 0, 1, 1);
        let input = Tile::<F32>::new(input_region, 1, input_data);
        let mut output_data = vec![0u8; 1];
        let mut output = TileMut::<U8>::new(output_region, 1, &mut output_data);
        let mut state = ();
        HistIsMonotonicOp::<F32>::new().process_region(&mut state, &input, &mut output);
        output_data[0]
    }

    #[test]
    fn hist_ismonotonic_true_for_non_decreasing_histogram() {
        assert_eq!(run_op(&[1.0, 2.0, 3.0, 4.0]), 1);
    }

    #[test]
    fn hist_ismonotonic_false_for_descending_step() {
        assert_eq!(run_op(&[1.0, 3.0, 2.0, 4.0]), 0);
    }

    #[test]
    fn hist_ismonotonic_empty_input_is_true() {
        assert_eq!(run_op(&[]), 1);
    }

    #[test]
    fn hist_ismonotonic_region_contract_is_identity() {
        let op = HistIsMonotonicOp::<F32>::new();
        let region = Region::new(12, -5, 8, 1);
        assert_eq!(op.demand_hint(), DemandHint::Any);
        assert_eq!(op.required_input_region(&region), region);
        op.start();
    }

    proptest! {
        #[test]
        fn hist_ismonotonic_sorted_input_is_true(values in proptest::collection::vec(0u16..=1024u16, 1..=64)) {
            let mut sorted = values;
            sorted.sort_unstable();
            let input = sorted.into_iter().map(f32::from).collect::<Vec<_>>();
            prop_assert_eq!(run_op(&input), 1);
        }
    }
}
