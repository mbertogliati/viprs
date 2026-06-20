use std::{any::Any, marker::PhantomData};

use bytemuck::{Pod, cast_slice, cast_slice_mut};

use viprs_core::{
    format::{BandFormat, BandFormatId},
    image::{DemandHint, Region},
    op::{DynOperation, NodeSpec},
};

/// Select pixels from one of N case images using a single-band U8 index image.
///
/// # Examples
/// ```ignore
/// use viprs::domain::ops::histogram::case::CaseOp;
///
/// let op = CaseOp::new(/* operation parameters */);
/// // Run `op` through a compiled image pipeline.
/// ```
pub struct CaseOp<F: BandFormat> {
    case_bands: Box<[u32]>,
    output_bands: u32,
    _phantom: PhantomData<F>,
}

impl<F: BandFormat> CaseOp<F> {
    #[must_use]
    /// Creates a new `CaseOp`.
    pub fn new(case_bands: Vec<u32>) -> Self {
        debug_assert!(
            !case_bands.is_empty() && case_bands.len() <= 256,
            "CaseOp: case count must be in 1..=256"
        );

        let output_bands = case_bands.iter().copied().max().unwrap_or(1);
        debug_assert!(
            case_bands
                .iter()
                .all(|&bands| bands == 1 || bands == output_bands),
            "CaseOp: each case must be single-band or match the output band count"
        );

        Self {
            case_bands: case_bands.into_boxed_slice(),
            output_bands,
            _phantom: PhantomData,
        }
    }

    #[must_use]
    /// Returns or performs choice count.
    pub fn choice_count(&self) -> usize {
        self.case_bands.len()
    }
}

impl<F> DynOperation for CaseOp<F>
where
    F: BandFormat,
    F::Sample: Pod + Copy,
{
    fn input_format(&self) -> BandFormatId {
        BandFormatId::U8
    }

    fn output_format(&self) -> BandFormatId {
        F::ID
    }

    fn bands(&self) -> u32 {
        self.output_bands
    }

    fn demand_hint(&self) -> DemandHint {
        DemandHint::ThinStrip
    }

    fn input_slot_count(&self) -> usize {
        1 + self.choice_count()
    }

    fn input_format_slot(&self, slot: usize) -> BandFormatId {
        match slot {
            0 => BandFormatId::U8,
            1.. => F::ID,
        }
    }

    fn input_bands_slot(&self, slot: usize) -> u32 {
        match slot {
            0 => 1,
            1.. => self.case_bands[slot - 1],
        }
    }

    fn required_input_region_slot(&self, output: &Region, slot: usize) -> Region {
        debug_assert!(slot < self.input_slot_count());
        *output
    }

    fn required_input_region(&self, output: &Region) -> Region {
        *output
    }

    fn node_spec(&self, tile_w: u32, tile_h: u32) -> NodeSpec {
        NodeSpec::identity(tile_w, tile_h)
    }

    fn dyn_start(&self) -> Box<dyn Any + Send> {
        Box::new(())
    }

    fn dyn_process_region(
        &self,
        _state: &mut dyn Any,
        _input: &[u8],
        _output: &mut [u8],
        _input_region: Region,
        _output_region: Region,
    ) {
        debug_assert!(
            false,
            "CaseOp: dyn_process_region called on a multi-input node — use dyn_process_region_multi"
        );
    }

    fn dyn_process_region_multi(
        &self,
        _state: &mut dyn Any,
        inputs: &[&[u8]],
        output: &mut [u8],
        input_regions: &[Region],
        output_region: Region,
    ) {
        debug_assert_eq!(inputs.len(), self.input_slot_count());
        debug_assert_eq!(input_regions.len(), self.input_slot_count());
        debug_assert!(input_regions.iter().all(|region| *region == output_region));

        let Some((&index, cases)) = inputs.split_first() else {
            return;
        };

        let output_samples = cast_slice_mut::<u8, F::Sample>(output);
        let pixel_count = output_region.pixel_count();
        let output_bands = self.output_bands as usize;
        debug_assert_eq!(index.len(), pixel_count);
        debug_assert_eq!(output_samples.len(), pixel_count * output_bands);

        for (px, &choice) in index.iter().take(pixel_count).enumerate() {
            let case_idx = usize::from(choice).min(self.choice_count() - 1);
            let selected_bands = self.case_bands[case_idx] as usize;
            let selected = cast_slice::<u8, F::Sample>(cases[case_idx]);
            let dst_base = px * output_bands;

            if selected_bands == 1 {
                let value = selected[px];
                output_samples[dst_base..dst_base + output_bands].fill(value);
            } else {
                let src_base = px * output_bands;
                output_samples[dst_base..dst_base + output_bands]
                    .copy_from_slice(&selected[src_base..src_base + output_bands]);
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use bytemuck::{cast_slice, cast_slice_mut};
    use proptest::prelude::*;
    use viprs_core::format::{F32, U8};

    fn run_case_u8(index: &[u8], cases: &[&[u8]], case_bands: Vec<u32>) -> Vec<u8> {
        let op = CaseOp::<U8>::new(case_bands);
        let region = Region::new(0, 0, index.len() as u32, 1);
        let input_regions = vec![region; 1 + cases.len()];
        let mut state = op.dyn_start();
        let mut output = vec![0u8; index.len() * op.bands() as usize];
        let mut inputs = Vec::with_capacity(1 + cases.len());
        inputs.push(index);
        inputs.extend_from_slice(cases);
        op.dyn_process_region_multi(state.as_mut(), &inputs, &mut output, &input_regions, region);
        output
    }

    #[test]
    fn zero_index_selects_first_case() {
        let output = run_case_u8(&[0, 0], &[&[10, 11], &[20, 21]], vec![1, 1]);
        assert_eq!(output, vec![10, 11]);
    }

    #[test]
    fn out_of_range_index_clamps_to_last_case() {
        let output = run_case_u8(&[7, 255], &[&[10, 11], &[20, 21]], vec![1, 1]);
        assert_eq!(output, vec![20, 21]);
    }

    #[test]
    fn single_band_case_broadcasts_to_output_bands() {
        let output = run_case_u8(&[0, 1], &[&[10, 20], &[1, 2, 3, 4, 5, 6]], vec![1, 3]);
        assert_eq!(output, vec![10, 10, 10, 4, 5, 6]);
    }

    #[test]
    fn metadata_reports_index_and_case_slots() {
        let op = CaseOp::<F32>::new(vec![1, 3, 3]);

        assert_eq!(op.input_slot_count(), 4);
        assert_eq!(op.input_format(), BandFormatId::U8);
        assert_eq!(op.output_format(), BandFormatId::F32);
        assert_eq!(op.input_format_slot(0), BandFormatId::U8);
        assert_eq!(op.input_format_slot(1), BandFormatId::F32);
        assert_eq!(op.input_bands_slot(0), 1);
        assert_eq!(op.input_bands_slot(1), 1);
        assert_eq!(op.input_bands_slot(2), 3);
        assert_eq!(op.bands(), 3);
        assert_eq!(op.demand_hint(), DemandHint::ThinStrip);
        assert_eq!(
            op.required_input_region(&Region::new(2, 3, 4, 5)),
            Region::new(2, 3, 4, 5)
        );
        assert_eq!(
            op.required_input_region_slot(&Region::new(2, 3, 4, 5), 2),
            Region::new(2, 3, 4, 5)
        );
        assert_eq!(op.node_spec(4, 5), NodeSpec::identity(4, 5));
    }

    #[test]
    fn f32_multi_band_case_copies_selected_pixels_without_repacking() {
        let op = CaseOp::<F32>::new(vec![1, 2]);
        let region = Region::new(0, 0, 2, 1);
        let index = [0u8, 1];
        let case0 = [1.5f32, 9.0];
        let case1 = [2.0f32, 3.0, 4.0, 5.0];
        let inputs = [index.as_slice(), cast_slice(&case0), cast_slice(&case1)];
        let input_regions = [region, region, region];
        let mut output = [0.0f32; 4];
        let mut state = op.dyn_start();

        op.dyn_process_region_multi(
            state.as_mut(),
            &inputs,
            cast_slice_mut(&mut output),
            &input_regions,
            region,
        );

        assert_eq!(output, [1.5, 1.5, 4.0, 5.0]);
    }

    #[test]
    #[should_panic(expected = "dyn_process_region called on a multi-input node")]
    fn single_input_entrypoint_panics_for_multi_input_case_op() {
        let op = CaseOp::<U8>::new(vec![1, 1]);
        let region = Region::new(0, 0, 1, 1);
        let mut state = op.dyn_start();
        let mut output = [0u8; 1];

        op.dyn_process_region(state.as_mut(), &[0], &mut output, region, region);
    }

    proptest! {
        #[test]
        fn all_zero_indices_preserve_first_case(index_len in 1usize..=32, first in proptest::collection::vec(any::<u8>(), 1..=32), second in proptest::collection::vec(any::<u8>(), 1..=32)) {
            let len = index_len.min(first.len()).min(second.len());
            let zeros = vec![0u8; len];
            let output = run_case_u8(&zeros, &[&first[..len], &second[..len]], vec![1, 1]);
            prop_assert_eq!(output, first[..len].to_vec());
        }
    }
}
