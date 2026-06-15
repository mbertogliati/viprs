use std::any::Any;

use bytemuck::Pod;

use crate::domain::{
    format::{BandFormat, BandFormatId},
    image::{DemandHint, Region},
    op::{DynOperation, NodeSpec},
    ops::{
        mosaicing::merge::MergeV,
        resample::sample_conv::{FromF64, ToF64},
    },
};

/// Libvips-style top-bottom merge with a feathered seam over the vertical overlap.
pub struct TbMerge<F: BandFormat> {
    inner: MergeV<F>,
}

impl<F: BandFormat> TbMerge<F> {
    #[allow(clippy::too_many_arguments)]
    #[must_use]
    /// Creates a new `TbMerge`.
    pub const fn new(
        ref_width: u32,
        ref_height: u32,
        sec_width: u32,
        sec_height: u32,
        dx: i32,
        dy: i32,
        overlap_height: u32,
        bands: u32,
    ) -> Self {
        Self {
            inner: MergeV::new(
                ref_width,
                ref_height,
                sec_width,
                sec_height,
                dx,
                dy,
                overlap_height,
                bands,
            ),
        }
    }

    #[must_use]
    /// Returns or performs output width.
    pub fn output_width(&self) -> u32 {
        self.inner.output_width()
    }

    #[must_use]
    /// Returns or performs output height.
    pub fn output_height(&self) -> u32 {
        self.inner.output_height()
    }
}

impl<F> DynOperation for TbMerge<F>
where
    F: BandFormat + Send + Sync,
    F::Sample: Copy + Pod + ToF64 + FromF64 + Send,
{
    fn input_format(&self) -> BandFormatId {
        self.inner.input_format()
    }

    fn output_format(&self) -> BandFormatId {
        self.inner.output_format()
    }

    fn bands(&self) -> u32 {
        self.inner.bands()
    }

    fn demand_hint(&self) -> DemandHint {
        self.inner.demand_hint()
    }

    fn input_slot_count(&self) -> usize {
        self.inner.input_slot_count()
    }

    fn required_input_region_slot(&self, output: &Region, slot: usize) -> Region {
        self.inner.required_input_region_slot(output, slot)
    }

    fn required_input_region(&self, output: &Region) -> Region {
        self.inner.required_input_region(output)
    }

    fn output_width(&self, input_w: u32) -> u32 {
        <MergeV<F> as DynOperation>::output_width(&self.inner, input_w)
    }

    fn output_height(&self, input_h: u32) -> u32 {
        <MergeV<F> as DynOperation>::output_height(&self.inner, input_h)
    }

    fn node_spec(&self, tile_w: u32, tile_h: u32) -> NodeSpec {
        self.inner.node_spec(tile_w, tile_h)
    }

    fn dyn_start(&self) -> Box<dyn Any + Send> {
        self.inner.dyn_start()
    }

    fn dyn_process_region(
        &self,
        state: &mut dyn Any,
        input: &[u8],
        output: &mut [u8],
        input_region: Region,
        output_region: Region,
    ) {
        self.inner
            .dyn_process_region(state, input, output, input_region, output_region);
    }

    fn dyn_process_region_multi(
        &self,
        state: &mut dyn Any,
        inputs: &[&[u8]],
        output: &mut [u8],
        input_regions: &[Region],
        output_region: Region,
    ) {
        self.inner
            .dyn_process_region_multi(state, inputs, output, input_regions, output_region);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::domain::{format::U8, op::DynOperation};
    use proptest::prelude::*;

    fn run(op: &TbMerge<U8>, reference: &[u8], secondary: &[u8], output_region: Region) -> Vec<u8> {
        let inputs: &[&[u8]] = &[reference, secondary];
        let input_regions = [
            op.required_input_region_slot(&output_region, 0),
            op.required_input_region_slot(&output_region, 1),
        ];
        let mut output = vec![0u8; output_region.pixel_count() * op.bands() as usize];
        let mut state = op.dyn_start();
        op.dyn_process_region_multi(
            state.as_mut(),
            inputs,
            &mut output,
            &input_regions,
            output_region,
        );
        output
    }

    proptest! {
        #[test]
        fn same_colour_half_overlap_stays_constant(
            width in 1u32..8,
            height in 2u32..16,
            colour in any::<u8>(),
        ) {
            let overlap = height / 2;
            let op = TbMerge::<U8>::new(width, height, width, height, 0, -(height as i32 - overlap as i32), overlap, 1);
            let len = width as usize * height as usize;
            let reference = vec![colour; len];
            let secondary = vec![colour; len];
            let output = run(&op, &reference, &secondary, Region::new(0, 0, width, height + height - overlap));

            prop_assert!(output.into_iter().all(|sample| sample == colour));
        }
    }

    #[test]
    fn zero_overlap_is_hard_cut() {
        let op = TbMerge::<U8>::new(1, 4, 1, 4, 0, -4, 0, 1);
        let output = run(&op, &[1, 2, 3, 4], &[5, 6, 7, 8], Region::new(0, 0, 1, 8));
        assert_eq!(output, vec![1, 2, 3, 4, 5, 6, 7, 8]);
    }

    #[test]
    fn dyn_contract_delegates() {
        let op = TbMerge::<U8>::new(2, 4, 2, 4, 0, -2, 2, 1);
        let output_region = Region::new(0, 0, 2, 6);

        assert_eq!(op.input_format(), crate::domain::format::BandFormatId::U8);
        assert_eq!(op.output_format(), crate::domain::format::BandFormatId::U8);
        assert_eq!(op.bands(), 1);
        assert_eq!(op.demand_hint(), DemandHint::ThinStrip);
        assert_eq!(op.input_slot_count(), 2);
        assert_eq!(op.output_width(), 2);
        assert_eq!(op.output_height(), 6);
        assert_eq!(<TbMerge<U8> as DynOperation>::output_width(&op, 99), 2);
        assert_eq!(<TbMerge<U8> as DynOperation>::output_height(&op, 99), 6);
        assert_eq!(op.required_input_region(&output_region), output_region);
        assert_eq!(op.node_spec(2, 3), NodeSpec::identity(2, 3));
        assert_eq!(
            op.required_input_region_slot(&output_region, 9),
            Region::new(0, 0, 0, 0)
        );
    }

    #[test]
    #[should_panic(expected = "MergeV: dyn_process_region called on a 2-input node")]
    fn dyn_process_region_panics_for_two_input_wrapper() {
        let op = TbMerge::<U8>::new(2, 4, 2, 4, 0, -2, 2, 1);
        let mut output = vec![0u8; 2];
        let mut state = op.dyn_start();
        op.dyn_process_region(
            state.as_mut(),
            &[4, 5, 6],
            &mut output,
            Region::new(0, 0, 3, 1),
            Region::new(0, 0, 2, 1),
        );
    }
}
