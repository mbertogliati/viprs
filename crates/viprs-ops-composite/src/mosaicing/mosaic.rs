use std::{any::Any, marker::PhantomData};

use bytemuck::Pod;

use crate::mosaicing::merge::{MergeH, MergeV};

use viprs_core::{
    format::BandFormat,
    image::{DemandHint, Region},
    op::{DynOperation, NodeSpec},
    shared_ops::sample_conv::{FromF64, ToF64},
};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
/// Enumerates the available mosaic direction values.
pub enum MosaicDirection {
    /// Uses the `Horizontal` variant of `MosaicDirection`.
    Horizontal,
    /// Uses the `Vertical` variant of `MosaicDirection`.
    Vertical,
}

/// Base 2-image mosaic: converts tie-points into a merge displacement and
/// delegates the actual stitching to `MergeH` / `MergeV`.
pub struct Mosaic<F: BandFormat> {
    direction: MosaicDirection,
    ref_width: u32,
    ref_height: u32,
    sec_width: u32,
    sec_height: u32,
    xref: i32,
    yref: i32,
    xsec: i32,
    ysec: i32,
    blend_width: u32,
    bands: u32,
    _format: PhantomData<F>,
}

impl<F: BandFormat> Mosaic<F> {
    #[allow(clippy::too_many_arguments)]
    #[must_use]
    /// Creates a new `Mosaic`.
    pub const fn new(
        direction: MosaicDirection,
        ref_width: u32,
        ref_height: u32,
        sec_width: u32,
        sec_height: u32,
        xref: i32,
        yref: i32,
        xsec: i32,
        ysec: i32,
        blend_width: u32,
        bands: u32,
    ) -> Self {
        Self {
            direction,
            ref_width,
            ref_height,
            sec_width,
            sec_height,
            xref,
            yref,
            xsec,
            ysec,
            blend_width,
            bands,
            _format: PhantomData,
        }
    }

    const fn dx(&self) -> i32 {
        self.xsec - self.xref
    }

    const fn dy(&self) -> i32 {
        self.ysec - self.yref
    }

    const fn merge_h(&self) -> MergeH<F> {
        MergeH::new(
            self.ref_width,
            self.ref_height,
            self.sec_width,
            self.sec_height,
            self.dx(),
            self.dy(),
            self.blend_width,
            self.bands,
        )
    }

    const fn merge_v(&self) -> MergeV<F> {
        MergeV::new(
            self.ref_width,
            self.ref_height,
            self.sec_width,
            self.sec_height,
            self.dx(),
            self.dy(),
            self.blend_width,
            self.bands,
        )
    }

    #[must_use]
    /// Returns or performs output width.
    pub fn output_width(&self) -> u32
    where
        F::Sample: Copy + Pod + ToF64 + FromF64,
    {
        match self.direction {
            MosaicDirection::Horizontal => self.merge_h().output_width(),
            MosaicDirection::Vertical => self.merge_v().output_width(),
        }
    }

    #[must_use]
    /// Returns or performs output height.
    pub fn output_height(&self) -> u32
    where
        F::Sample: Copy + Pod + ToF64 + FromF64,
    {
        match self.direction {
            MosaicDirection::Horizontal => self.merge_h().output_height(),
            MosaicDirection::Vertical => self.merge_v().output_height(),
        }
    }
}

impl<F> DynOperation for Mosaic<F>
where
    F: BandFormat + Send + Sync,
    F::Sample: Copy + Pod + ToF64 + FromF64 + Send,
{
    fn input_format(&self) -> viprs_core::format::BandFormatId {
        F::ID
    }

    fn output_format(&self) -> viprs_core::format::BandFormatId {
        F::ID
    }

    fn bands(&self) -> u32 {
        self.bands
    }

    fn demand_hint(&self) -> DemandHint {
        DemandHint::ThinStrip
    }

    fn input_slot_count(&self) -> usize {
        2
    }

    fn required_input_region_slot(&self, output: &Region, slot: usize) -> Region {
        match self.direction {
            MosaicDirection::Horizontal => self.merge_h().required_input_region_slot(output, slot),
            MosaicDirection::Vertical => self.merge_v().required_input_region_slot(output, slot),
        }
    }

    fn required_input_region(&self, output: &Region) -> Region {
        *output
    }

    fn output_width(&self, _input_w: u32) -> u32 {
        Self::output_width(self)
    }

    fn output_height(&self, _input_h: u32) -> u32 {
        Self::output_height(self)
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
        input: &[u8],
        output: &mut [u8],
        _input_region: Region,
        _output_region: Region,
    ) {
        debug_assert!(
            false,
            "Mosaic: dyn_process_region called on a 2-input node — use dyn_process_region_multi"
        );
        let len = output.len().min(input.len());
        output[..len].copy_from_slice(&input[..len]);
    }

    #[inline]
    fn dyn_process_region_multi(
        &self,
        state: &mut dyn Any,
        inputs: &[&[u8]],
        output: &mut [u8],
        input_regions: &[Region],
        output_region: Region,
    ) {
        match self.direction {
            MosaicDirection::Horizontal => self.merge_h().dyn_process_region_multi(
                state,
                inputs,
                output,
                input_regions,
                output_region,
            ),
            MosaicDirection::Vertical => self.merge_v().dyn_process_region_multi(
                state,
                inputs,
                output,
                input_regions,
                output_region,
            ),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use viprs_core::{format::BandFormatId, format::U8, image::DemandHint, op::DynOperation};

    fn run_mosaic(
        op: &Mosaic<U8>,
        reference: &[u8],
        secondary: &[u8],
        output_region: Region,
    ) -> Vec<u8> {
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

    #[test]
    fn mosaic_horizontal_delegates_to_merge_offset() {
        let op = Mosaic::<U8>::new(MosaicDirection::Horizontal, 4, 1, 4, 1, 1, 0, 0, 0, 3, 1);
        let output = run_mosaic(
            &op,
            &[10, 20, 30, 40],
            &[100, 110, 120, 130],
            Region::new(0, 0, 5, 1),
        );
        assert_eq!(output[0], 10);
        assert!(output[2] > 30 && output[2] < 110);
        assert_eq!(output[4], 130);
    }

    #[test]
    fn mosaic_vertical_output_dimensions_follow_tie_points() {
        let op = Mosaic::<U8>::new(MosaicDirection::Vertical, 3, 4, 3, 4, 1, 3, 1, 0, 2, 1);
        assert_eq!(op.output_width(), 3);
        assert_eq!(op.output_height(), 7);
    }

    #[test]
    fn mosaic_vertical_delegates_to_merge_offset() {
        let op = Mosaic::<U8>::new(MosaicDirection::Vertical, 1, 4, 1, 4, 0, 1, 0, 0, 3, 1);
        let output = run_mosaic(
            &op,
            &[10, 20, 30, 40],
            &[100, 110, 120, 130],
            Region::new(0, 0, 1, 5),
        );
        assert_eq!(output[0], 10);
        assert!(output[2] > 30 && output[2] < 110);
        assert_eq!(output[4], 130);
    }

    #[test]
    fn mosaic_dyn_metadata_matches_merge_delegation() {
        let horizontal =
            Mosaic::<U8>::new(MosaicDirection::Horizontal, 4, 3, 4, 3, 2, 1, 0, 1, 2, 2);
        let vertical = Mosaic::<U8>::new(MosaicDirection::Vertical, 3, 4, 3, 4, 1, 2, 1, 0, 2, 2);
        let horizontal_dyn: &dyn DynOperation = &horizontal;
        let vertical_dyn: &dyn DynOperation = &vertical;
        let horizontal_region =
            Region::new(0, 0, horizontal.output_width(), horizontal.output_height());
        let vertical_region = Region::new(0, 0, vertical.output_width(), vertical.output_height());

        assert_eq!(horizontal_dyn.input_format(), BandFormatId::U8);
        assert_eq!(horizontal_dyn.output_format(), BandFormatId::U8);
        assert_eq!(horizontal_dyn.bands(), 2);
        assert_eq!(horizontal_dyn.demand_hint(), DemandHint::ThinStrip);
        assert_eq!(horizontal_dyn.input_slot_count(), 2);
        assert_eq!(
            horizontal_dyn.required_input_region_slot(&horizontal_region, 0),
            Region::new(0, 0, 4, 3)
        );
        assert_eq!(
            horizontal_dyn.required_input_region_slot(&horizontal_region, 1),
            Region::new(0, 0, 4, 3)
        );
        assert_eq!(
            horizontal_dyn.required_input_region(&horizontal_region),
            horizontal_region
        );
        assert_eq!(horizontal_dyn.output_width(1), horizontal.output_width());
        assert_eq!(horizontal_dyn.output_height(1), horizontal.output_height());
        assert_eq!(horizontal_dyn.node_spec(32, 16), NodeSpec::identity(32, 16));

        assert_eq!(vertical_dyn.input_format(), BandFormatId::U8);
        assert_eq!(vertical_dyn.output_format(), BandFormatId::U8);
        assert_eq!(vertical_dyn.bands(), 2);
        assert_eq!(vertical_dyn.demand_hint(), DemandHint::ThinStrip);
        assert_eq!(vertical_dyn.input_slot_count(), 2);
        assert_eq!(
            vertical_dyn.required_input_region_slot(&vertical_region, 0),
            Region::new(0, 0, 3, 4)
        );
        assert_eq!(
            vertical_dyn.required_input_region_slot(&vertical_region, 1),
            Region::new(0, 0, 3, 4)
        );
        assert_eq!(
            vertical_dyn.required_input_region(&vertical_region),
            vertical_region
        );
        assert_eq!(vertical_dyn.output_width(1), vertical.output_width());
        assert_eq!(vertical_dyn.output_height(1), vertical.output_height());
        assert_eq!(vertical_dyn.node_spec(16, 32), NodeSpec::identity(16, 32));
    }

    #[test]
    #[should_panic(expected = "Mosaic: dyn_process_region called on a 2-input node")]
    fn mosaic_single_input_path_panics_for_multi_input_nodes() {
        let op = Mosaic::<U8>::new(MosaicDirection::Horizontal, 1, 1, 1, 1, 0, 0, 0, 0, 1, 1);
        let mut state = op.dyn_start();
        let mut output = [0u8; 1];
        op.dyn_process_region(
            state.as_mut(),
            &[0u8; 1],
            &mut output,
            Region::new(0, 0, 1, 1),
            Region::new(0, 0, 1, 1),
        );
    }
}
