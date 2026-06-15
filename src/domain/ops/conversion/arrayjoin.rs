use std::any::Any;

use crate::domain::{
    format::BandFormatId,
    image::{DemandHint, Region},
    op::{DynOperation, NodeSpec},
};

/// Join an array of equally-sized images by concatenating their bands.
///
/// # Examples
/// ```ignore
/// use viprs::domain::ops::conversion::arrayjoin::ArrayJoinOp;
///
/// let op = ArrayJoinOp::new(/* operation parameters */);
/// // Run `op` through a compiled image pipeline.
/// ```
pub struct ArrayJoinOp {
    input_count: usize,
    input_bands: u32,
    format: BandFormatId,
}

impl ArrayJoinOp {
    #[must_use]
    /// Creates a new `ArrayJoinOp`.
    pub fn new(input_count: usize, input_bands: u32, format: BandFormatId) -> Self {
        debug_assert!(input_count > 0, "ArrayJoinOp requires at least one input");
        debug_assert!(
            input_bands > 0,
            "ArrayJoinOp requires at least one band per input"
        );
        Self {
            input_count,
            input_bands,
            format,
        }
    }

    #[must_use]
    /// Returns or performs output bands.
    pub const fn output_bands(&self) -> u32 {
        self.input_count as u32 * self.input_bands
    }
}

impl DynOperation for ArrayJoinOp {
    fn input_format(&self) -> BandFormatId {
        self.format
    }

    fn output_format(&self) -> BandFormatId {
        self.format
    }

    fn bands(&self) -> u32 {
        self.output_bands()
    }

    fn demand_hint(&self) -> DemandHint {
        DemandHint::ThinStrip
    }

    fn input_slot_count(&self) -> usize {
        self.input_count
    }

    fn required_input_region_slot(&self, output: &Region, _slot: usize) -> Region {
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
        input: &[u8],
        output: &mut [u8],
        _input_region: Region,
        _output_region: Region,
    ) {
        debug_assert!(
            false,
            "ArrayJoinOp: dyn_process_region called on a multi-input node — use dyn_process_region_multi"
        );
        let len = output.len().min(input.len());
        output[..len].copy_from_slice(&input[..len]);
    }

    #[inline]
    fn dyn_process_region_multi(
        &self,
        _state: &mut dyn Any,
        inputs: &[&[u8]],
        output: &mut [u8],
        _input_regions: &[Region],
        output_region: Region,
    ) {
        debug_assert_eq!(inputs.len(), self.input_count);

        let sample_size = sample_size_bytes(self.format);
        let input_bands = self.input_bands as usize;
        let output_bands = self.output_bands() as usize;
        let pixel_count = output_region.pixel_count();
        let input_stride = input_bands * sample_size;

        for input in inputs {
            debug_assert_eq!(input.len(), pixel_count * input_stride);
        }
        debug_assert_eq!(output.len(), pixel_count * output_bands * sample_size);

        for px in 0..pixel_count {
            let dst_base = px * output_bands * sample_size;
            for (slot, input) in inputs.iter().enumerate() {
                let src_base = px * input_stride;
                let slot_dst = dst_base + slot * input_stride;
                output[slot_dst..slot_dst + input_stride]
                    .copy_from_slice(&input[src_base..src_base + input_stride]);
            }
        }
    }
}

const fn sample_size_bytes(format: BandFormatId) -> usize {
    match format {
        BandFormatId::U8 => 1,
        BandFormatId::U16 | BandFormatId::I16 => 2,
        BandFormatId::U32 | BandFormatId::I32 | BandFormatId::F32 => 4,
        BandFormatId::F64 => 8,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::domain::ops::conversion::BandJoin;

    #[test]
    fn join_two_single_band_images_matches_bandjoin() {
        let arrayjoin = ArrayJoinOp::new(2, 1, BandFormatId::U8);
        let bandjoin = BandJoin::new(1, 1, BandFormatId::U8);
        let inputs: &[&[u8]] = &[&[10, 50], &[20, 60]];
        let regions = [Region::new(0, 0, 2, 1); 2];
        let output_region = Region::new(0, 0, 2, 1);

        let mut arrayjoin_out = [0u8; 4];
        let mut bandjoin_out = [0u8; 4];

        let mut arrayjoin_state = arrayjoin.dyn_start();
        arrayjoin.dyn_process_region_multi(
            arrayjoin_state.as_mut(),
            inputs,
            &mut arrayjoin_out,
            &regions,
            output_region,
        );

        let mut bandjoin_state = bandjoin.dyn_start();
        bandjoin.dyn_process_region_multi(
            bandjoin_state.as_mut(),
            inputs,
            &mut bandjoin_out,
            &regions,
            output_region,
        );

        assert_eq!(arrayjoin_out, bandjoin_out);
        assert_eq!(arrayjoin_out, [10, 20, 50, 60]);
    }

    #[test]
    fn output_bands_scales_with_input_count() {
        let op = ArrayJoinOp::new(3, 2, BandFormatId::U8);
        assert_eq!(op.input_slot_count(), 3);
        assert_eq!(op.output_bands(), 6);
        assert_eq!(op.bands(), 6);
    }

    #[test]
    fn metadata_reports_identity_geometry_for_each_slot() {
        let op = ArrayJoinOp::new(2, 3, BandFormatId::F64);
        let region = Region::new(4, 5, 6, 7);

        assert_eq!(op.input_format(), BandFormatId::F64);
        assert_eq!(op.output_format(), BandFormatId::F64);
        assert_eq!(op.demand_hint(), DemandHint::ThinStrip);
        assert_eq!(op.required_input_region(&region), region);
        assert_eq!(op.required_input_region_slot(&region, 0), region);
        assert_eq!(op.required_input_region_slot(&region, 1), region);
        assert_eq!(op.node_spec(8, 9), NodeSpec::identity(8, 9));
    }

    #[test]
    fn sample_size_bytes_covers_all_formats() {
        assert_eq!(sample_size_bytes(BandFormatId::U8), 1);
        assert_eq!(sample_size_bytes(BandFormatId::U16), 2);
        assert_eq!(sample_size_bytes(BandFormatId::I16), 2);
        assert_eq!(sample_size_bytes(BandFormatId::U32), 4);
        assert_eq!(sample_size_bytes(BandFormatId::I32), 4);
        assert_eq!(sample_size_bytes(BandFormatId::F32), 4);
        assert_eq!(sample_size_bytes(BandFormatId::F64), 8);
    }

    #[test]
    fn join_two_u16_images_preserves_per_slot_band_order() {
        let op = ArrayJoinOp::new(2, 2, BandFormatId::U16);
        let region = Region::new(0, 0, 2, 1);
        let first = [1u16, 2, 5, 6];
        let second = [3u16, 4, 7, 8];
        let inputs = [
            bytemuck::cast_slice::<u16, u8>(&first),
            bytemuck::cast_slice::<u16, u8>(&second),
        ];
        let mut output = [0u16; 8];
        let mut state = op.dyn_start();

        op.dyn_process_region_multi(
            state.as_mut(),
            &inputs,
            bytemuck::cast_slice_mut::<u16, u8>(&mut output),
            &[region, region],
            region,
        );

        assert_eq!(output, [1, 2, 3, 4, 5, 6, 7, 8]);
    }
}
