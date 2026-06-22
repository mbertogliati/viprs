use std::any::Any;
use viprs_core::{
    format::BandFormatId,
    image::{DemandHint, Region},
    op::{DynOperation, NodeSpec},
};

/// Joins two images by concatenating their bands.
///
/// Input slot 0 has `a_bands` bands and input slot 1 has `b_bands` bands.
/// The output has `a_bands + b_bands` bands, with A's bands first.
///
/// Both inputs must have the same pixel format, width, and height.
///
/// # Multi-input design
///
/// `BandJoin` implements `DynOperation` directly (not via `OperationBridge`) because
/// it reads from two upstream input slots. `OperationBridge` only bridges single-input
/// `Op` implementations. The format is stored as `BandFormatId` and dispatched at
/// runtime inside `dyn_process_region_multi`.
///
/// When the DAG scheduler gains full multi-input slot support for arbitrary
/// `DynOperation` nodes, this type will automatically benefit with no API change.
pub struct BandJoin {
    a_bands: u32,
    b_bands: u32,
    format: BandFormatId,
}

impl BandJoin {
    /// Construct a `BandJoin` merge node.
    ///
    /// `a_bands` and `b_bands` are the channel counts of input slots 0 and 1.
    /// `format` is the shared sample format of both inputs and the output.
    #[must_use]
    pub const fn new(a_bands: u32, b_bands: u32, format: BandFormatId) -> Self {
        Self {
            a_bands,
            b_bands,
            format,
        }
    }

    /// The number of bands in the output image (`a_bands + b_bands`).
    #[must_use]
    pub const fn output_bands(&self) -> u32 {
        self.a_bands + self.b_bands
    }
}

impl DynOperation for BandJoin {
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
        2
    }

    fn required_input_region_slot(&self, output: &Region, _slot: usize) -> Region {
        // Both input slots cover the same spatial region as the output.
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
        // Single-input path should not be reached for a correctly compiled pipeline.
        // The scheduler calls dyn_process_region_multi for nodes with input_slot_count() == 2.
        // Provided only to satisfy the trait contract.
        debug_assert!(
            false,
            "BandJoin: dyn_process_region called on a 2-input node — \
             pipeline construction bug; use dyn_process_region_multi"
        );
        let len = output.len().min(input.len());
        output[..len].copy_from_slice(&input[..len]);
    }

    /// Interleave samples from two input tiles into the output.
    ///
    /// For each pixel, copies `a_bands` samples from slot 0 followed by
    /// `b_bands` samples from slot 1. Output layout is interleaved (RGBARGB…)
    /// matching libvips and the rest of vipers. No heap allocation: all work
    /// is done directly in the pre-allocated output slice.
    #[inline]
    fn dyn_process_region_multi(
        &self,
        _state: &mut dyn Any,
        inputs: &[&[u8]],
        output: &mut [u8],
        _input_regions: &[Region],
        output_region: Region,
    ) {
        debug_assert_eq!(inputs.len(), 2, "BandJoin: expected exactly 2 input slices");

        let (Some(&a_bytes), Some(&b_bytes)) = (inputs.first(), inputs.get(1)) else {
            debug_assert!(false, "BandJoin: missing input slices");
            return;
        };

        let sample_size = self.format.sample_size_bytes();
        let a_bands = self.a_bands as usize;
        let b_bands = self.b_bands as usize;
        let out_bands = a_bands + b_bands;
        let pixel_count = output_region.pixel_count();

        debug_assert_eq!(
            a_bytes.len(),
            pixel_count * a_bands * sample_size,
            "BandJoin: input[0] size mismatch"
        );
        debug_assert_eq!(
            b_bytes.len(),
            pixel_count * b_bands * sample_size,
            "BandJoin: input[1] size mismatch"
        );
        debug_assert_eq!(
            output.len(),
            pixel_count * out_bands * sample_size,
            "BandJoin: output size mismatch"
        );

        for px in 0..pixel_count {
            let a_src = px * a_bands * sample_size;
            let b_src = px * b_bands * sample_size;
            let dst = px * out_bands * sample_size;

            output[dst..dst + a_bands * sample_size]
                .copy_from_slice(&a_bytes[a_src..a_src + a_bands * sample_size]);
            output[dst + a_bands * sample_size..dst + out_bands * sample_size]
                .copy_from_slice(&b_bytes[b_src..b_src + b_bands * sample_size]);
        }
    }
}

/// Extension on `BandFormatId` for byte-level buffer sizing.
trait FormatSampleSize {
    fn sample_size_bytes(self) -> usize;
}

impl FormatSampleSize for BandFormatId {
    fn sample_size_bytes(self) -> usize {
        match self {
            Self::U8 => 1,
            Self::U16 | Self::I16 => 2,
            Self::U32 | Self::I32 | Self::F32 => 4,
            Self::F64 => 8,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use viprs_core::image::Region;

    fn make_region(w: u32, h: u32) -> Region {
        Region::new(0, 0, w, h)
    }

    /// Drive `dyn_process_region_multi` with the given byte slices.
    fn run_join(op: &BandJoin, a: &[u8], b: &[u8], output: &mut [u8], pixel_count: usize) {
        let inputs: &[&[u8]] = &[a, b];
        let regions = [make_region(pixel_count as u32, 1); 2];
        let out_region = make_region(pixel_count as u32, 1);
        let mut state = op.dyn_start();
        op.dyn_process_region_multi(state.as_mut(), inputs, output, &regions, out_region);
    }

    #[test]
    fn join_r_and_g_produces_rg() {
        // 2-pixel R image + 2-pixel G image → 2-pixel RG image
        let a = [10u8, 50]; // R channel: pixel0=10, pixel1=50
        let b = [20u8, 60]; // G channel: pixel0=20, pixel1=60
        let op = BandJoin::new(1, 1, BandFormatId::U8);
        let mut output = [0u8; 4];
        run_join(&op, &a, &b, &mut output, 2);
        assert_eq!(output, [10, 20, 50, 60]);
    }

    #[test]
    fn join_rgb_and_alpha_produces_rgba() {
        // 2-pixel RGB + 2-pixel A → 2-pixel RGBA
        let a = [10u8, 20, 30, 50, 60, 70]; // RGB
        let b = [40u8, 80]; // A
        let op = BandJoin::new(3, 1, BandFormatId::U8);
        let mut output = [0u8; 8];
        run_join(&op, &a, &b, &mut output, 2);
        assert_eq!(output, [10, 20, 30, 40, 50, 60, 70, 80]);
    }

    #[test]
    fn input_slot_count_is_2() {
        let op = BandJoin::new(2, 2, BandFormatId::U8);
        assert_eq!(op.input_slot_count(), 2);
    }

    #[test]
    fn output_bands_sums_inputs() {
        let op = BandJoin::new(3, 1, BandFormatId::U8);
        assert_eq!(op.output_bands(), 4);
        assert_eq!(op.bands(), 4);
    }

    #[test]
    fn format_passthrough() {
        let op = BandJoin::new(2, 2, BandFormatId::F32);
        assert_eq!(op.input_format(), BandFormatId::F32);
        assert_eq!(op.output_format(), BandFormatId::F32);
    }

    #[test]
    fn required_input_region_slot_is_identity() {
        let op = BandJoin::new(1, 1, BandFormatId::U8);
        let r = make_region(16, 8);
        assert_eq!(op.required_input_region_slot(&r, 0), r);
        assert_eq!(op.required_input_region_slot(&r, 1), r);
        assert_eq!(op.required_input_region(&r), r);
    }

    #[test]
    fn node_spec_is_identity() {
        let op = BandJoin::new(2, 2, BandFormatId::U8);
        assert_eq!(op.node_spec(64, 32), NodeSpec::identity(64, 32));
    }

    #[test]
    fn join_u16_inputs_preserves_two_byte_sample_boundaries() {
        let a = bytemuck::cast_slice(&[1000u16, 2000u16]).to_vec();
        let b = bytemuck::cast_slice(&[3000u16, 4000u16]).to_vec();
        let op = BandJoin::new(1, 1, BandFormatId::U16);
        let mut output = vec![0u8; 8];

        run_join(&op, &a, &b, &mut output, 2);

        assert_eq!(
            bytemuck::cast_slice::<u8, u16>(&output),
            &[1000u16, 3000, 2000, 4000]
        );
    }

    #[test]
    fn single_input_fallback_panics_in_debug_builds() {
        let op = BandJoin::new(1, 1, BandFormatId::U8);
        let input = vec![1u8, 2, 3];
        let mut output = vec![9u8; 4];
        let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            let mut state = op.dyn_start();
            op.dyn_process_region(
                state.as_mut(),
                &input,
                &mut output,
                make_region(3, 1),
                make_region(4, 1),
            );
        }));

        assert!(result.is_err());
    }

    #[test]
    fn demand_hint_is_thin_strip() {
        let op = BandJoin::new(1, 2, BandFormatId::F32);
        assert_eq!(op.demand_hint(), DemandHint::ThinStrip);
    }

    #[test]
    fn round_trip_bandsplit_then_bandjoin() {
        use crate::conversion::bandsplit::BandSplit;
        use viprs_core::op::Op;
        use viprs_core::{
            format::U8,
            image::{Tile, TileMut},
        };

        // Original RGBA image, 3 pixels
        let rgba = [10u8, 20, 30, 40, 11, 21, 31, 41, 12, 22, 32, 42];

        // Extract band 0 (R) and band 1 (G) via BandSplit
        let region = Region::new(0, 0, 3, 1);
        let mut r_data = [0u8; 3];
        let mut g_data = [0u8; 3];

        let split_r = BandSplit::<U8>::new(0, 4);
        let split_g = BandSplit::<U8>::new(1, 4);

        {
            let input = Tile::<U8>::new(region, 4, &rgba);
            let mut out = TileMut::<U8>::new(region, 1, &mut r_data);
            split_r.start();
            split_r.process_region(&mut (), &input, &mut out);
        }
        {
            let input = Tile::<U8>::new(region, 4, &rgba);
            let mut out = TileMut::<U8>::new(region, 1, &mut g_data);
            split_g.start();
            split_g.process_region(&mut (), &input, &mut out);
        }

        assert_eq!(r_data, [10, 11, 12], "R channel mismatch");
        assert_eq!(g_data, [20, 21, 22], "G channel mismatch");

        // Rejoin R + G → RG (first 2 bands of RGBA)
        let join = BandJoin::new(1, 1, BandFormatId::U8);
        let mut rg_data = [0u8; 6];
        run_join(&join, &r_data, &g_data, &mut rg_data, 3);

        // Compare against direct ExtractBands output for bands [0,2)
        use crate::conversion::extract_bands::ExtractBands;
        let extract = ExtractBands::<U8>::new(0, 2, 4);
        let mut expected = [0u8; 6];
        {
            let input = Tile::<U8>::new(region, 4, &rgba);
            let mut out = TileMut::<U8>::new(region, 2, &mut expected);
            extract.start();
            extract.process_region(&mut (), &input, &mut out);
        }

        assert_eq!(
            rg_data, expected,
            "BandJoin(BandSplit(0), BandSplit(1)) must equal ExtractBands([0,2))"
        );
    }
}
