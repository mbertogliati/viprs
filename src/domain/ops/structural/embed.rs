#![allow(dead_code)]
// REASON: structural bridge wiring is staged for future pipeline-builder exposure.

use bytemuck::Zeroable;
use std::marker::PhantomData;

use crate::{
    domain::op::{NodeSpec, Op, OperationBridge},
    domain::{
        format::BandFormat,
        image::{DemandHint, Region, Tile, TileMut, clamp_i64_to_i32},
    },
};

/// Fill mode for the border region outside the embedded source image.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ExtendMode {
    /// Fill with the zero value of the sample type (black for unsigned formats, 0.0 for float).
    Black,
    /// Replicate the nearest edge pixel of the source (clamp-to-edge).
    ///
    /// This works because `MemorySource::read_region` already clamps out-of-bounds
    /// coordinates to the source edges, so `process_region` copies the clamped input
    /// pixel directly into the output for any canvas position outside the source bounds.
    Edge,
}

/// Embed a `src_width × src_height` source image into a larger canvas of
/// `dst_width × dst_height`, placing the top-left corner at `(x_off, y_off)`.
///
/// Pixels in the canvas that fall outside the source image are filled according
/// to `extend`:
/// - `ExtendMode::Black`: zero-filled (black for unsigned types, 0.0 for floats).
/// - `ExtendMode::Edge`: replicated from the nearest edge pixel of the source.
///   This is free because `MemorySource::read_region` already performs clamp-to-edge.
///
/// `Embed` is an `Op` (not a `ViewOp`) because it must actively write fill pixels
/// into the output buffer — unlike `ExtractArea`, which is a pure coordinate remap.
pub struct Embed<F: BandFormat> {
    dst_width: u32,
    dst_height: u32,
    x_off: u32,
    y_off: u32,
    src_width: u32,
    src_height: u32,
    extend: ExtendMode,
    _fmt: PhantomData<F>,
}

impl<F: BandFormat + Send + Sync> Embed<F> {
    /// Create a new `Embed` operation.
    ///
    /// - `dst_width`, `dst_height`: dimensions of the output canvas.
    /// - `x_off`, `y_off`: top-left position of the source image within the canvas.
    /// - `src_width`, `src_height`: dimensions of the source image (must match the
    ///   upstream pipeline output).
    /// - `extend`: fill mode for canvas pixels outside the source region.
    #[allow(clippy::too_many_arguments)]
    #[must_use]
    pub const fn new(
        dst_width: u32,
        dst_height: u32,
        x_off: u32,
        y_off: u32,
        src_width: u32,
        src_height: u32,
        extend: ExtendMode,
    ) -> Self {
        Self {
            dst_width,
            dst_height,
            x_off,
            y_off,
            src_width,
            src_height,
            extend,
            _fmt: PhantomData,
        }
    }
}

impl<F: BandFormat> Op for Embed<F>
where
    F::Sample: Zeroable,
{
    type Input = F;
    type Output = F;
    type State = ();

    fn demand_hint(&self) -> DemandHint {
        DemandHint::ThinStrip
    }

    /// Map a canvas output region back to source coordinates by subtracting the offset.
    ///
    /// The resulting region may have negative x/y or exceed source dimensions.
    /// `MemorySource::read_region` clamps those to the source edges automatically,
    /// which provides correct pixels for `ExtendMode::Edge` at zero extra cost.
    /// For `ExtendMode::Black`, `process_region` checks `in_bounds` and writes zeros.
    fn required_input_region(&self, output: &Region) -> Region {
        Region::new(
            clamp_i64_to_i32(i64::from(output.x) - i64::from(self.x_off)),
            clamp_i64_to_i32(i64::from(output.y) - i64::from(self.y_off)),
            output.width,
            output.height,
        )
    }

    fn start(&self) {}

    /// Fill the output canvas tile, copying from the source where in-bounds and
    /// applying the `extend` fill mode elsewhere.
    ///
    /// `output.region` is in canvas coordinates (`0`..`dst_width` × `0`..`dst_height`).
    /// `input.region` is in source coordinates (may be negative or exceed `src` bounds).
    /// Both tiles have identical width and height — they are the same tile at different
    /// coordinate origins.
    #[inline]
    fn process_region(&self, _state: &mut (), input: &Tile<F>, output: &mut TileMut<F>) {
        let bands = output.bands as usize;
        let tile_width = output.region.width as usize;
        let canvas_origin_x = i64::from(output.region.x);
        let canvas_origin_y = i64::from(output.region.y);
        let x_off = i64::from(self.x_off);
        let y_off = i64::from(self.y_off);
        let src_width = i64::from(self.src_width);
        let src_height = i64::from(self.src_height);

        for row in 0..output.region.height as usize {
            let canvas_y = canvas_origin_y + row as i64;
            for col in 0..output.region.width as usize {
                let canvas_x = canvas_origin_x + col as i64;

                // Source coordinates: canvas position minus the embed offset.
                let src_x = canvas_x - x_off;
                let src_y = canvas_y - y_off;

                let in_bounds = src_x >= 0 && src_x < src_width && src_y >= 0 && src_y < src_height;

                let dst_idx = (row * tile_width + col) * bands;
                let src_idx = (row * tile_width + col) * bands;

                if in_bounds || self.extend == ExtendMode::Edge {
                    // Copy the input pixel (source clamp-to-edge already applied by
                    // MemorySource::read_region for out-of-bounds Edge positions).
                    output.data[dst_idx..dst_idx + bands]
                        .copy_from_slice(&input.data[src_idx..src_idx + bands]);
                } else {
                    // ExtendMode::Black: fill with zero for every band.
                    // bytemuck::Zeroable::zeroed() produces the zero value for any
                    // Pod-compatible sample type (0u8, 0.0f32, etc.) with no branches.
                    for b in 0..bands {
                        output.data[dst_idx + b] = F::Sample::zeroed();
                    }
                }
            }
        }
    }
}

/// `DynOperation` wrapper for `Embed` that overrides `output_width`/`output_height`.
///
/// `OperationBridge` delegates `output_width`/`output_height` to the identity default
/// in `DynOperation`. `Embed` changes image dimensions (dst != src), so a wrapper is
/// needed that stores dst dimensions and overrides those two methods. Same pattern as
/// `Rotate90Bridge`.
///
/// `pub(crate)` — callers use `PipelineBuilder::embed`, not this type directly.
pub(crate) struct EmbedBridge<F: BandFormat>
where
    F::Sample: bytemuck::Pod + Zeroable,
{
    inner: OperationBridge<Embed<F>>,
}

impl<F: BandFormat> EmbedBridge<F>
where
    F::Sample: bytemuck::Pod + Zeroable,
{
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        dst_width: u32,
        dst_height: u32,
        x_off: u32,
        y_off: u32,
        src_width: u32,
        src_height: u32,
        extend: ExtendMode,
        bands: u32,
    ) -> Self {
        Self {
            inner: OperationBridge::new(
                Embed::new(
                    dst_width, dst_height, x_off, y_off, src_width, src_height, extend,
                ),
                bands,
            ),
        }
    }
}

impl<F: BandFormat> crate::domain::op::DynOperation for EmbedBridge<F>
where
    F::Sample: bytemuck::Pod + Zeroable + Send,
{
    fn input_format(&self) -> crate::domain::format::BandFormatId {
        self.inner.input_format()
    }

    fn output_format(&self) -> crate::domain::format::BandFormatId {
        self.inner.output_format()
    }

    fn bands(&self) -> u32 {
        self.inner.bands()
    }

    fn demand_hint(&self) -> DemandHint {
        self.inner.demand_hint()
    }

    fn required_input_region(&self, output: &Region) -> Region {
        self.inner.required_input_region(output)
    }

    fn node_spec(&self, tile_w: u32, tile_h: u32) -> NodeSpec {
        self.inner.node_spec(tile_w, tile_h)
    }

    fn output_width(&self, _input_w: u32) -> u32 {
        self.inner.op.dst_width
    }

    fn output_height(&self, _input_h: u32) -> u32 {
        self.inner.op.dst_height
    }

    fn dyn_start(&self) -> Box<dyn std::any::Any + Send> {
        self.inner.dyn_start()
    }

    fn dyn_start_with_tile(&self, tile_w: u32, tile_h: u32) -> Box<dyn std::any::Any + Send> {
        self.inner.dyn_start_with_tile(tile_w, tile_h)
    }

    fn dyn_process_region(
        &self,
        state: &mut dyn std::any::Any,
        input: &[u8],
        output: &mut [u8],
        input_region: Region,
        output_region: Region,
    ) {
        self.inner
            .dyn_process_region(state, input, output, input_region, output_region);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::domain::{format::U8, image::Region};
    use proptest::prelude::*;

    /// Helper: run Embed process_region with given pixel data.
    ///
    /// `src_pixels` is a flat row-major buffer for a `tile_w × tile_h` tile.
    /// `output_region` is in canvas coordinates; `input_region` is the shifted region
    /// (canvas minus offset), both with the same width × height.
    fn run_embed(
        op: &Embed<U8>,
        src_pixels: &[u8],
        input_region: Region,
        output_region: Region,
        bands: u32,
    ) -> Vec<u8> {
        let n = input_region.pixel_count() * bands as usize;
        let mut out = vec![0u8; n];
        let input = Tile::<U8>::new(input_region, bands, src_pixels);
        let mut output = TileMut::<U8>::new(output_region, bands, &mut out);
        let mut state = ();
        op.process_region(&mut state, &input, &mut output);
        out
    }

    // ── required_input_region ────────────────────────────────────────────────

    #[test]
    fn required_input_region_subtracts_offset() {
        // Canvas 4×4, source 2×2 at (1, 1).
        let op = Embed::<U8>::new(4, 4, 1, 1, 2, 2, ExtendMode::Black);
        // Requesting output tile at canvas (1, 1, 2, 2) → source region (0, 0, 2, 2).
        let out_region = Region::new(1, 1, 2, 2);
        let in_region = op.required_input_region(&out_region);
        assert_eq!(in_region.x, 0);
        assert_eq!(in_region.y, 0);
        assert_eq!(in_region.width, 2);
        assert_eq!(in_region.height, 2);
    }

    #[test]
    fn required_input_region_can_go_negative() {
        // Canvas 8×8, source 4×4 at (3, 3).
        let op = Embed::<U8>::new(8, 8, 3, 3, 4, 4, ExtendMode::Black);
        // Requesting canvas tile at (0, 0, 3, 3) → source region (-3, -3, 3, 3).
        let out_region = Region::new(0, 0, 3, 3);
        let in_region = op.required_input_region(&out_region);
        assert_eq!(in_region.x, -3);
        assert_eq!(in_region.y, -3);
    }

    #[test]
    fn required_input_region_clamps_large_offsets_to_i32_min() {
        let op = Embed::<U8>::new(10, 10, u32::MAX, u32::MAX, 1, 1, ExtendMode::Black);
        let out_region = Region::new(0, 0, 1, 1);
        let in_region = op.required_input_region(&out_region);
        assert_eq!(in_region.x, i32::MIN);
        assert_eq!(in_region.y, i32::MIN);
        assert_eq!(in_region.width, 1);
        assert_eq!(in_region.height, 1);
    }

    // ── EmbedBridge output dimensions ────────────────────────────────────────

    #[test]
    fn output_dimensions_are_dst_dimensions() {
        use crate::domain::op::DynOperation;
        let bridge = EmbedBridge::<U8>::new(10, 20, 0, 0, 5, 5, ExtendMode::Black, 1);
        assert_eq!(bridge.output_width(5), 10);
        assert_eq!(bridge.output_height(5), 20);
    }

    // ── ExtendMode::Black ─────────────────────────────────────────────────────

    #[test]
    fn black_fill_outside_source() {
        // Source 2×2 embedded in 4×4 canvas at (1, 1).
        // Request the first canvas row (y=0, full width=4): all pixels are outside source.
        let op = Embed::<U8>::new(4, 4, 1, 1, 2, 2, ExtendMode::Black);
        // Canvas row y=0, width=4, height=1.
        // input_region: (0-1, 0-1, 4, 1) = (-1, -1, 4, 1).
        // MemorySource would clamp these to edge pixels, but Embed::Black overrides with 0.
        let src = vec![255u8; 4]; // whatever the clamped edge pixel is, we expect 0
        let in_region = Region::new(-1, -1, 4, 1);
        let out_region = Region::new(0, 0, 4, 1);
        let result = run_embed(&op, &src, in_region, out_region, 1);
        // All canvas row 0 pixels are before the source (y < y_off=1), so all should be 0.
        assert_eq!(result, vec![0u8, 0, 0, 0]);
    }

    #[test]
    fn black_fill_treats_large_offsets_as_out_of_bounds() {
        let op = Embed::<U8>::new(1, 1, u32::MAX, 0, 2, 1, ExtendMode::Black);
        let src = [255u8];
        let in_region = Region::new(i32::MIN, 0, 1, 1);
        let out_region = Region::new(0, 0, 1, 1);
        let result = run_embed(&op, &src, in_region, out_region, 1);
        assert_eq!(result, vec![0u8]);
    }

    #[test]
    fn source_pixel_appears_at_correct_canvas_position() {
        // Source 2×2 embedded in 4×4 canvas at offset (1, 1).
        // Source pixels: [10, 20, 30, 40] (row-major).
        // Request canvas tile (1, 1, 2, 2) — the tile that overlaps the source exactly.
        // input_region shifts to (0, 0, 2, 2).
        let op = Embed::<U8>::new(4, 4, 1, 1, 2, 2, ExtendMode::Black);
        let src = vec![10u8, 20, 30, 40];
        let in_region = Region::new(0, 0, 2, 2);
        let out_region = Region::new(1, 1, 2, 2);
        let result = run_embed(&op, &src, in_region, out_region, 1);
        // All four pixels are in-bounds → copied from source.
        assert_eq!(result, vec![10u8, 20, 30, 40]);
    }

    #[test]
    fn embed_with_zero_offset_and_same_size_is_identity() {
        // Source and canvas have the same dimensions, offset = (0, 0).
        let op = Embed::<U8>::new(4, 4, 0, 0, 4, 4, ExtendMode::Black);
        let src: Vec<u8> = (0..16).collect();
        let region = Region::new(0, 0, 4, 4);
        let result = run_embed(&op, &src, region, region, 1);
        assert_eq!(result, src);
    }

    // ── ExtendMode::Edge ──────────────────────────────────────────────────────

    #[test]
    fn edge_fill_copies_clamped_pixel() {
        // Source 2×2 at (2, 2) in a 6×6 canvas.
        // Request canvas row 0 (before the source): Embed::Edge passes through the
        // clamped input (which MemorySource already clamped to the top-edge of source).
        // In this unit test, we manually supply what the clamped input would look like.
        let op = Embed::<U8>::new(6, 6, 2, 2, 2, 2, ExtendMode::Edge);
        // Simulate that MemorySource clamped (src_x=-2, src_y=-2) → source pixel (0,0) = 99.
        let src = vec![99u8; 4]; // the clamped edge pixel repeated across the tile
        let in_region = Region::new(-2, -2, 2, 2);
        let out_region = Region::new(0, 0, 2, 2);
        let result = run_embed(&op, &src, in_region, out_region, 1);
        // Edge mode: all pixels are copied from the (clamped) input, even if out of source bounds.
        assert_eq!(result, vec![99u8; 4]);
    }

    // ── Multi-band ────────────────────────────────────────────────────────────

    #[test]
    fn black_fill_multi_band() {
        // 3-band RGB, source 1×1 at (1, 0) in 2×1 canvas.
        // Canvas pixel 0 is outside source → should be [0, 0, 0].
        // Canvas pixel 1 is the source → should be [10, 20, 30].
        let op = Embed::<U8>::new(2, 1, 1, 0, 1, 1, ExtendMode::Black);
        // Full canvas tile 2×1, 3 bands.
        // input_region = canvas - offset = (0-1, 0-0, 2, 1) = (-1, 0, 2, 1).
        // MemorySource would give: col0→clamped src px 0 = [10,20,30], col1→src px 0 = [10,20,30].
        // But col0 in canvas (x=0) has src_x = 0 - 1 = -1 < 0, so Black writes zeros.
        // col1 in canvas (x=1) has src_x = 1 - 1 = 0, in bounds, copy input pixel at idx 1.
        let src = vec![10u8, 20, 30, 10, 20, 30]; // clamped: both cols give source [10,20,30]
        let in_region = Region::new(-1, 0, 2, 1);
        let out_region = Region::new(0, 0, 2, 1);
        let result = run_embed(&op, &src, in_region, out_region, 3);
        assert_eq!(result, vec![0u8, 0, 0, 10, 20, 30]);
    }

    #[test]
    fn edge_extend_can_mix_clamped_and_in_bounds_pixels() {
        let op = Embed::<U8>::new(3, 1, 1, 0, 2, 1, ExtendMode::Edge);
        let src = vec![10u8, 10, 20];
        let in_region = Region::new(-1, 0, 3, 1);
        let out_region = Region::new(0, 0, 3, 1);
        let result = run_embed(&op, &src, in_region, out_region, 1);
        assert_eq!(result, vec![10, 10, 20]);
    }

    #[test]
    fn black_fill_bottom_right_border_zeroes_all_bands() {
        let op = Embed::<U8>::new(2, 2, 0, 0, 1, 1, ExtendMode::Black);
        let src = vec![8u8, 9, 10, 8, 9, 10];
        let in_region = Region::new(0, 1, 1, 2);
        let out_region = Region::new(0, 1, 1, 2);
        let result = run_embed(&op, &src, in_region, out_region, 3);
        assert_eq!(result, vec![0, 0, 0, 0, 0, 0]);
    }

    #[test]
    fn embed_bridge_delegates_dyn_contract_and_processing() {
        use crate::domain::{format::BandFormatId, op::DynOperation};

        let bridge = EmbedBridge::<U8>::new(4, 3, 1, 1, 2, 1, ExtendMode::Black, 1);
        let output_region = Region::new(1, 1, 2, 1);
        let input_region = bridge.required_input_region(&output_region);
        let mut output = vec![0u8; 2];
        let mut state = bridge.dyn_start_with_tile(2, 1);

        assert_eq!(bridge.input_format(), BandFormatId::U8);
        assert_eq!(bridge.output_format(), BandFormatId::U8);
        assert_eq!(bridge.bands(), 1);
        assert_eq!(bridge.demand_hint(), DemandHint::ThinStrip);
        assert_eq!(bridge.node_spec(2, 1), NodeSpec::identity(2, 1));

        bridge.dyn_process_region(
            state.as_mut(),
            &[12, 34],
            &mut output,
            input_region,
            output_region,
        );

        assert_eq!(output, vec![12, 34]);
    }

    // ── Proptest ──────────────────────────────────────────────────────────────

    proptest! {
        /// Embedding with offset (0, 0) and dst == src dimensions must be the identity.
        ///
        /// This is a pure process_region test; it does not exercise the source or scheduler.
        #[test]
        fn embed_identity_zero_offset(
            pixels in proptest::collection::vec(0u8..=255u8, 1..=64),
        ) {
            let len = pixels.len() as u32;
            let op = Embed::<U8>::new(len, 1, 0, 0, len, 1, ExtendMode::Black);
            let region = Region::new(0, 0, len, 1);
            let result = run_embed(&op, &pixels, region, region, 1);
            prop_assert_eq!(result, pixels);
        }

        /// For ExtendMode::Black: any canvas column with canvas_x < x_off must be zero.
        ///
        /// The left border is [0, x_off) — all pixels strictly before the source offset.
        /// The tile covers exactly those pixels, so src_x = canvas_x - x_off is negative
        /// for all of them. Black fill must produce zero regardless of the clamped source.
        #[test]
        fn black_left_border_is_zero(
            x_off in 1u32..=8,
            fill_val in 0u8..=255,
        ) {
            // Canvas: (x_off + 1) × 1, source 1×1 at (x_off, 0).
            // The left border is exactly x_off pixels wide: columns [0, x_off).
            let dst_w = x_off + 1;
            let op = Embed::<U8>::new(dst_w, 1, x_off, 0, 1, 1, ExtendMode::Black);

            // Tile covers only the left border [0, x_off).
            // input_region shifts to (-x_off, 0, x_off, 1).
            // All src_x values (0 - x_off .. x_off - x_off) are < 0 → Black fills zeros.
            let src = vec![fill_val; x_off as usize]; // clamped edge pixel
            let in_region = Region::new(-(x_off as i32), 0, x_off, 1);
            let out_region = Region::new(0, 0, x_off, 1);
            let result = run_embed(&op, &src, in_region, out_region, 1);
            prop_assert!(result.iter().all(|&v| v == 0),
                "left border must be zero, got {:?}", result);
        }
    }
}
