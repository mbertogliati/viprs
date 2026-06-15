#![allow(dead_code)]
// REASON: structural bridge wiring is staged for future pipeline-builder exposure.

use crate::{
    domain::op::{NodeSpec, Op},
    domain::{
        format::BandFormat,
        image::{DemandHint, Region, Tile, TileMut, clamp_i64_to_i32},
    },
};

/// Clockwise 90° rotation.
///
/// For an image of original height `H`, output pixel `(ox, oy)` comes from input
/// pixel `(ix=oy, iy=H-1-ox)`. This swaps the image dimensions: a W×H input
/// becomes an H×W output.
pub struct Rotate90<F: BandFormat> {
    /// Width of the image before rotation. Used in `required_input_region` and
    /// `output_height`.
    image_width: u32,
    /// Height of the image before rotation. Used in `output_width`.
    image_height: u32,
    _format: std::marker::PhantomData<F>,
}

impl<F: BandFormat + Send + Sync> Rotate90<F> {
    #[must_use]
    /// Creates a new `Rotate90`.
    pub const fn new(image_width: u32, image_height: u32) -> Self {
        Self {
            image_width,
            image_height,
            _format: std::marker::PhantomData,
        }
    }
}

impl<F: BandFormat> Op for Rotate90<F>
where
    F::Sample: Copy,
{
    type Input = F;
    type Output = F;
    type State = ();

    fn preferred_tile_geometry(&self) -> DemandHint {
        DemandHint::SmallTile
    }

    /// Map an output tile in rotated-image space to the input tile it requires.
    ///
    /// For output tile `(ox, oy, ow, oh)`:
    /// - Required input x-range is `[oy, oy+oh)` (output rows map to input cols).
    /// - Required input y-range is `[H-ox-ow, H-ox)` where H = original image height.
    /// - Input tile dimensions are `(oh, ow)` — transposed from output.
    fn required_input_region(&self, output: &Region) -> Region {
        Region::new(
            output.y,
            clamp_i64_to_i32(
                i64::from(self.image_height) - i64::from(output.x) - i64::from(output.width),
            ),
            output.height,
            output.width,
        )
    }

    /// Buffer sizing: input tile is `(tile_h × tile_w)`, output tile is `(tile_w × tile_h)`.
    fn node_spec(&self, tile_w: u32, tile_h: u32) -> NodeSpec {
        NodeSpec {
            input_tile_w: tile_h,
            input_tile_h: tile_w,
            output_tile_w: tile_w,
            output_tile_h: tile_h,
            coordinate_driven_source: None,
        }
    }

    fn start(&self) {}

    /// Transpose-with-reversal pixel copy implementing CW 90° rotation.
    ///
    /// Input tile: width=oh, height=ow (transposed from output).
    /// Output tile: width=ow, height=oh.
    ///
    /// For each output sample at `(col, row)` where col ∈ 0..ow, row ∈ 0..oh:
    ///   `input_col` = row          (input width dimension = oh)
    ///   `input_row` = ow - 1 - col (input height dimension = ow; reversed)
    #[inline]
    fn process_region(&self, _state: &mut (), input: &Tile<F>, output: &mut TileMut<F>) {
        let bands = input.bands as usize;
        let ow = output.region.width as usize;
        let oh = output.region.height as usize;
        // Input tile width = oh, input row stride = oh * bands.
        let input_row_stride = oh * bands;

        for row in 0..oh {
            for col in 0..ow {
                let input_col = row;
                let input_row = ow - 1 - col;
                let src = input_row * input_row_stride + input_col * bands;
                let dst = row * ow * bands + col * bands;
                output.data[dst..dst + bands].copy_from_slice(&input.data[src..src + bands]);
            }
        }
    }
}

/// `DynOperation` wrapper for `Rotate90` that overrides `output_width`/`output_height`.
///
/// `OperationBridge` delegates `output_width`/`output_height` to the default identity
/// implementation in `DynOperation`. Rotate90 changes image dimensions, so we need a
/// wrapper that stores the pre-rotation dimensions and overrides those two methods.
///
/// `pub(crate)` — callers use `PipelineBuilder::rotate90`, not this type directly.
pub(crate) struct Rotate90Bridge<F: BandFormat>
where
    F::Sample: bytemuck::Pod + Copy,
{
    inner: crate::domain::op::OperationBridge<Rotate90<F>>,
}

impl<F: BandFormat> Rotate90Bridge<F>
where
    F::Sample: bytemuck::Pod + Copy,
{
    pub fn new(image_width: u32, image_height: u32, bands: u32) -> Self {
        Self {
            inner: crate::domain::op::OperationBridge::new(
                Rotate90::new(image_width, image_height),
                bands,
            ),
        }
    }
}

impl<F: BandFormat> crate::domain::op::DynOperation for Rotate90Bridge<F>
where
    F::Sample: bytemuck::Pod + Copy + Send,
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
        self.inner.op.image_height
    }

    fn output_height(&self, _input_h: u32) -> u32 {
        self.inner.op.image_width
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
    use crate::domain::{
        format::{BandFormatId, U8},
        image::Region,
        op::DynOperation,
    };
    use proptest::prelude::*;

    #[test]
    fn required_input_region_correct() {
        // Image 8×6 (W=8, H=6). Output tile (0, 0, 4, 3) → input region (0, 2, 3, 4).
        let op = Rotate90::<U8>::new(8, 6);
        let output = Region::new(0, 0, 4, 3);
        let input = op.required_input_region(&output);
        assert_eq!(input.x, 0);
        assert_eq!(input.y, 2); // image_height - ox - ow = 6 - 0 - 4 = 2
        assert_eq!(input.width, 3); // oh = 3
        assert_eq!(input.height, 4); // ow = 4
    }

    #[test]
    fn required_input_region_clamps_large_image_height() {
        let op = Rotate90::<U8>::new(1, u32::MAX);
        let output = Region::new(0, 0, 1, 1);
        let input = op.required_input_region(&output);
        assert_eq!(input.x, 0);
        assert_eq!(input.y, i32::MAX);
        assert_eq!(input.width, 1);
        assert_eq!(input.height, 1);
    }

    #[test]
    fn node_spec_transposes_dimensions() {
        let op = Rotate90::<U8>::new(1024, 768);
        let spec = op.node_spec(512, 256);
        assert_eq!(spec.input_tile_w, 256);
        assert_eq!(spec.input_tile_h, 512);
        assert_eq!(spec.output_tile_w, 512);
        assert_eq!(spec.output_tile_h, 256);
    }

    #[test]
    fn output_dimensions() {
        // Image 8×6: after CW rotation → 6×8.
        // output_width/output_height are on DynOperation (via Rotate90Bridge),
        // not on Op directly — test through the bridge.
        use crate::domain::op::DynOperation;
        let bridge = Rotate90Bridge::<U8>::new(8, 6, 1);
        assert_eq!(bridge.output_width(8), 6); // image_height
        assert_eq!(bridge.output_height(6), 8); // image_width
    }

    #[test]
    fn process_region_correct() {
        // 2×2 image. Full image as single tile.
        // Input (row-major): row 0 = [1, 2], row 1 = [3, 4]
        // CW 90°: out(ox, oy) = in(x=oy, y=W-1-ox) with W=2:
        //   out(0,0) = in(x=0, y=1) = 3
        //   out(1,0) = in(x=0, y=0) = 1
        //   out(0,1) = in(x=1, y=1) = 4
        //   out(1,1) = in(x=1, y=0) = 2
        // Expected output (row-major): [3, 1, 4, 2]
        let op = Rotate90::<U8>::new(2, 2);
        // Full image: required_input_region for output (0,0,2,2)
        // = (y=0, W-0-2=0, oh=2, ow=2) = (0, 0, 2, 2)
        let output_region = Region::new(0, 0, 2, 2);
        let input_region = op.required_input_region(&output_region);
        let input_data = vec![1u8, 2, 3, 4];
        let mut output_data = vec![0u8; 4];
        let input = Tile::<U8>::new(input_region, 1, &input_data);
        let mut output = TileMut::<U8>::new(output_region, 1, &mut output_data);
        let mut state = ();
        op.process_region(&mut state, &input, &mut output);
        assert_eq!(output_data, vec![3u8, 1, 4, 2]);
    }

    #[test]
    fn rotate90_bridge_processes_bytes_and_reports_metadata() {
        let bridge = Rotate90Bridge::<U8>::new(2, 3, 1);
        let output_region = Region::new(0, 0, 3, 2);
        let input_region = bridge.required_input_region(&output_region);
        let input_data = [1u8, 2, 3, 4, 5, 6];
        let mut output_data = [0u8; 6];
        let mut state = bridge.dyn_start();

        assert_eq!(bridge.input_format(), BandFormatId::U8);
        assert_eq!(bridge.output_format(), BandFormatId::U8);
        assert_eq!(
            bridge.demand_hint(),
            crate::domain::image::DemandHint::SmallTile
        );
        assert_eq!(
            bridge.node_spec(4, 2),
            NodeSpec {
                input_tile_w: 2,
                input_tile_h: 4,
                output_tile_w: 4,
                output_tile_h: 2,
                coordinate_driven_source: None,
            }
        );

        bridge.dyn_process_region(
            state.as_mut(),
            &input_data,
            &mut output_data,
            input_region,
            output_region,
        );

        assert_eq!(bridge.output_width(2), 3);
        assert_eq!(bridge.output_height(3), 2);
        assert_eq!(output_data, [5u8, 3, 1, 6, 4, 2]);
    }

    proptest! {
        /// Rotating CW 90° four times must restore the original image.
        #[test]
        fn rotate_four_times_is_identity(
            rows in 1usize..=6usize,
            cols in 1usize..=6usize,
        ) {
            let pixels: Vec<u8> = (0..(rows * cols) as u8).collect();

            let rotate_once = |data: &[u8], w: usize, h: usize| -> Vec<u8> {
                let op = Rotate90::<U8>::new(w as u32, h as u32);
                let out_w = h; // image_height
                let out_h = w; // image_width
                let output_region = Region::new(0, 0, out_w as u32, out_h as u32);
                let input_region = op.required_input_region(&output_region);
                let mut result = vec![0u8; out_w * out_h];
                let input = Tile::<U8>::new(input_region, 1, data);
                let mut output = TileMut::<U8>::new(output_region, 1, &mut result);
                let mut state = ();
                op.process_region(&mut state, &input, &mut output);
                result
            };

            // W×H → H×W → W×H → H×W → W×H
            let r1 = rotate_once(&pixels, cols, rows);
            let r2 = rotate_once(&r1, rows, cols);
            let r3 = rotate_once(&r2, cols, rows);
            let r4 = rotate_once(&r3, rows, cols);

            prop_assert_eq!(r4, pixels);
        }
    }
}
