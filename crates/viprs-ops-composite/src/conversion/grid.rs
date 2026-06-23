use std::{any::Any, marker::PhantomData};

use viprs_core::{
    format::BandFormat,
    image::{DemandHint, Region, Tile, TileMut},
    op::{NodeSpec, Op, OperationBridge},
};

/// Rearrange a strip of `tile_height`-tall frames into a row-major grid.
///
/// # Examples
/// ```ignore
/// use viprs_ops_composite::conversion::grid::GridOp;
///
/// let op = GridOp::new(/* operation parameters */);
/// // Run `op` through a compiled image pipeline.
/// ```
pub struct GridOp<F: BandFormat> {
    image_width: u32,
    image_height: u32,
    tile_height: u32,
    across: u32,
    _format: PhantomData<F>,
}

impl<F: BandFormat> GridOp<F> {
    #[must_use]
    /// Creates a new `GridOp`.
    pub fn new(image_width: u32, image_height: u32, tile_height: u32, across: u32) -> Self {
        debug_assert!(tile_height > 0, "GridOp: tile_height must be >= 1");
        debug_assert!(across > 0, "GridOp: across must be >= 1");
        debug_assert!(
            image_height.is_multiple_of(tile_height),
            "GridOp: input height must be a multiple of tile_height"
        );

        Self {
            image_width,
            image_height,
            tile_height,
            across,
            _format: PhantomData,
        }
    }

    const fn frame_count(&self) -> u32 {
        self.image_height.div_ceil(self.tile_height)
    }
}

impl<F> Op for GridOp<F>
where
    F: BandFormat,
    F::Sample: Copy + Default,
{
    type Input = F;
    type Output = F;
    type State = ();

    fn demand_hint(&self) -> DemandHint {
        DemandHint::SmallTile
    }

    fn required_input_region(&self, _output: &Region) -> Region {
        Region::new(0, 0, self.image_width, self.image_height)
    }

    fn node_spec(&self, tile_w: u32, tile_h: u32) -> NodeSpec {
        NodeSpec {
            input_tile_w: self.image_width,
            input_tile_h: self.image_height,
            output_tile_w: tile_w,
            output_tile_h: tile_h,
            coordinate_driven_source: None,
        }
    }

    fn start(&self) {}

    #[inline]
    fn process_region(&self, _state: &mut (), input: &Tile<F>, output: &mut TileMut<F>) {
        let bands = output.bands as usize;
        let input_width = input.region.width as usize;
        let output_width = output.region.width as usize;
        let frame_count = self.frame_count();
        let tile_height = self.tile_height as usize;
        let base_y = input.region.y;
        let base_x = input.region.x;

        for row in 0..output.region.height as usize {
            let out_y = output.region.y + row as i32;
            let cell_y = out_y.div_euclid(self.tile_height as i32) as u32;
            let local_y = out_y.rem_euclid(self.tile_height as i32) as u32;

            for col in 0..output.region.width as usize {
                let out_x = output.region.x + col as i32;
                let cell_x = out_x.div_euclid(self.image_width as i32) as u32;
                let local_x = out_x.rem_euclid(self.image_width as i32) as u32;
                let tile_index = cell_y * self.across + cell_x;
                let dst_base = (row * output_width + col) * bands;

                if tile_index >= frame_count {
                    output.data[dst_base..dst_base + bands].fill(F::Sample::default());
                    continue;
                }

                let src_y = tile_index as usize * tile_height + local_y as usize;
                let src_x = local_x as usize;
                let src_base =
                    ((src_y - base_y as usize) * input_width + (src_x - base_x as usize)) * bands;

                output.data[dst_base..dst_base + bands]
                    .copy_from_slice(&input.data[src_base..src_base + bands]);
            }
        }
    }
}

/// Represents a grid bridge.
pub struct GridBridge<F: BandFormat>
where
    F::Sample: bytemuck::Pod + Copy + Default,
{
    inner: OperationBridge<GridOp<F>>,
}

impl<F: BandFormat> GridBridge<F>
where
    F::Sample: bytemuck::Pod + Copy + Default,
{
    #[must_use]
    /// Creates a new `GridBridge`.
    pub fn new(
        image_width: u32,
        image_height: u32,
        tile_height: u32,
        across: u32,
        bands: u32,
    ) -> Self {
        Self {
            inner: OperationBridge::new(
                GridOp::new(image_width, image_height, tile_height, across),
                bands,
            ),
        }
    }
}

impl<F> viprs_core::op::DynOperation for GridBridge<F>
where
    F: BandFormat + Send + Sync,
    F::Sample: bytemuck::Pod + Copy + Default + Send,
{
    fn input_format(&self) -> viprs_core::format::BandFormatId {
        self.inner.input_format()
    }

    fn output_format(&self) -> viprs_core::format::BandFormatId {
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

    fn output_width(&self, input_w: u32) -> u32 {
        input_w.saturating_mul(self.inner.op.across)
    }

    fn output_height(&self, input_h: u32) -> u32 {
        let frame_count = input_h.div_ceil(self.inner.op.tile_height);
        frame_count.div_ceil(self.inner.op.across) * self.inner.op.tile_height
    }

    fn dyn_start(&self) -> Box<dyn Any + Send> {
        self.inner.dyn_start()
    }

    fn dyn_start_with_tile(&self, tile_w: u32, tile_h: u32) -> Box<dyn Any + Send> {
        self.inner.dyn_start_with_tile(tile_w, tile_h)
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
}

#[cfg(all(test, feature = "_integration"))]
mod tests {
    use super::*;
    use proptest::prelude::*;
    use viprs_core::{format::U8, op::DynOperation};
    fn run_grid(
        input_width: u32,
        input_height: u32,
        tile_height: u32,
        across: u32,
        bands: u32,
        input_data: &[u8],
        output_region: Region,
    ) -> Vec<u8> {
        let op = GridOp::<U8>::new(input_width, input_height, tile_height, across);
        let input_region = op.required_input_region(&output_region);
        let mut state = ();
        let input = Tile::<U8>::new(input_region, bands, input_data);
        let mut output_data = vec![0u8; output_region.pixel_count() * bands as usize];
        let mut output = TileMut::<U8>::new(output_region, bands, &mut output_data);
        op.process_region(&mut state, &input, &mut output);
        output_data
    }

    #[test]
    fn strip_of_four_frames_becomes_two_by_two_grid() {
        let width = 100u32;
        let tile_height = 100u32;
        let frames = [10u8, 20, 30, 40];
        let mut input = Vec::with_capacity((width * tile_height * frames.len() as u32) as usize);

        for &frame in &frames {
            input.extend(std::iter::repeat_n(frame, (width * tile_height) as usize));
        }

        let output = run_grid(
            width,
            400,
            tile_height,
            2,
            1,
            &input,
            Region::new(0, 0, 200, 200),
        );

        assert_eq!(output[0], 10);
        assert_eq!(output[199], 20);
        assert_eq!(output[200 * 199], 30);
        assert_eq!(output[200 * 200 - 1], 40);
    }

    #[test]
    fn incomplete_last_row_zero_fills_missing_cells() {
        let input = vec![
            1u8, 1, //
            2, 2, //
            3, 3,
        ];
        let output = run_grid(1, 6, 2, 2, 1, &input, Region::new(0, 0, 2, 4));
        assert_eq!(output, vec![1, 2, 1, 2, 3, 0, 3, 0]);
    }

    proptest! {
        #[test]
        fn across_one_is_identity(
            width in 1u32..=6,
            tile_height in 1u32..=4,
            frames in 1u32..=4,
            pixels in proptest::collection::vec(0u8..=255, 1usize..=144),
        ) {
            let len = (width * tile_height * frames) as usize;
            prop_assume!(pixels.len() >= len);
            let input = pixels[..len].to_vec();
            let output = run_grid(
                width,
                tile_height * frames,
                tile_height,
                1,
                1,
                &input,
                Region::new(0, 0, width, tile_height * frames),
            );
            prop_assert_eq!(output, input);
        }

        #[test]
        fn output_dimensions_follow_grid_geometry(
            width in 1u32..=8,
            tile_height in 1u32..=8,
            frames in 1u32..=8,
            across in 1u32..=4,
        ) {
            let bridge = GridBridge::<U8>::new(width, tile_height * frames, tile_height, across, 1);
            prop_assert_eq!(bridge.output_width(width), width * across);
            prop_assert_eq!(
                bridge.output_height(tile_height * frames),
                frames.div_ceil(across) * tile_height
            );
        }
    }
}
