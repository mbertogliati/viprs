#![allow(dead_code)]
// REASON: structural bridge wiring is staged for future pipeline-builder exposure.
#![allow(clippy::used_underscore_binding)]
// REASON: underscore-prefixed parameters document intentionally ignored node-spec sizing inputs.

use std::marker::PhantomData;
use viprs_core::{
    format::BandFormat,
    image::{DemandHint, Region, Tile, TileMut},
    op::{NodeSpec, Op, OperationBridge},
};

/// Tile an image `across × down` times.
pub struct Replicate<F: BandFormat> {
    image_width: u32,
    image_height: u32,
    across: u32,
    down: u32,
    _format: PhantomData<F>,
}

impl<F: BandFormat + Send + Sync> Replicate<F> {
    #[must_use]
    /// Creates a new `Replicate`.
    pub fn new(image_width: u32, image_height: u32, across: u32, down: u32) -> Self {
        debug_assert!(across >= 1, "Replicate: across must be >= 1");
        debug_assert!(down >= 1, "Replicate: down must be >= 1");
        Self {
            image_width,
            image_height,
            across,
            down,
            _format: PhantomData,
        }
    }
}

impl<F: BandFormat> Op for Replicate<F>
where
    F::Sample: Copy,
{
    type Input = F;
    type Output = F;
    type State = ();

    fn demand_hint(&self) -> DemandHint {
        DemandHint::SmallTile
    }

    fn required_input_region(&self, output: &Region) -> Region {
        let _ = output;
        Region::new(0, 0, self.image_width, self.image_height)
    }

    fn node_spec(&self, _tile_w: u32, _tile_h: u32) -> NodeSpec {
        NodeSpec {
            input_tile_w: self.image_width,
            input_tile_h: self.image_height,
            output_tile_w: _tile_w,
            output_tile_h: _tile_h,
            coordinate_driven_source: None,
        }
    }

    fn start(&self) {}

    #[inline]
    fn process_region(&self, _state: &mut (), input: &Tile<F>, output: &mut TileMut<F>) {
        let input_width = input.region.width as usize;
        let bands = output.bands as usize;
        let output_width = output.region.width as usize;

        for row in 0..output.region.height as usize {
            let src_y =
                (output.region.y + row as i32).rem_euclid(self.image_height as i32) as usize;
            for col in 0..output.region.width as usize {
                let src_x =
                    (output.region.x + col as i32).rem_euclid(self.image_width as i32) as usize;
                let src_idx = (src_y * input_width + src_x) * bands;
                let dst_idx = (row * output_width + col) * bands;
                output.data[dst_idx..dst_idx + bands]
                    .copy_from_slice(&input.data[src_idx..src_idx + bands]);
            }
        }
    }
}

pub(crate) struct ReplicateBridge<F: BandFormat>
where
    F::Sample: bytemuck::Pod + Copy,
{
    inner: OperationBridge<Replicate<F>>,
}

impl<F: BandFormat> ReplicateBridge<F>
where
    F::Sample: bytemuck::Pod + Copy,
{
    pub fn new(image_width: u32, image_height: u32, across: u32, down: u32, bands: u32) -> Self {
        Self {
            inner: OperationBridge::new(
                Replicate::new(image_width, image_height, across, down),
                bands,
            ),
        }
    }
}

impl<F: BandFormat> viprs_core::op::DynOperation for ReplicateBridge<F>
where
    F::Sample: bytemuck::Pod + Copy + Send,
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
        input_h.saturating_mul(self.inner.op.down)
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
    use proptest::prelude::*;
    use viprs_core::{
        format::U8,
        image::{Region, Tile, TileMut},
        op::DynOperation,
    };

    #[test]
    fn required_input_region_wraps_coordinates() {
        let op = Replicate::<U8>::new(4, 3, 2, 2);
        let output = Region::new(5, 4, 2, 1);
        let input = op.required_input_region(&output);
        assert_eq!(input, Region::new(0, 0, 4, 3));
    }

    #[test]
    fn output_dimensions_scale_by_repeat_count() {
        let bridge = ReplicateBridge::<U8>::new(4, 3, 3, 2, 1);
        assert_eq!(bridge.output_width(4), 12);
        assert_eq!(bridge.output_height(3), 6);
    }

    #[test]
    fn process_region_tiles_across_boundaries() {
        let op = Replicate::<U8>::new(2, 2, 2, 2);
        let input_region = Region::new(0, 0, 2, 2);
        let output_region = Region::new(1, 1, 2, 2);
        let input_data = vec![1u8, 2, 3, 4];
        let mut output_data = vec![0u8; 4];
        let input = Tile::<U8>::new(input_region, 1, &input_data);
        let mut output = TileMut::<U8>::new(output_region, 1, &mut output_data);
        let mut state = ();
        op.process_region(&mut state, &input, &mut output);
        assert_eq!(output_data, vec![4, 3, 2, 1]);
    }

    #[test]
    fn bridge_metadata_matches_inner_operation() {
        let bridge = ReplicateBridge::<U8>::new(4, 3, 2, 3, 2);
        let output = Region::new(3, 4, 2, 1);

        assert_eq!(bridge.input_format(), viprs_core::format::BandFormatId::U8);
        assert_eq!(bridge.output_format(), viprs_core::format::BandFormatId::U8);
        assert_eq!(bridge.bands(), 2);
        assert_eq!(bridge.demand_hint(), DemandHint::SmallTile);
        assert_eq!(
            bridge.required_input_region(&output),
            Region::new(0, 0, 4, 3)
        );
        assert_eq!(
            bridge.node_spec(5, 6),
            NodeSpec {
                input_tile_w: 4,
                input_tile_h: 3,
                output_tile_w: 5,
                output_tile_h: 6,
                coordinate_driven_source: None,
            }
        );

        let _ = bridge.dyn_start();
        let _ = bridge.dyn_start_with_tile(5, 6);
    }

    #[test]
    fn bridge_dyn_process_region_tiles_multi_band_bytes() {
        let bridge = ReplicateBridge::<U8>::new(2, 2, 2, 2, 2);
        let input_region = Region::new(0, 0, 2, 2);
        let output_region = Region::new(1, 1, 2, 2);
        let input_data = [1u8, 11, 2, 12, 3, 13, 4, 14];
        let mut output_data = [0u8; 8];
        let mut state = bridge.dyn_start_with_tile(2, 2);

        bridge.dyn_process_region(
            state.as_mut(),
            &input_data,
            &mut output_data,
            input_region,
            output_region,
        );

        assert_eq!(output_data, [4, 14, 3, 13, 2, 12, 1, 11]);
    }

    proptest! {
        #[test]
        fn replicate_1x1_is_identity(rows in 1usize..=6, cols in 1usize..=6) {
            let pixels: Vec<u8> = (0..(rows * cols) as u8).collect();
            let region = Region::new(0, 0, cols as u32, rows as u32);
            let op = Replicate::<U8>::new(cols as u32, rows as u32, 1, 1);
            let input_region = op.required_input_region(&region);
            let mut output = vec![0u8; pixels.len()];
            let input = Tile::<U8>::new(input_region, 1, &pixels);
            let mut output_tile = TileMut::<U8>::new(region, 1, &mut output);
            let mut state = ();
            op.process_region(&mut state, &input, &mut output_tile);
            prop_assert_eq!(output, pixels);
        }

        #[test]
        fn replicate_output_matches_modulo_tiling(
            across in 1u32..=3,
            down in 1u32..=3,
            raw_output_x in 0usize..=11,
            raw_output_y in 0usize..=11,
            raw_tile_w in 1usize..=4,
            raw_tile_h in 1usize..=4,
            pixels in proptest::collection::vec(0u8..=255, 12),
        ) {
            let width = 4usize;
            let height = 3usize;
            let output_width = width * across as usize;
            let output_height = height * down as usize;
            let output_x = raw_output_x % output_width;
            let output_y = raw_output_y % output_height;
            let tile_w = raw_tile_w.min(output_width - output_x);
            let tile_h = raw_tile_h.min(output_height - output_y);

            let op = Replicate::<U8>::new(width as u32, height as u32, across, down);
            let input_region = Region::new(0, 0, width as u32, height as u32);
            let output_region =
                Region::new(output_x as i32, output_y as i32, tile_w as u32, tile_h as u32);
            let input = Tile::<U8>::new(input_region, 1, &pixels);
            let mut output = vec![0u8; tile_w * tile_h];
            let mut output_tile = TileMut::<U8>::new(output_region, 1, &mut output);
            let mut state = ();
            op.process_region(&mut state, &input, &mut output_tile);

            for row in 0..tile_h {
                for col in 0..tile_w {
                    let src_x = (output_x + col) % width;
                    let src_y = (output_y + row) % height;
                    prop_assert_eq!(output[row * tile_w + col], pixels[src_y * width + src_x]);
                }
            }
        }
    }
}
