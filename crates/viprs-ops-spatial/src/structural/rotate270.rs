#![allow(dead_code)]
// REASON: structural bridge wiring is staged for future pipeline-builder exposure.

use std::marker::PhantomData;
use viprs_core::{
    format::BandFormat,
    image::{DemandHint, Region, Tile, TileMut, clamp_i64_to_i32},
    op::{NodeSpec, Op, OperationBridge},
};

/// Counter-clockwise 90° rotation (270° clockwise).
pub struct Rotate270<F: BandFormat> {
    image_width: u32,
    image_height: u32,
    _format: PhantomData<F>,
}

impl<F: BandFormat + Send + Sync> Rotate270<F> {
    #[must_use]
    /// Creates a new `Rotate270`.
    pub const fn new(image_width: u32, image_height: u32) -> Self {
        Self {
            image_width,
            image_height,
            _format: PhantomData,
        }
    }
}

impl<F: BandFormat> Op for Rotate270<F>
where
    F::Sample: Copy,
{
    type Input = F;
    type Output = F;
    type State = ();

    fn preferred_tile_geometry(&self) -> DemandHint {
        DemandHint::SmallTile
    }

    fn required_input_region(&self, output: &Region) -> Region {
        Region::new(
            clamp_i64_to_i32(
                i64::from(self.image_width) - i64::from(output.y) - i64::from(output.height),
            ),
            output.x,
            output.height,
            output.width,
        )
    }

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

    #[inline]
    fn process_region(&self, _state: &mut (), input: &Tile<F>, output: &mut TileMut<F>) {
        let bands = input.bands as usize;
        let ow = output.region.width as usize;
        let oh = output.region.height as usize;
        let input_row_stride = oh * bands;

        for row in 0..oh {
            for col in 0..ow {
                let input_col = oh - 1 - row;
                let input_row = col;
                let src = input_row * input_row_stride + input_col * bands;
                let dst = (row * ow + col) * bands;
                output.data[dst..dst + bands].copy_from_slice(&input.data[src..src + bands]);
            }
        }
    }
}

pub(crate) struct Rotate270Bridge<F: BandFormat>
where
    F::Sample: bytemuck::Pod + Copy,
{
    inner: OperationBridge<Rotate270<F>>,
}

impl<F: BandFormat> Rotate270Bridge<F>
where
    F::Sample: bytemuck::Pod + Copy,
{
    pub fn new(image_width: u32, image_height: u32, bands: u32) -> Self {
        Self {
            inner: OperationBridge::new(Rotate270::new(image_width, image_height), bands),
        }
    }
}

impl<F: BandFormat> viprs_core::op::DynOperation for Rotate270Bridge<F>
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
    use proptest::prelude::*;
    use viprs_core::{
        format::{BandFormatId, U8},
        image::{Region, Tile, TileMut},
        op::DynOperation,
    };

    #[test]
    fn required_input_region_transposes_and_reflects() {
        let op = Rotate270::<U8>::new(8, 6);
        let output = Region::new(1, 2, 3, 2);
        let input = op.required_input_region(&output);
        assert_eq!(input, Region::new(4, 1, 2, 3));
    }

    #[test]
    fn required_input_region_clamps_large_image_width() {
        let op = Rotate270::<U8>::new(u32::MAX, 1);
        let output = Region::new(0, 0, 1, 1);
        let input = op.required_input_region(&output);
        assert_eq!(input, Region::new(i32::MAX, 0, 1, 1));
    }

    #[test]
    fn output_dimensions_transpose() {
        let bridge = Rotate270Bridge::<U8>::new(8, 6, 1);
        assert_eq!(bridge.output_width(8), 6);
        assert_eq!(bridge.output_height(6), 8);
    }

    #[test]
    fn process_region_rotates_2x2() {
        let op = Rotate270::<U8>::new(2, 2);
        let output_region = Region::new(0, 0, 2, 2);
        let input_region = op.required_input_region(&output_region);
        let input_data = vec![1u8, 2, 3, 4];
        let mut output_data = vec![0u8; 4];
        let input = Tile::<U8>::new(input_region, 1, &input_data);
        let mut output = TileMut::<U8>::new(output_region, 1, &mut output_data);
        let mut state = ();
        op.process_region(&mut state, &input, &mut output);
        assert_eq!(output_data, vec![2u8, 4, 1, 3]);
    }

    #[test]
    fn rotate270_bridge_processes_bytes_and_reports_metadata() {
        let bridge = Rotate270Bridge::<U8>::new(2, 3, 1);
        let output_region = Region::new(0, 0, 3, 2);
        let input_region = bridge.required_input_region(&output_region);
        let input_data = [1u8, 2, 3, 4, 5, 6];
        let mut output_data = [0u8; 6];
        let mut state = bridge.dyn_start();

        assert_eq!(bridge.input_format(), BandFormatId::U8);
        assert_eq!(bridge.output_format(), BandFormatId::U8);
        assert_eq!(
            bridge.demand_hint(),
            viprs_core::image::DemandHint::SmallTile
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
        assert_eq!(output_data, [2u8, 4, 6, 1, 3, 5]);
    }

    proptest! {
        #[test]
        fn rotate_then_rotate90_is_identity(rows in 1usize..=6, cols in 1usize..=6) {
            let pixels: Vec<u8> = (0..(rows * cols) as u8).collect();

            let rotate270 = |data: &[u8], w: usize, h: usize| -> Vec<u8> {
                let op = Rotate270::<U8>::new(w as u32, h as u32);
                let out_w = h;
                let out_h = w;
                let output_region = Region::new(0, 0, out_w as u32, out_h as u32);
                let input_region = op.required_input_region(&output_region);
                let mut result = vec![0u8; out_w * out_h];
                let input = Tile::<U8>::new(input_region, 1, data);
                let mut output = TileMut::<U8>::new(output_region, 1, &mut result);
                let mut state = ();
                op.process_region(&mut state, &input, &mut output);
                result
            };

            let rotate90 = |data: &[u8], w: usize, h: usize| -> Vec<u8> {
                let op = crate::structural::rotate90::Rotate90::<U8>::new(w as u32, h as u32);
                let out_w = h;
                let out_h = w;
                let output_region = Region::new(0, 0, out_w as u32, out_h as u32);
                let input_region = op.required_input_region(&output_region);
                let mut result = vec![0u8; out_w * out_h];
                let input = Tile::<U8>::new(input_region, 1, data);
                let mut output = TileMut::<U8>::new(output_region, 1, &mut result);
                let mut state = ();
                op.process_region(&mut state, &input, &mut output);
                result
            };

            let ccw = rotate270(&pixels, cols, rows);
            let back = rotate90(&ccw, rows, cols);
            prop_assert_eq!(back, pixels);
        }
    }
}
