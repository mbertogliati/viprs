use std::{any::Any, marker::PhantomData};

use crate::{
    domain::op::{DynOperation, NodeSpec, Op, OperationBridge},
    domain::{
        format::{BandFormat, BandFormatId},
        image::{DemandHint, Region, Tile, TileMut},
    },
};

/// Integer nearest-neighbour upscale by `xfac` by `yfac`.
pub struct Zoom<F: BandFormat> {
    inner: crate::domain::ops::structural::zoom::Zoom<F>,
    xfac: u32,
    yfac: u32,
    _format: PhantomData<F>,
}

impl<F: BandFormat + Send + Sync> Zoom<F> {
    #[must_use]
    /// Creates a new `Zoom`.
    pub fn new(xfac: u32, yfac: u32) -> Self {
        Self {
            inner: crate::domain::ops::structural::zoom::Zoom::new(xfac, yfac),
            xfac,
            yfac,
            _format: PhantomData,
        }
    }
}

impl<F: BandFormat> Op for Zoom<F>
where
    F::Sample: Copy,
{
    type Input = F;
    type Output = F;
    type State = ();

    fn demand_hint(&self) -> DemandHint {
        self.inner.demand_hint()
    }

    fn required_input_region(&self, output: &Region) -> Region {
        self.inner.required_input_region(output)
    }

    fn node_spec(&self, tile_w: u32, tile_h: u32) -> NodeSpec {
        self.inner.node_spec(tile_w, tile_h)
    }

    fn start(&self) {}

    #[inline]
    fn process_region(&self, state: &mut (), input: &Tile<F>, output: &mut TileMut<F>) {
        self.inner.process_region(state, input, output);
    }
}

pub(crate) struct ZoomBridge<F: BandFormat>
where
    F::Sample: bytemuck::Pod + Copy,
{
    inner: OperationBridge<Zoom<F>>,
}

impl<F: BandFormat> ZoomBridge<F>
where
    F::Sample: bytemuck::Pod + Copy,
{
    pub fn new(xfac: u32, yfac: u32, bands: u32) -> Self {
        Self {
            inner: OperationBridge::new(Zoom::new(xfac, yfac), bands),
        }
    }
}

impl<F: BandFormat> DynOperation for ZoomBridge<F>
where
    F::Sample: bytemuck::Pod + Copy + Send,
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

    fn required_input_region(&self, output: &Region) -> Region {
        self.inner.required_input_region(output)
    }

    fn node_spec(&self, tile_w: u32, tile_h: u32) -> NodeSpec {
        self.inner.node_spec(tile_w, tile_h)
    }

    fn output_width(&self, input_w: u32) -> u32 {
        input_w.saturating_mul(self.inner.op.xfac)
    }

    fn output_height(&self, input_h: u32) -> u32 {
        input_h.saturating_mul(self.inner.op.yfac)
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::domain::{format::U8, op::DynOperation};
    use proptest::prelude::*;

    fn run_zoom(op: &Zoom<U8>, pixels: &[u8], output_region: Region) -> Vec<u8> {
        let input_region = op.required_input_region(&output_region);
        let mut output = vec![0u8; output_region.pixel_count()];
        let input = Tile::<U8>::new(input_region, 1, pixels);
        let mut output_tile = TileMut::<U8>::new(output_region, 1, &mut output);
        let mut state = ();
        op.process_region(&mut state, &input, &mut output_tile);
        output
    }

    #[test]
    fn bridge_scales_dimensions() {
        let bridge = ZoomBridge::<U8>::new(3, 2, 1);
        assert_eq!(bridge.output_width(4), 12);
        assert_eq!(bridge.output_height(5), 10);
    }

    #[test]
    fn boundary_single_pixel_expands_to_uniform_block() {
        let op = Zoom::<U8>::new(3, 2);
        let output = run_zoom(&op, &[7u8], Region::new(0, 0, 3, 2));
        assert_eq!(output, vec![7u8; 6]);
    }

    proptest! {
        #[test]
        fn factor_one_is_identity(rows in 1usize..=8, cols in 1usize..=8) {
            let pixels = (0..rows * cols).map(|idx| (idx % 251) as u8).collect::<Vec<_>>();
            let op = Zoom::<U8>::new(1, 1);
            let output = run_zoom(&op, &pixels, Region::new(0, 0, cols as u32, rows as u32));
            prop_assert_eq!(output, pixels);
        }

        #[test]
        fn first_source_pixel_repeats_over_top_left_block(
            value in 0u8..=255,
            xfac in 1u32..=8,
            yfac in 1u32..=8,
        ) {
            let op = Zoom::<U8>::new(xfac, yfac);
            let output = run_zoom(&op, &[value], Region::new(0, 0, xfac, yfac));
            prop_assert!(output.iter().all(|&sample| sample == value));
        }
    }
}
