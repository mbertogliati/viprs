#![allow(missing_docs)]
// REASON: these bridge helpers are public only for cross-crate workspace wiring, not end-user API.

use std::{any::Any, marker::PhantomData};

use viprs_core::{
    format::{BandFormat, BandFormatId},
    image::{DemandHint, Region, Tile, TileMut},
    op::{DynOperation, NodeSpec, Op, OperationBridge},
};

/// Tile an image `across` by `down` times.
pub struct Replicate<F: BandFormat> {
    inner: viprs_ops_spatial::structural::replicate::Replicate<F>,
    across: u32,
    down: u32,
    _format: PhantomData<F>,
}

impl<F: BandFormat + Send + Sync> Replicate<F> {
    #[must_use]
    /// Creates a new `Replicate`.
    pub fn new(image_width: u32, image_height: u32, across: u32, down: u32) -> Self {
        Self {
            inner: viprs_ops_spatial::structural::replicate::Replicate::new(
                image_width,
                image_height,
                across,
                down,
            ),
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

pub struct ReplicateBridge<F: BandFormat>
where
    F::Sample: bytemuck::Pod + Copy,
{
    inner: OperationBridge<Replicate<F>>,
}

impl<F: BandFormat> ReplicateBridge<F>
where
    F::Sample: bytemuck::Pod + Copy,
{
    #[must_use]
    pub fn new(image_width: u32, image_height: u32, across: u32, down: u32, bands: u32) -> Self {
        Self {
            inner: OperationBridge::new(
                Replicate::new(image_width, image_height, across, down),
                bands,
            ),
        }
    }
}

impl<F: BandFormat> DynOperation for ReplicateBridge<F>
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
        input_w.saturating_mul(self.inner.op.across)
    }

    fn output_height(&self, input_h: u32) -> u32 {
        input_h.saturating_mul(self.inner.op.down)
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
    use proptest::prelude::*;
    use viprs_core::{format::U8, op::DynOperation};

    fn run_replicate(op: &Replicate<U8>, pixels: &[u8], output_region: Region) -> Vec<u8> {
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
        let bridge = ReplicateBridge::<U8>::new(4, 3, 3, 2, 1);
        assert_eq!(bridge.output_width(4), 12);
        assert_eq!(bridge.output_height(3), 6);
    }

    #[test]
    fn tile_crossing_boundary_wraps_to_source_modulo() {
        let op = Replicate::<U8>::new(2, 2, 2, 2);
        let output = run_replicate(&op, &[1u8, 2, 3, 4], Region::new(1, 1, 2, 2));
        assert_eq!(output, vec![4u8, 3, 2, 1]);
    }

    proptest! {
        #[test]
        fn replicate_1x1_is_identity(rows in 1usize..=6, cols in 1usize..=6) {
            let pixels = (0..rows * cols).map(|idx| (idx % 251) as u8).collect::<Vec<_>>();
            let op = Replicate::<U8>::new(cols as u32, rows as u32, 1, 1);
            let output = run_replicate(&op, &pixels, Region::new(0, 0, cols as u32, rows as u32));
            prop_assert_eq!(output, pixels);
        }

        #[test]
        fn single_pixel_source_fills_replicated_output(
            value in 0u8..=255,
            across in 1u32..=8,
            down in 1u32..=8,
        ) {
            let op = Replicate::<U8>::new(1, 1, across, down);
            let output = run_replicate(&op, &[value], Region::new(0, 0, across, down));
            prop_assert!(output.iter().all(|&sample| sample == value));
        }
    }
}
