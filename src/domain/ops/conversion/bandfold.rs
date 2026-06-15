use std::{any::Any, marker::PhantomData};

use crate::{
    domain::op::{NodeSpec, Op, OperationBridge},
    domain::{
        format::BandFormat,
        image::{DemandHint, Region, Tile, TileMut},
    },
};

/// Fold horizontal pixels into bands.
///
/// # Examples
/// ```ignore
/// use viprs::domain::ops::conversion::bandfold::BandfoldOp;
///
/// let op = BandfoldOp::new(/* operation parameters */);
/// // Run `op` through a compiled image pipeline.
/// ```
pub struct BandfoldOp<F: BandFormat> {
    factor: u32,
    _format: PhantomData<F>,
}

impl<F: BandFormat> BandfoldOp<F> {
    #[must_use]
    /// Creates a new `BandfoldOp`.
    pub fn new(factor: u32) -> Self {
        debug_assert!(factor > 0, "BandfoldOp: factor must be >= 1");
        Self {
            factor,
            _format: PhantomData,
        }
    }
}

impl<F> Op for BandfoldOp<F>
where
    F: BandFormat,
    F::Sample: Copy,
{
    type Input = F;
    type Output = F;
    type State = ();

    fn demand_hint(&self) -> DemandHint {
        DemandHint::ThinStrip
    }

    fn required_input_region(&self, output: &Region) -> Region {
        Region::new(
            output.x * self.factor as i32,
            output.y,
            output.width * self.factor,
            output.height,
        )
    }

    fn node_spec(&self, tile_w: u32, tile_h: u32) -> NodeSpec {
        NodeSpec {
            input_tile_w: tile_w * self.factor,
            input_tile_h: tile_h,
            output_tile_w: tile_w,
            output_tile_h: tile_h,
            coordinate_driven_source: None,
        }
    }

    fn start(&self) {}

    #[inline]
    fn process_region(&self, _state: &mut (), input: &Tile<F>, output: &mut TileMut<F>) {
        debug_assert_eq!(
            input.region.width,
            output.region.width * self.factor,
            "BandfoldOp input width must equal output width * factor"
        );
        debug_assert_eq!(
            input.region.height, output.region.height,
            "BandfoldOp preserves image height"
        );
        debug_assert_eq!(
            output.bands,
            input.bands * self.factor,
            "BandfoldOp output bands must equal input bands * factor"
        );

        let input_row_samples = input.region.width as usize * input.bands as usize;
        let output_row_samples = output.region.width as usize * output.bands as usize;

        for row in 0..output.region.height as usize {
            let src_base = row * input_row_samples;
            let dst_base = row * output_row_samples;
            output.data[dst_base..dst_base + output_row_samples]
                .copy_from_slice(&input.data[src_base..src_base + output_row_samples]);
        }
    }
}

/// Represents a bandfold bridge.
pub struct BandfoldBridge<F: BandFormat>
where
    F::Sample: bytemuck::Pod + Copy,
{
    inner: OperationBridge<BandfoldOp<F>>,
}

impl<F: BandFormat> BandfoldBridge<F>
where
    F::Sample: bytemuck::Pod + Copy,
{
    #[must_use]
    /// Creates a new `BandfoldBridge`.
    pub fn new(factor: u32, input_width: u32, input_bands: u32) -> Self {
        debug_assert!(factor > 0, "BandfoldBridge: factor must be >= 1");
        debug_assert_eq!(
            input_width % factor,
            0,
            "BandfoldBridge: factor must divide the input width"
        );
        Self {
            inner: OperationBridge::with_dynamic_bands(
                BandfoldOp::new(factor),
                input_bands,
                input_bands * factor,
            ),
        }
    }
}

impl<F> crate::domain::op::DynOperation for BandfoldBridge<F>
where
    F: BandFormat + Send + Sync,
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

    fn output_width(&self, input_width: u32) -> u32 {
        input_width / self.inner.op.factor
    }

    fn dyn_start(&self) -> Box<dyn Any + Send> {
        self.inner.dyn_start()
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
    use crate::{
        adapters::{
            pipeline::PipelineBuilder, scheduler::rayon_scheduler::RayonScheduler,
            sinks::memory::MemorySink, sources::memory::MemorySource,
        },
        domain::format::U8,
        ports::scheduler::TileScheduler,
    };
    use proptest::prelude::*;

    fn run_bandfold(
        factor: u32,
        input_width: u32,
        height: u32,
        bands: u32,
        input_data: &[u8],
    ) -> Vec<u8> {
        let op = BandfoldOp::<U8>::new(factor);
        let output_region = Region::new(0, 0, input_width / factor, height);
        let input_region = op.required_input_region(&output_region);
        let input = Tile::<U8>::new(input_region, bands, input_data);
        let mut output_data = vec![0u8; output_region.pixel_count() * (bands * factor) as usize];
        let mut output = TileMut::<U8>::new(output_region, bands * factor, &mut output_data);
        let mut state = ();
        op.process_region(&mut state, &input, &mut output);
        output_data
    }

    #[test]
    fn fold_by_two_on_two_band_row() {
        let input = [
            1u8, 10, 2, 20, //
            3, 30, 4, 40,
        ];
        let output = run_bandfold(2, 4, 1, 2, &input);
        assert_eq!(output, vec![1, 10, 2, 20, 3, 30, 4, 40]);
    }

    #[test]
    fn bridge_reports_folded_geometry_and_runs_pipeline() {
        let source = MemorySource::<U8>::new(4, 1, 2, vec![1, 10, 2, 20, 3, 30, 4, 40])
            .expect("valid source");
        let pipeline = PipelineBuilder::from_source(source)
            .then(Box::new(BandfoldBridge::<U8>::new(2, 4, 2)))
            .expect("bandfold op")
            .build()
            .expect("compiled pipeline");
        let mut sink = MemorySink::for_pipeline(&pipeline).unwrap();
        RayonScheduler::new(1)
            .expect("scheduler")
            .run(&pipeline, &mut sink)
            .expect("bandfold execution");

        assert_eq!(pipeline.width, 2);
        assert_eq!(pipeline.height, 1);
        assert_eq!(pipeline.output_bands, 4);
        assert_eq!(sink.into_buffer(), vec![1, 10, 2, 20, 3, 30, 4, 40]);
    }

    #[test]
    fn required_input_region_scales_horizontally() {
        let op = BandfoldOp::<U8>::new(3);
        assert_eq!(
            op.required_input_region(&Region::new(2, 4, 5, 6)),
            Region::new(6, 4, 15, 6)
        );
    }

    proptest! {
        #[test]
        fn factor_one_is_identity(
            width in 1u32..=8,
            height in 1u32..=4,
            bands in 1u32..=4,
        ) {
            let len = (width * height * bands) as usize;
            let input = (0..len).map(|idx| (idx % 251) as u8).collect::<Vec<_>>();
            let output = run_bandfold(1, width, height, bands, &input);
            prop_assert_eq!(output, input);
        }

        #[test]
        fn fold_then_unfold_is_identity(
            output_width in 1u32..=6,
            height in 1u32..=4,
            bands in 1u32..=4,
            factor in 1u32..=4,
        ) {
            let input_width = output_width * factor;
            let len = (input_width * height * bands) as usize;
            let input = (0..len).map(|idx| (idx % 251) as u8).collect::<Vec<_>>();

            let folded = run_bandfold(factor, input_width, height, bands, &input);
            let unfolded = super::super::bandunfold::tests::run_bandunfold_for_bandfold_roundtrip(
                factor,
                output_width,
                height,
                bands * factor,
                &folded,
            );

            prop_assert_eq!(unfolded, input);
        }
    }
}
