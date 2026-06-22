use std::{any::Any, marker::PhantomData};

use viprs_core::{
    format::BandFormat,
    image::{DemandHint, Region, Tile, TileMut},
    op::{NodeSpec, Op, OperationBridge},
};

/// Unfold bands into horizontal pixels.
///
/// # Examples
/// ```ignore
/// use viprs_ops_composite::conversion::bandunfold::BandunfoldOp;
///
/// let op = BandunfoldOp::new(/* operation parameters */);
/// // Run `op` through a compiled image pipeline.
/// ```
pub struct BandunfoldOp<F: BandFormat> {
    factor: u32,
    _format: PhantomData<F>,
}

impl<F: BandFormat> BandunfoldOp<F> {
    #[must_use]
    /// Creates a new `BandunfoldOp`.
    pub fn new(factor: u32) -> Self {
        debug_assert!(factor > 0, "BandunfoldOp: factor must be >= 1");
        Self {
            factor,
            _format: PhantomData,
        }
    }
}

impl<F> Op for BandunfoldOp<F>
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
        if output.width == 0 {
            return Region::new(
                output.x.div_euclid(self.factor as i32),
                output.y,
                0,
                output.height,
            );
        }

        let start_x = output.x.div_euclid(self.factor as i32);
        let end_x = (output.x + output.width as i32 - 1).div_euclid(self.factor as i32);
        Region::new(
            start_x,
            output.y,
            (end_x - start_x + 1) as u32,
            output.height,
        )
    }

    fn node_spec(&self, tile_w: u32, tile_h: u32) -> NodeSpec {
        NodeSpec {
            input_tile_w: tile_w.div_ceil(self.factor),
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
            input.region.height, output.region.height,
            "BandunfoldOp preserves image height"
        );
        debug_assert_eq!(
            input.bands % self.factor,
            0,
            "BandunfoldOp input bands must be divisible by factor"
        );

        let output_bands = (input.bands / self.factor) as usize;
        debug_assert_eq!(
            output.bands as usize, output_bands,
            "BandunfoldOp output bands must equal input bands / factor"
        );

        let input_row_samples = input.region.width as usize * input.bands as usize;
        let output_row_samples = output.region.width as usize * output.bands as usize;
        let band_offset = output.region.x.rem_euclid(self.factor as i32) as usize * output_bands;

        for row in 0..output.region.height as usize {
            let src_base = row * input_row_samples + band_offset;
            let dst_base = row * output_row_samples;
            output.data[dst_base..dst_base + output_row_samples]
                .copy_from_slice(&input.data[src_base..src_base + output_row_samples]);
        }
    }
}

/// Represents a bandunfold bridge.
pub struct BandunfoldBridge<F: BandFormat>
where
    F::Sample: bytemuck::Pod + Copy,
{
    inner: OperationBridge<BandunfoldOp<F>>,
}

impl<F: BandFormat> BandunfoldBridge<F>
where
    F::Sample: bytemuck::Pod + Copy,
{
    #[must_use]
    /// Creates a new `BandunfoldBridge`.
    pub fn new(factor: u32, input_bands: u32) -> Self {
        debug_assert!(factor > 0, "BandunfoldBridge: factor must be >= 1");
        debug_assert_eq!(
            input_bands % factor,
            0,
            "BandunfoldBridge: factor must divide the input band count"
        );
        Self {
            inner: OperationBridge::with_dynamic_bands(
                BandunfoldOp::new(factor),
                input_bands,
                input_bands / factor,
            ),
        }
    }
}

impl<F> viprs_core::op::DynOperation for BandunfoldBridge<F>
where
    F: BandFormat + Send + Sync,
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

    fn output_width(&self, input_width: u32) -> u32 {
        input_width * self.inner.op.factor
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

#[cfg(all(test, feature = "_integration"))]
pub(crate) mod tests {
    use super::*;
    use proptest::prelude::*;
    use viprs_core::format::U8;
    use viprs_ports::scheduler::TileScheduler;
    use viprs_runtime::{
        pipeline::PipelineBuilder, scheduler::rayon_scheduler::RayonScheduler,
        sinks::memory::MemorySink, sources::memory::MemorySource,
    };

    pub fn run_bandunfold_for_bandfold_roundtrip(
        factor: u32,
        input_width: u32,
        height: u32,
        bands: u32,
        input_data: &[u8],
    ) -> Vec<u8> {
        run_bandunfold(factor, input_width, height, bands, input_data)
    }

    fn run_bandunfold(
        factor: u32,
        input_width: u32,
        height: u32,
        bands: u32,
        input_data: &[u8],
    ) -> Vec<u8> {
        let op = BandunfoldOp::<U8>::new(factor);
        let output_region = Region::new(0, 0, input_width * factor, height);
        let input_region = op.required_input_region(&output_region);
        let input = Tile::<U8>::new(input_region, bands, input_data);
        let mut output_data = vec![0u8; output_region.pixel_count() * (bands / factor) as usize];
        let mut output = TileMut::<U8>::new(output_region, bands / factor, &mut output_data);
        let mut state = ();
        op.process_region(&mut state, &input, &mut output);
        output_data
    }

    #[test]
    fn unfold_by_two_on_four_band_row() {
        let input = [
            1u8, 10, 2, 20, //
            3, 30, 4, 40,
        ];
        let output = run_bandunfold(2, 2, 1, 4, &input);
        assert_eq!(output, vec![1, 10, 2, 20, 3, 30, 4, 40]);
    }

    #[test]
    fn bridge_reports_unfolded_geometry_and_runs_pipeline() {
        let source = MemorySource::<U8>::new(2, 1, 4, vec![1, 10, 2, 20, 3, 30, 4, 40])
            .expect("valid source");
        let pipeline = PipelineBuilder::from_source(source)
            .then(Box::new(BandunfoldBridge::<U8>::new(2, 4)))
            .expect("bandunfold op")
            .build()
            .expect("compiled pipeline");
        let mut sink = MemorySink::for_pipeline(&pipeline).unwrap();
        RayonScheduler::new(1)
            .expect("scheduler")
            .run(&pipeline, &mut sink)
            .expect("bandunfold execution");

        assert_eq!(pipeline.width, 4);
        assert_eq!(pipeline.height, 1);
        assert_eq!(pipeline.output_bands, 2);
        assert_eq!(sink.into_buffer(), vec![1, 10, 2, 20, 3, 30, 4, 40]);
    }

    #[test]
    fn required_input_region_rounds_to_cover_partial_fold_groups() {
        let op = BandunfoldOp::<U8>::new(2);
        assert_eq!(
            op.required_input_region(&Region::new(1, 3, 4, 2)),
            Region::new(0, 3, 3, 2)
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
            let output = run_bandunfold(1, width, height, bands, &input);
            prop_assert_eq!(output, input);
        }

        #[test]
        fn non_aligned_output_region_reads_correct_band_offset(
            width in 2u32..=6,
            height in 1u32..=4,
            bands in 1u32..=4,
        ) {
            let factor = 2u32;
            let input_bands = bands * factor;
            let input_len = (width * height * input_bands) as usize;
            let input = (0..input_len).map(|idx| (idx % 251) as u8).collect::<Vec<_>>();
            let op = BandunfoldOp::<U8>::new(factor);
            let output_region = Region::new(1, 0, width * factor - 1, height);
            let input_region = op.required_input_region(&output_region);
            let input_tile = Tile::<U8>::new(input_region, input_bands, &input[..input_region.pixel_count() * input_bands as usize]);
            let mut output_data = vec![0u8; output_region.pixel_count() * bands as usize];
            let mut output_tile = TileMut::<U8>::new(output_region, bands, &mut output_data);
            let mut state = ();
            op.process_region(&mut state, &input_tile, &mut output_tile);

            let input_row_samples = width as usize * input_bands as usize;
            let output_row_samples = (width * factor - 1) as usize * bands as usize;
            let mut expected = Vec::with_capacity(output_data.len());
            for row in 0..height as usize {
                let row_start = row * input_row_samples + bands as usize;
                expected.extend_from_slice(&input[row_start..row_start + output_row_samples]);
            }
            prop_assert_eq!(output_data, expected);
        }
    }
}
