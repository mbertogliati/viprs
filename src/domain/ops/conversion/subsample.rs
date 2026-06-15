use std::{any::Any, marker::PhantomData};

use crate::{
    domain::op::{DynOperation, NodeSpec, Op, OperationBridge, SourceReadPlan},
    domain::{
        error::BuildError,
        format::{BandFormat, BandFormatId},
        image::{DemandHint, Region, Tile, TileMut},
    },
};

fn validate_subsample_factors(xfac: u32, yfac: u32) -> Result<(), BuildError> {
    if xfac == 0 || yfac == 0 {
        return Err(BuildError::SourceHint {
            context: "subsample",
            message: "xfac and yfac must be >= 1".to_string(),
        });
    }

    Ok(())
}

/// Integer nearest-neighbour decimation by taking every `xfac`-th and `yfac`-th pixel.
///
/// # Examples
/// ```ignore
/// use viprs::domain::ops::conversion::subsample::SubsampleOp;
///
/// let op = SubsampleOp::new(/* operation parameters */);
/// // Run `op` through a compiled image pipeline.
/// ```
pub struct SubsampleOp<F: BandFormat> {
    xfac: u32,
    yfac: u32,
    point: bool,
    _format: PhantomData<F>,
}

impl<F: BandFormat + Send + Sync> SubsampleOp<F> {
    /// Creates a new `SubsampleOp`.
    pub fn new(xfac: u32, yfac: u32) -> Result<Self, BuildError> {
        Self::with_point(xfac, yfac, false)
    }

    /// Returns this value configured with point.
    pub fn with_point(xfac: u32, yfac: u32, point: bool) -> Result<Self, BuildError> {
        validate_subsample_factors(xfac, yfac)?;
        Ok(Self {
            xfac,
            yfac,
            point,
            _format: PhantomData,
        })
    }

    #[must_use]
    /// Returns or performs xfac.
    pub const fn xfac(&self) -> u32 {
        self.xfac
    }

    #[must_use]
    /// Returns or performs yfac.
    pub const fn yfac(&self) -> u32 {
        self.yfac
    }

    #[must_use]
    /// Returns or performs point.
    pub const fn point(&self) -> bool {
        self.point
    }

    const fn point_input_region(&self, output: &Region) -> Region {
        Region::new(
            output.x * self.xfac as i32,
            output.y * self.yfac as i32,
            output.width,
            output.height,
        )
    }

    const fn bounding_input_region(&self, output: &Region) -> Region {
        if output.is_empty() {
            return Region::new(
                output.x * self.xfac as i32,
                output.y * self.yfac as i32,
                0,
                0,
            );
        }

        Region::new(
            output.x * self.xfac as i32,
            output.y * self.yfac as i32,
            output.width * self.xfac - (self.xfac - 1),
            output.height * self.yfac - (self.yfac - 1),
        )
    }
}

impl<F: BandFormat> Op for SubsampleOp<F>
where
    F::Sample: Copy,
{
    type Input = F;
    type Output = F;
    type State = ();

    fn demand_hint(&self) -> DemandHint {
        DemandHint::ThinStrip
    }

    fn required_input_region(&self, output: &Region) -> Region {
        if self.point {
            self.point_input_region(output)
        } else {
            self.bounding_input_region(output)
        }
    }

    fn node_spec(&self, tile_w: u32, tile_h: u32) -> NodeSpec {
        if tile_w == 0 || tile_h == 0 {
            return NodeSpec {
                input_tile_w: 0,
                input_tile_h: 0,
                output_tile_w: tile_w,
                output_tile_h: tile_h,
                coordinate_driven_source: None,
            };
        }

        NodeSpec {
            input_tile_w: (tile_w * self.xfac).saturating_sub(self.xfac - 1),
            input_tile_h: (tile_h * self.yfac).saturating_sub(self.yfac - 1),
            output_tile_w: tile_w,
            output_tile_h: tile_h,
            coordinate_driven_source: None,
        }
    }

    fn start(&self) {}

    #[inline]
    fn process_region(&self, _state: &mut (), input: &Tile<F>, output: &mut TileMut<F>) {
        debug_assert_eq!(
            input.bands, output.bands,
            "SubsampleOp preserves band count"
        );

        if self.point
            && input.region.width == output.region.width
            && input.region.height == output.region.height
        {
            debug_assert_eq!(
                input.data.len(),
                output.data.len(),
                "packed point-mode subsample input must match output length"
            );
            output.data.copy_from_slice(input.data);
            return;
        }

        let bands = input.bands as usize;
        let input_width = input.region.width as usize;
        let output_width = output.region.width as usize;
        let output_height = output.region.height as usize;
        let xfac = self.xfac as usize;
        let yfac = self.yfac as usize;

        for row in 0..output_height {
            let src_row = row * yfac;
            for col in 0..output_width {
                let src_col = col * xfac;
                let src = (src_row * input_width + src_col) * bands;
                let dst = (row * output_width + col) * bands;
                output.data[dst..dst + bands].copy_from_slice(&input.data[src..src + bands]);
            }
        }
    }
}

/// Dynamic bridge that reports subsampled output dimensions.
pub struct SubsampleBridge<F: BandFormat>
where
    F::Sample: bytemuck::Pod + Copy,
{
    inner: OperationBridge<SubsampleOp<F>>,
}

impl<F: BandFormat> SubsampleBridge<F>
where
    F::Sample: bytemuck::Pod + Copy,
{
    /// Creates a new `SubsampleBridge`.
    pub fn new(xfac: u32, yfac: u32, bands: u32) -> Result<Self, BuildError> {
        Self::with_point(xfac, yfac, bands, false)
    }

    /// Returns this value configured with point.
    pub fn with_point(xfac: u32, yfac: u32, bands: u32, point: bool) -> Result<Self, BuildError> {
        Ok(Self {
            inner: OperationBridge::new(SubsampleOp::with_point(xfac, yfac, point)?, bands),
        })
    }
}

impl<F: BandFormat> DynOperation for SubsampleBridge<F>
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

    fn required_input_region_slot(&self, output: &Region, slot: usize) -> Region {
        debug_assert_eq!(slot, 0);
        self.inner.required_input_region(output)
    }

    fn source_read_plan_slot(&self, output: &Region, slot: usize) -> SourceReadPlan {
        debug_assert_eq!(slot, 0);
        if self.inner.op.point {
            SourceReadPlan::PointGrid {
                input_region: *output,
                source_origin_x: output.x * self.inner.op.xfac as i32,
                source_origin_y: output.y * self.inner.op.yfac as i32,
                x_step: self.inner.op.xfac,
                y_step: self.inner.op.yfac,
            }
        } else {
            SourceReadPlan::rect(self.required_input_region(output))
        }
    }

    fn node_spec(&self, tile_w: u32, tile_h: u32) -> NodeSpec {
        self.inner.node_spec(tile_w, tile_h)
    }

    fn output_width(&self, input_w: u32) -> u32 {
        input_w / self.inner.op.xfac
    }

    fn output_height(&self, input_h: u32) -> u32 {
        input_h / self.inner.op.yfac
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
        if self.inner.op.point && input_region == output_region {
            debug_assert_eq!(
                input.len(),
                output.len(),
                "packed point-mode subsample input must match output length"
            );
            output.copy_from_slice(input);
            return;
        }

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
        domain::{
            error::{BuildError, ViprsError},
            format::{BandFormatId, U8},
            image::{DemandHint, Region, Tile, TileMut},
            op::{DynOperation, OperationBridge, SourceReadPlan},
            ops::arithmetic::invert::Invert,
        },
        ports::{scheduler::TileScheduler, source::ImageSource},
    };
    use proptest::prelude::*;
    use std::sync::{Arc, Mutex};

    fn expected_subsample(
        input: &[u8],
        input_width: usize,
        bands: usize,
        output_width: usize,
        output_height: usize,
        xfac: usize,
        yfac: usize,
    ) -> Vec<u8> {
        let mut expected = vec![0u8; output_width * output_height * bands];

        for row in 0..output_height {
            let src_row = row * yfac;
            for col in 0..output_width {
                let src_col = col * xfac;
                let src = (src_row * input_width + src_col) * bands;
                let dst = (row * output_width + col) * bands;
                expected[dst..dst + bands].copy_from_slice(&input[src..src + bands]);
            }
        }

        expected
    }

    fn run_subsample(
        xfac: u32,
        yfac: u32,
        input_width: u32,
        output_width: u32,
        output_height: u32,
        bands: u32,
        input_data: &[u8],
    ) -> Vec<u8> {
        let op = SubsampleOp::<U8>::new(xfac, yfac).unwrap();
        let output_region = Region::new(0, 0, output_width, output_height);
        let input_region = op.required_input_region(&output_region);
        let input = Tile::<U8>::new(input_region, bands, input_data);
        let mut output_data = vec![0u8; output_region.pixel_count() * bands as usize];
        let mut output = TileMut::<U8>::new(output_region, bands, &mut output_data);
        let mut state = op.start();
        assert_eq!(input.region.width, input_width);
        op.process_region(&mut state, &input, &mut output);
        output_data
    }

    #[test]
    fn decimates_integer_grid() {
        let input = vec![1u8, 2, 3, 4, 5, 6, 7, 8, 9];
        let output = run_subsample(2, 2, 3, 2, 2, 1, &input);
        assert_eq!(output, vec![1u8, 3, 7, 9]);
    }

    #[test]
    fn required_input_region_matches_libvips_line_mode_bbox() {
        let op = SubsampleOp::<U8>::new(2, 3).unwrap();
        assert_eq!(
            op.required_input_region(&Region::new(1, 2, 4, 3)),
            Region::new(2, 6, 7, 7)
        );
    }

    #[test]
    fn point_mode_required_input_region_is_minimal_sample_rect() {
        let op = SubsampleOp::<U8>::with_point(12, 5, true).unwrap();
        assert!(op.point());
        assert_eq!(
            op.required_input_region(&Region::new(2, 3, 4, 2)),
            Region::new(24, 15, 4, 2)
        );
    }

    #[test]
    fn point_mode_source_read_plan_is_sampled_grid() {
        let bridge = SubsampleBridge::<U8>::with_point(12, 5, 1, true).unwrap();
        assert_eq!(
            bridge.source_read_plan_slot(&Region::new(2, 3, 4, 2), 0),
            SourceReadPlan::PointGrid {
                input_region: Region::new(2, 3, 4, 2),
                source_origin_x: 24,
                source_origin_y: 15,
                x_step: 12,
                y_step: 5,
            }
        );
    }

    #[test]
    fn point_mode_dynamic_slot_preserves_minimal_sample_rect() {
        let bridge = SubsampleBridge::<U8>::with_point(12, 5, 1, true).unwrap();
        assert_eq!(
            bridge.required_input_region_slot(&Region::new(2, 3, 4, 2), 0),
            Region::new(24, 15, 4, 2)
        );
    }

    #[test]
    fn bridge_reports_floor_output_dimensions() {
        let bridge = SubsampleBridge::<U8>::new(2, 3, 1).unwrap();
        assert_eq!(bridge.output_width(7), 3);
        assert_eq!(bridge.output_height(8), 2);
        assert_eq!(bridge.input_format(), BandFormatId::U8);
        assert_eq!(bridge.output_format(), BandFormatId::U8);
    }

    #[test]
    fn conversion_subsample_bridge_zero_x_factor_returns_error() {
        let err = match SubsampleBridge::<U8>::new(0, 1, 1) {
            Ok(_) => panic!("xfac=0 must be rejected"),
            Err(err) => err,
        };
        assert!(matches!(
            err,
            BuildError::SourceHint {
                context: "subsample",
                ..
            }
        ));
    }

    #[test]
    fn conversion_subsample_bridge_zero_y_factor_returns_error() {
        let err = match SubsampleBridge::<U8>::new(1, 0, 1) {
            Ok(_) => panic!("yfac=0 must be rejected"),
            Err(err) => err,
        };
        assert!(matches!(
            err,
            BuildError::SourceHint {
                context: "subsample",
                ..
            }
        ));
    }

    #[test]
    fn node_spec_uses_empty_input_tile_for_zero_sized_output() {
        let op = SubsampleOp::<U8>::with_point(12, 5, true).unwrap();
        let spec = op.node_spec(0, 1);
        assert_eq!(spec.input_tile_w, 0);
        assert_eq!(spec.input_tile_h, 0);
        assert_eq!(spec.output_tile_w, 0);
        assert_eq!(spec.output_tile_h, 1);
    }

    #[test]
    fn pipeline_uses_subsampled_dimensions() {
        let source = MemorySource::<U8>::new(4, 4, 1, (0u8..16).collect()).unwrap();
        let pipeline = PipelineBuilder::from_source(source)
            .then(Box::new(SubsampleBridge::<U8>::new(2, 2, 1).unwrap()))
            .unwrap()
            .build()
            .unwrap();
        let mut sink = MemorySink::for_pipeline(&pipeline).unwrap();
        RayonScheduler::new(1)
            .unwrap()
            .run(&pipeline, &mut sink)
            .unwrap();

        assert_eq!(pipeline.width, 2);
        assert_eq!(pipeline.height, 2);
        assert_eq!(sink.into_buffer(), vec![0u8, 2, 8, 10]);
    }

    struct RecordingSource {
        reads: Arc<Mutex<Vec<Region>>>,
    }

    impl ImageSource for RecordingSource {
        type Format = U8;

        fn width(&self) -> u32 {
            64
        }

        fn height(&self) -> u32 {
            32
        }

        fn bands(&self) -> u32 {
            1
        }

        fn demand_hint(&self) -> DemandHint {
            DemandHint::ThinStrip
        }

        fn read_region(&self, region: Region, output: &mut [u8]) -> Result<(), ViprsError> {
            self.reads.lock().unwrap().push(region);
            for row in 0..region.height {
                for col in 0..region.width {
                    let x = (region.x + col as i32).clamp(0, 63) as u8;
                    let y = (region.y + row as i32).clamp(0, 31) as u8;
                    output[row as usize * region.width as usize + col as usize] = x + y;
                }
            }
            Ok(())
        }
    }

    #[test]
    fn point_mode_pipeline_reads_single_source_pixels() {
        let reads = Arc::new(Mutex::new(Vec::new()));
        let source = RecordingSource {
            reads: Arc::clone(&reads),
        };
        let pipeline = PipelineBuilder::from_source(source)
            .then(Box::new(
                SubsampleBridge::<U8>::with_point(12, 5, 1, true).unwrap(),
            ))
            .unwrap()
            .build()
            .unwrap();
        let mut sink = MemorySink::for_pipeline(&pipeline).unwrap();

        RayonScheduler::new(1)
            .unwrap()
            .run(&pipeline, &mut sink)
            .unwrap();

        let reads = reads.lock().unwrap();
        assert_eq!(reads.len(), 30);
        assert!(
            reads
                .iter()
                .all(|region| region.width == 1 && region.height == 1)
        );
        assert_eq!(reads[0], Region::new(0, 0, 1, 1));
        assert_eq!(reads[1], Region::new(12, 0, 1, 1));
        assert_eq!(reads[5], Region::new(0, 5, 1, 1));
        assert!(!reads.contains(&Region::new(0, 0, 49, 26)));

        let expected = expected_subsample(
            &(0..32)
                .flat_map(|y| (0..64).map(move |x| (x + y) as u8))
                .collect::<Vec<_>>(),
            64,
            1,
            5,
            6,
            12,
            5,
        );
        assert_eq!(sink.into_buffer(), expected);
    }

    #[test]
    fn point_mode_pipeline_after_invert_reads_single_source_pixels() {
        let reads = Arc::new(Mutex::new(Vec::new()));
        let source = RecordingSource {
            reads: Arc::clone(&reads),
        };
        let pipeline = PipelineBuilder::from_source(source)
            .then(Box::new(OperationBridge::new_pixel_local(
                Invert::<U8>::new(),
                1,
            )))
            .unwrap()
            .then(Box::new(
                SubsampleBridge::<U8>::with_point(12, 5, 1, true).unwrap(),
            ))
            .unwrap()
            .build()
            .unwrap();
        let mut sink = MemorySink::for_pipeline(&pipeline).unwrap();

        RayonScheduler::new(1)
            .unwrap()
            .run(&pipeline, &mut sink)
            .unwrap();

        let reads = reads.lock().unwrap();
        assert_eq!(reads.len(), 30);
        assert!(
            reads
                .iter()
                .all(|region| region.width == 1 && region.height == 1)
        );
        assert_eq!(reads[0], Region::new(0, 0, 1, 1));
        assert_eq!(reads[1], Region::new(12, 0, 1, 1));
        assert_eq!(reads[5], Region::new(0, 5, 1, 1));
        assert!(!reads.contains(&Region::new(0, 0, 49, 26)));

        let expected = expected_subsample(
            &(0..32)
                .flat_map(|y| (0..64).map(move |x| 255 - (x + y) as u8))
                .collect::<Vec<_>>(),
            64,
            1,
            5,
            6,
            12,
            5,
        );
        assert_eq!(sink.into_buffer(), expected);
    }

    proptest! {
        #[test]
        fn factor_one_is_identity(
            width in 1usize..=8,
            height in 1usize..=8,
            bands in 1usize..=4,
        ) {
            let len = width * height * bands;
            let input = (0..len).map(|idx| (idx % 251) as u8).collect::<Vec<_>>();
            let output = run_subsample(1, 1, width as u32, width as u32, height as u32, bands as u32, &input);
            prop_assert_eq!(output, input);
        }

        #[test]
        fn matches_reference_nearest_neighbor(
            output_width in 1usize..=5,
            output_height in 1usize..=5,
            bands in 1usize..=3,
            xfac in 1usize..=4,
            yfac in 1usize..=4,
        ) {
            let input_width = output_width * xfac - (xfac - 1);
            let input_height = output_height * yfac - (yfac - 1);
            let len = input_width * input_height * bands;
            let input = (0..len).map(|idx| (idx % 251) as u8).collect::<Vec<_>>();

            let output = run_subsample(
                xfac as u32,
                yfac as u32,
                input_width as u32,
                output_width as u32,
                output_height as u32,
                bands as u32,
                &input,
            );
            let expected = expected_subsample(
                &input,
                input_width,
                bands,
                output_width,
                output_height,
                xfac,
                yfac,
            );

            prop_assert_eq!(output, expected);
        }
    }
}
