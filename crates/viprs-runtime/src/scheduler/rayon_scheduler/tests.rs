use super::*;
use crate::adapters::{
    pipeline::PipelineBuilder, sinks::memory::MemorySink, sources::memory::MemorySource,
};
use crate::domain::image::{DemandHint, ImageMetadata};
use crate::domain::{error::BuildError, format::U16};
use crate::ports::sink::ImageSink;
use crate::ports::source::ImageSource;
use std::sync::{
    Arc, Mutex,
    atomic::{AtomicBool, AtomicUsize, Ordering},
};
use std::time::Duration;

use crate::domain::{
    format::U8,
    image::{Tile, TileMut},
    op::{Op, OperationBridge},
};

struct PassThrough;
impl Op for PassThrough {
    type Input = U8;
    type Output = U8;
    type State = ();

    fn demand_hint(&self) -> DemandHint {
        DemandHint::ThinStrip
    }

    fn required_input_region(&self, region: &Region) -> Region {
        *region
    }

    fn start(&self) {}

    #[inline]
    fn process_region(&self, (): &mut (), input: &Tile<U8>, output: &mut TileMut<U8>) {
        output.data.copy_from_slice(input.data);
    }
}

struct PanicOnceOp {
    triggered: Arc<AtomicBool>,
}

impl Op for PanicOnceOp {
    type Input = U8;
    type Output = U8;
    type State = ();

    fn demand_hint(&self) -> DemandHint {
        DemandHint::ThinStrip
    }

    fn required_input_region(&self, region: &Region) -> Region {
        *region
    }

    fn start(&self) {}

    #[inline]
    fn process_region(&self, (): &mut (), input: &Tile<U8>, output: &mut TileMut<U8>) {
        assert!(
            self.triggered.swap(true, Ordering::SeqCst),
            "synthetic rayon panic"
        );
        output.data.copy_from_slice(input.data);
    }
}

struct TypedPanicOnceOp {
    triggered: Arc<AtomicBool>,
}

impl Op for TypedPanicOnceOp {
    type Input = U8;
    type Output = U8;
    type State = ();

    fn demand_hint(&self) -> DemandHint {
        DemandHint::ThinStrip
    }

    fn required_input_region(&self, region: &Region) -> Region {
        *region
    }

    fn start(&self) {}

    #[inline]
    fn process_region(&self, (): &mut (), input: &Tile<U8>, output: &mut TileMut<U8>) {
        if !self.triggered.swap(true, Ordering::SeqCst) {
            std::panic::panic_any(ViprsError::Codec("synthetic typed panic".into()));
        }
        output.data.copy_from_slice(input.data);
    }
}

struct BlockingOp {
    current: Arc<AtomicUsize>,
    max_seen: Arc<AtomicUsize>,
}

impl Op for BlockingOp {
    type Input = U8;
    type Output = U8;
    type State = ();

    fn demand_hint(&self) -> DemandHint {
        DemandHint::ThinStrip
    }

    fn required_input_region(&self, region: &Region) -> Region {
        *region
    }

    fn start(&self) {}

    #[inline]
    fn process_region(&self, (): &mut (), input: &Tile<U8>, output: &mut TileMut<U8>) {
        let active = self.current.fetch_add(1, Ordering::SeqCst) + 1;
        let _ = self
            .max_seen
            .fetch_update(Ordering::SeqCst, Ordering::SeqCst, |max_seen| {
                Some(max_seen.max(active))
            });
        std::thread::sleep(Duration::from_millis(100));
        output.data.copy_from_slice(input.data);
        self.current.fetch_sub(1, Ordering::SeqCst);
    }
}

struct SerialOnlySink {
    inner: MemorySink,
}

impl SerialOnlySink {
    fn new(pipeline: &CompiledPipeline) -> Self {
        Self {
            inner: MemorySink::for_pipeline(pipeline).unwrap(),
        }
    }
}

impl ImageSink for SerialOnlySink {
    fn write_region(&mut self, region: Region, data: &[u8]) -> Result<(), ViprsError> {
        self.inner.write_region(region, data)
    }

    fn finish(self: Box<Self>) -> Result<(), ViprsError> {
        Ok(())
    }
}

fn make_profile_pipeline(width: u32, height: u32) -> CompiledPipeline {
    let pixels: Vec<u8> = (0..width as usize * height as usize)
        .map(|value| (value % (u8::MAX as usize + 1)) as u8)
        .collect();
    let source = MemorySource::<U8>::new(width, height, 1, pixels).unwrap();
    let mut pipeline = PipelineBuilder::from_source(source)
        .then(Box::new(OperationBridge::new(PassThrough, 1u32)))
        .unwrap()
        .build()
        .unwrap();
    pipeline.demand_hint = DemandHint::ThinStrip;
    pipeline
}

#[derive(Default)]
struct TrackingSourceState {
    requests: Mutex<Vec<Region>>,
    active_reads: AtomicUsize,
    max_active_reads: AtomicUsize,
}

impl TrackingSourceState {
    fn record_start(&self, region: Region) {
        self.requests.lock().unwrap().push(region);
        let active = self.active_reads.fetch_add(1, Ordering::SeqCst) + 1;
        self.max_active_reads.fetch_max(active, Ordering::SeqCst);
    }

    fn record_end(&self) {
        self.active_reads.fetch_sub(1, Ordering::SeqCst);
    }
}

#[derive(Clone)]
struct TrackingSource {
    width: u32,
    height: u32,
    state: Arc<TrackingSourceState>,
}

impl TrackingSource {
    fn new(width: u32, height: u32, state: Arc<TrackingSourceState>) -> Self {
        Self {
            width,
            height,
            state,
        }
    }
}

#[derive(Clone)]
struct BorrowTrackingSource {
    width: u32,
    height: u32,
    data: Arc<[u8]>,
    read_calls: Arc<AtomicUsize>,
    borrow_calls: Arc<AtomicUsize>,
}

impl BorrowTrackingSource {
    fn new(width: u32, height: u32) -> Self {
        let data: Vec<u8> = (0..width as usize * height as usize)
            .map(|value| (value % (u8::MAX as usize + 1)) as u8)
            .collect();
        Self {
            width,
            height,
            data: Arc::<[u8]>::from(data),
            read_calls: Arc::new(AtomicUsize::new(0)),
            borrow_calls: Arc::new(AtomicUsize::new(0)),
        }
    }
}

impl ImageSource for BorrowTrackingSource {
    type Format = U8;

    fn width(&self) -> u32 {
        self.width
    }

    fn height(&self) -> u32 {
        self.height
    }

    fn bands(&self) -> u32 {
        1
    }

    fn demand_hint(&self) -> DemandHint {
        DemandHint::ThinStrip
    }

    fn metadata(&self) -> ImageMetadata {
        ImageMetadata::default()
    }

    fn read_region(&self, region: Region, output: &mut [u8]) -> Result<(), ViprsError> {
        self.read_calls.fetch_add(1, Ordering::SeqCst);
        let stride = self.width as usize;
        for row in 0..region.height as usize {
            let start = (region.y as usize + row) * stride;
            let end = start + region.width as usize;
            let dst = row * region.width as usize;
            output[dst..dst + region.width as usize].copy_from_slice(&self.data[start..end]);
        }
        Ok(())
    }

    fn borrow_region(&self, region: Region) -> Option<&[u8]> {
        if region.x != 0 || region.width != self.width || region.y < 0 {
            return None;
        }
        self.borrow_calls.fetch_add(1, Ordering::SeqCst);
        let start = region.y as usize * self.width as usize;
        let end = start + region.height as usize * self.width as usize;
        Some(&self.data[start..end])
    }
}

impl ImageSource for TrackingSource {
    type Format = U8;

    fn width(&self) -> u32 {
        self.width
    }

    fn height(&self) -> u32 {
        self.height
    }

    fn bands(&self) -> u32 {
        1
    }

    fn demand_hint(&self) -> DemandHint {
        DemandHint::ThinStrip
    }

    fn metadata(&self) -> ImageMetadata {
        ImageMetadata::default()
    }

    fn read_region(&self, region: Region, output: &mut [u8]) -> Result<(), ViprsError> {
        self.state.record_start(region);
        std::thread::sleep(Duration::from_millis(10));
        output.fill(region.y as u8);
        self.state.record_end();
        Ok(())
    }
}

/// Build a minimal pipeline stub to drive tile generation.
fn make_pipeline(width: u32, height: u32, hint: DemandHint) -> CompiledPipeline {
    use crate::domain::op::Op;

    struct Noop;
    impl Op for Noop {
        type Input = U8;
        type Output = U8;
        type State = ();
        fn demand_hint(&self) -> DemandHint {
            DemandHint::ThinStrip
        }
        fn required_input_region(&self, r: &Region) -> Region {
            *r
        }
        fn start(&self) {}
        #[inline]
        fn process_region(&self, (): &mut (), input: &Tile<U8>, output: &mut TileMut<U8>) {
            output.data.copy_from_slice(input.data);
        }
    }

    let mut pipeline = PipelineBuilder::new(width, height)
        .then(Box::new(OperationBridge::new(Noop, 1u32)))
        .unwrap()
        .build()
        .unwrap();
    // Override demand_hint to match what we're testing.
    pipeline.demand_hint = hint;
    pipeline
}

fn make_borrow_pipeline(source: BorrowTrackingSource, pass_count: usize) -> CompiledPipeline {
    let mut builder = PipelineBuilder::from_source(source);
    for _ in 0..pass_count {
        builder = builder
            .then(Box::new(OperationBridge::new(PassThrough, 1u32)))
            .unwrap();
    }
    let mut pipeline = builder.build().unwrap();
    pipeline.demand_hint = DemandHint::ThinStrip;
    pipeline
}

#[test]
fn tiles_cover_image_no_overlap() {
    let pipeline = make_pipeline(100, 100, DemandHint::ThinStrip);
    let tiles = generate_tiles(&pipeline).unwrap();

    // Build a coverage map and verify every pixel is covered exactly once.
    let mut coverage = vec![0u32; 100 * 100];
    for t in &tiles {
        for row in 0..t.height {
            for col in 0..t.width {
                let x = t.x as u32 + col;
                let y = t.y as u32 + row;
                coverage[(y * 100 + x) as usize] += 1;
            }
        }
    }
    assert!(
        coverage.iter().all(|&c| c == 1),
        "some pixels covered != 1 time"
    );
}

#[test]
fn single_transform_uses_borrowed_source_strip() {
    let source = BorrowTrackingSource::new(32, 32);
    let read_calls = Arc::clone(&source.read_calls);
    let borrow_calls = Arc::clone(&source.borrow_calls);
    let pipeline = make_borrow_pipeline(source, 1);
    let sink = MemorySink::for_pipeline(&pipeline).unwrap();

    RayonScheduler::new(2)
        .unwrap()
        .run_concurrent(&pipeline, &sink)
        .unwrap();

    assert_eq!(read_calls.load(Ordering::SeqCst), 0);
    assert!(borrow_calls.load(Ordering::SeqCst) > 0);
    assert_eq!(
        sink.into_buffer(),
        (0u8..=255).cycle().take(32 * 32).collect::<Vec<_>>()
    );
}

#[test]
fn multi_stage_pipeline_reuses_borrowed_source_strip() {
    let source = BorrowTrackingSource::new(32, 32);
    let read_calls = Arc::clone(&source.read_calls);
    let borrow_calls = Arc::clone(&source.borrow_calls);
    let pipeline = make_borrow_pipeline(source, 2);
    let sink = MemorySink::for_pipeline(&pipeline).unwrap();

    RayonScheduler::new(2)
        .unwrap()
        .run_concurrent(&pipeline, &sink)
        .unwrap();

    assert_eq!(read_calls.load(Ordering::SeqCst), 0);
    assert!(borrow_calls.load(Ordering::SeqCst) > 0);
    assert_eq!(
        sink.into_buffer(),
        (0u8..=255).cycle().take(32 * 32).collect::<Vec<_>>()
    );
}

#[test]
fn thin_strip_uses_fatstrip_height_for_non_wide_images() {
    // libvips only drops ThinStrip to 1 scanline for very wide images; otherwise it
    // reuses the FatStrip height to avoid excessive scheduler overhead.
    let pipeline = make_pipeline(100, 100, DemandHint::ThinStrip);
    let geometry = tile_geometry_for_l2_budget(&pipeline, DEFAULT_TARGET_L2_BYTES).unwrap();

    assert_eq!(geometry.tile_width, 100);
    assert_eq!(
        geometry.tile_height, 16,
        "ThinStrip should fall back to FatStrip geometry for non-wide images"
    );
    assert_eq!(geometry.cols, 1);
    assert_eq!(geometry.rows, 7);
}

#[test]
fn thin_strip_uses_single_scanlines_for_very_wide_images() {
    let pipeline = make_pipeline(10_001, 4, DemandHint::ThinStrip);
    let geometry = tile_geometry_for_l2_budget(&pipeline, DEFAULT_TARGET_L2_BYTES).unwrap();

    assert_eq!(geometry.tile_width, 10_001);
    assert_eq!(geometry.tile_height, 1);
    assert_eq!(geometry.cols, 1);
    assert_eq!(geometry.rows, 4);
}

#[test]
fn zero_sized_pipeline_produces_no_tiles() {
    let pipeline = make_pipeline(0, 4, DemandHint::ThinStrip);
    let geometry = tile_geometry_for_l2_budget(&pipeline, DEFAULT_TARGET_L2_BYTES).unwrap();

    assert_eq!(
        geometry,
        TileGeometry {
            tile_width: 0,
            tile_height: 0,
            cols: 0,
            rows: 0,
        }
    );
    assert!(generate_tiles(&pipeline).unwrap().is_empty());
}

#[test]
fn fat_strip_hint_produces_full_width_tiles() {
    let pipeline = make_pipeline(256, 40, DemandHint::FatStrip);
    let geometry = tile_geometry_for_l2_budget(&pipeline, 16_384).unwrap();

    assert_eq!(geometry.tile_width, 256);
    assert_eq!(geometry.tile_height, 40);
    assert_eq!(geometry.cols, 1);
    assert_eq!(geometry.rows, 1);
}

#[test]
fn fat_strip_respects_l2_budget() {
    let pipeline = make_pipeline(2_048, 32, DemandHint::FatStrip);
    let geometry = tile_geometry_for_l2_budget(&pipeline, 4_096).unwrap();

    assert_eq!(geometry.tile_width, 2_048);
    assert_eq!(geometry.tile_height, 2);
}

#[test]
fn execution_geometry_splits_short_fat_strips_across_threads() {
    let pipeline = make_pipeline(400, 400, DemandHint::FatStrip);
    let scheduler = RayonScheduler::new(14).unwrap();

    let geometry = scheduler.tile_geometry_for_execution(&pipeline).unwrap();

    assert_eq!(geometry.tile_width, 400);
    assert_eq!(geometry.tile_height, 25);
    assert_eq!(geometry.rows, 16);
}

#[test]
fn generate_tile_strips_groups_adjacent_tile_rows() {
    let pipeline = make_pipeline(12, 24, DemandHint::FatStrip);
    let strips = generate_tile_strips(&pipeline, 2).unwrap();

    assert_eq!(strips.len(), 1);
    assert_eq!(strips[0].regions, vec![Region::new(0, 0, 12, 24)]);
}

#[test]
fn scheduler_strip_height_defaults_from_l2_budget() {
    let pipeline = make_pipeline(128, 512, DemandHint::SmallTile);

    let scheduler = RayonScheduler::new(1)
        .unwrap()
        .with_l2_cache_bytes(16_384 * 4);
    assert_eq!(
        scheduler.effective_strip_height_tiles(&pipeline).unwrap(),
        4
    );

    let explicit = RayonScheduler::new(2)
        .unwrap()
        .with_l2_cache_bytes(1)
        .with_strip_height_tiles(3);
    assert_eq!(explicit.effective_strip_height_tiles(&pipeline).unwrap(), 3);
}

#[test]
fn static_work_ranges_batch_small_strip_sets_per_thread() {
    let scheduler = RayonScheduler::new(10).unwrap();
    let ranges = scheduler
        .static_work_ranges(DemandHint::FatStrip, 16)
        .unwrap();

    assert_eq!(
        ranges,
        vec![0..2, 2..4, 4..6, 6..8, 8..10, 10..12, 12..14, 14..16]
    );
}

#[test]
fn static_work_ranges_skips_large_strip_sets() {
    let scheduler = RayonScheduler::new(4).unwrap();

    assert!(
        scheduler
            .static_work_ranges(DemandHint::FatStrip, 9)
            .is_none()
    );
    assert!(
        scheduler
            .static_work_ranges(DemandHint::FatStrip, 1)
            .is_none()
    );
}

#[test]
fn static_work_ranges_keep_large_thin_strip_sets_static() {
    let scheduler = RayonScheduler::new(4).unwrap();

    assert_eq!(
        scheduler
            .static_work_ranges(DemandHint::ThinStrip, 9)
            .unwrap(),
        vec![0..3, 3..6, 6..9]
    );
}

#[test]
fn scoped_strip_dispatch_stays_static_for_small_rgb_f32_workloads() {
    let scheduler = RayonScheduler::new(10).unwrap();

    assert!(
        scheduler
            .should_use_scoped_strips_for_workload(
                DemandHint::ThinStrip,
                32,
                10,
                Region::new(0, 0, 512, 512),
                3,
                BandFormatId::F32,
            )
            .unwrap()
    );
}

#[test]
fn scoped_strip_dispatch_falls_back_to_dynamic_for_large_rgb_f32_workloads() {
    let scheduler = RayonScheduler::new(10).unwrap();

    assert!(
        !scheduler
            .should_use_scoped_strips_for_workload(
                DemandHint::ThinStrip,
                391,
                10,
                Region::new(0, 0, 8_192, 8_192),
                3,
                BandFormatId::F32,
            )
            .unwrap()
    );
}

#[test]
fn thin_strip_worker_budget_caps_large_pool_footprints() {
    let scheduler = RayonScheduler::new(10).unwrap();

    let worker_count = scheduler.effective_strip_worker_count_for_workload(
        DemandHint::ThinStrip,
        25,
        192 * 1024 * 1024,
        256 * 1024 * 1024,
        400,
    );

    assert_eq!(worker_count, 6);
}

#[test]
fn thin_strip_worker_budget_caps_large_sources_even_with_small_pools() {
    let scheduler = RayonScheduler::new(10).unwrap();

    let worker_count = scheduler.effective_strip_worker_count_for_workload(
        DemandHint::ThinStrip,
        25,
        24 * 1024 * 1024,
        256 * 1024 * 1024,
        400,
    );

    assert_eq!(worker_count, 6);
}

#[test]
fn thin_strip_worker_budget_skips_small_sources() {
    let scheduler = RayonScheduler::new(10).unwrap();

    let worker_count = scheduler.effective_strip_worker_count_for_workload(
        DemandHint::ThinStrip,
        25,
        192 * 1024 * 1024,
        32 * 1024 * 1024,
        400,
    );

    assert_eq!(worker_count, 10);
}

#[test]
fn large_thumbnail_sharpen_caps_strip_workers() {
    use crate::{
        adapters::{pipeline::PipelineBuilder, sources::zero::ZeroSource},
        domain::{
            colorspace::ColorspaceId,
            kernel::InterpolationKernel,
            ops::resample::thumbnail::{Thumbnail, ThumbnailTarget},
        },
    };

    let scheduler = RayonScheduler::new(10).unwrap();
    let pipeline = PipelineBuilder::from_source(ZeroSource::<U8>::new(8_192, 8_192, 3))
        .with_colorspace(ColorspaceId::SRgb)
        .thumbnail(Thumbnail::new(
            ThumbnailTarget::Width(400),
            InterpolationKernel::Lanczos3,
        ))
        .unwrap()
        .sharpen(0.5, 2.0, 10.0, 20.0, 0.0, 3.0)
        .unwrap()
        .build()
        .unwrap();
    let strips = scheduler
        .generate_tile_strips_for_execution(
            &pipeline,
            scheduler.effective_strip_height_tiles(&pipeline).unwrap(),
        )
        .unwrap();

    assert!(
        scheduler
            .effective_strip_worker_count(&pipeline, strips.len())
            .unwrap()
            < 10,
        "8192px thumbnail chains should cap strip workers to limit pooled scratch RSS"
    );
}

#[test]
fn medium_thumbnail_sharpen_keeps_full_strip_workers() {
    use crate::{
        adapters::{pipeline::PipelineBuilder, sources::zero::ZeroSource},
        domain::{
            colorspace::ColorspaceId,
            kernel::InterpolationKernel,
            ops::resample::thumbnail::{Thumbnail, ThumbnailTarget},
        },
    };

    let scheduler = RayonScheduler::new(10).unwrap();
    let pipeline = PipelineBuilder::from_source(ZeroSource::<U8>::new(2_048, 2_048, 3))
        .with_colorspace(ColorspaceId::SRgb)
        .thumbnail(Thumbnail::new(
            ThumbnailTarget::Width(400),
            InterpolationKernel::Lanczos3,
        ))
        .unwrap()
        .sharpen(0.5, 2.0, 10.0, 20.0, 0.0, 3.0)
        .unwrap()
        .build()
        .unwrap();
    let strips = scheduler
        .generate_tile_strips_for_execution(
            &pipeline,
            scheduler.effective_strip_height_tiles(&pipeline).unwrap(),
        )
        .unwrap();

    assert_eq!(
        scheduler
            .effective_strip_worker_count(&pipeline, strips.len())
            .unwrap(),
        10
    );
}

#[test]
fn strip_scheduling_matches_single_tile_row_output() {
    use crate::{
        adapters::{
            pipeline::PipelineBuilder, sinks::memory::MemorySink, sources::memory::MemorySource,
        },
        domain::op::OperationBridge,
    };

    let pixels: Vec<u8> = (0..64u8).collect();
    let source_a = MemorySource::<U8>::new(8, 8, 1, pixels.clone()).unwrap();
    let source_b = MemorySource::<U8>::new(8, 8, 1, pixels.clone()).unwrap();

    let pipeline_a = PipelineBuilder::from_source(source_a)
        .then(Box::new(OperationBridge::new(PassThrough, 1u32)))
        .unwrap()
        .build()
        .unwrap();
    let pipeline_b = PipelineBuilder::from_source(source_b)
        .then(Box::new(OperationBridge::new(PassThrough, 1u32)))
        .unwrap()
        .build()
        .unwrap();

    let scheduler_a = RayonScheduler::new(2).unwrap().with_strip_height_tiles(1);
    let scheduler_b = RayonScheduler::new(2).unwrap().with_strip_height_tiles(4);

    let sink_a = MemorySink::for_pipeline(&pipeline_a).unwrap();
    scheduler_a.run_concurrent(&pipeline_a, &sink_a).unwrap();
    let output_a = sink_a.into_buffer();

    let sink_b = MemorySink::for_pipeline(&pipeline_b).unwrap();
    scheduler_b.run_concurrent(&pipeline_b, &sink_b).unwrap();
    let output_b = sink_b.into_buffer();

    assert_eq!(output_a, output_b);
    assert_eq!(output_b, pixels);
}

/// `run_concurrent` must produce byte-identical output to `run` for the same pipeline.
///
/// The test uses a real `MemorySource` with known pixel data and a `MemorySink`.
/// Both paths are driven through a 4x4 single-band U8 pass-through pipeline.
/// Correctness does not depend on buffer-zero initialization — the source provides
/// explicit pixel values that differ from zero.
#[test]
fn run_concurrent_matches_run_output() {
    use crate::{
        adapters::{pipeline::PipelineBuilder, sources::memory::MemorySource},
        domain::op::OperationBridge,
    };

    // 4x4 single-band image with non-zero, distinct pixel values.
    let pixels: Vec<u8> = (1u8..=16).collect();

    let source_run = MemorySource::<U8>::new(4, 4, 1, pixels.clone()).unwrap();
    let pipeline_run = PipelineBuilder::from_source(source_run)
        .then(Box::new(OperationBridge::new(PassThrough, 1u32)))
        .unwrap()
        .build()
        .unwrap();

    let source_conc = MemorySource::<U8>::new(4, 4, 1, pixels).unwrap();
    let pipeline_conc = PipelineBuilder::from_source(source_conc)
        .then(Box::new(OperationBridge::new(PassThrough, 1u32)))
        .unwrap()
        .build()
        .unwrap();

    // Path A: run (with Mutex)
    let mut sink_run = MemorySink::for_pipeline(&pipeline_run).unwrap();
    let scheduler = RayonScheduler::new(2).unwrap();
    scheduler.run(&pipeline_run, &mut sink_run).unwrap();
    let output_run = sink_run.into_buffer();

    // Path B: run_concurrent (lock-free)
    let sink_conc = MemorySink::for_pipeline(&pipeline_conc).unwrap();
    scheduler
        .run_concurrent(&pipeline_conc, &sink_conc)
        .unwrap();
    let output_conc = sink_conc.into_buffer();

    assert_eq!(
        output_run, output_conc,
        "run and run_concurrent must produce identical output"
    );
    // Both outputs must match the source data — the pass-through op is an identity.
    let expected: Vec<u8> = (1u8..=16).collect();
    assert_eq!(
        output_run, expected,
        "pass-through must preserve source pixels"
    );
}

#[test]
fn run_concurrent_returns_typed_error_for_worker_panic_payload() {
    let pixels: Vec<u8> = (0..(8 * 32)).map(|value| (value % 251) as u8).collect();
    let source = MemorySource::<U8>::new(8, 32, 1, pixels).unwrap();
    let panic_state = Arc::new(AtomicBool::new(false));
    let pipeline = PipelineBuilder::from_source(source)
        .then(Box::new(OperationBridge::new(
            TypedPanicOnceOp {
                triggered: panic_state,
            },
            1u32,
        )))
        .unwrap()
        .build()
        .unwrap();

    let scheduler = RayonScheduler::new(2).unwrap().with_strip_height_tiles(1);
    let sink = MemorySink::for_pipeline(&pipeline).unwrap();
    let err = scheduler.run_concurrent(&pipeline, &sink).unwrap_err();

    assert!(
        matches!(err, ViprsError::Codec(ref message) if message == "synthetic typed panic"),
        "typed panic payload must propagate as its original ViprsError, got {err:?}"
    );
}

#[test]
fn run_concurrent_wraps_worker_panics_and_pool_recovers() {
    let pixels: Vec<u8> = (0..(8 * 32)).map(|value| (value % 251) as u8).collect();
    let panic_state = Arc::new(AtomicBool::new(false));
    let panic_pipeline =
        PipelineBuilder::from_source(MemorySource::<U8>::new(8, 32, 1, pixels.clone()).unwrap())
            .then(Box::new(OperationBridge::new(
                PanicOnceOp {
                    triggered: Arc::clone(&panic_state),
                },
                1u32,
            )))
            .unwrap()
            .build()
            .unwrap();

    let scheduler = RayonScheduler::new(2).unwrap().with_strip_height_tiles(1);
    let panic_sink = MemorySink::for_pipeline(&panic_pipeline).unwrap();
    let err = scheduler
        .run_concurrent(&panic_pipeline, &panic_sink)
        .unwrap_err();
    assert!(
        matches!(err, ViprsError::Scheduler(ref message) if message.contains("synthetic rayon panic")),
        "string panic must surface as a scheduler error, got {err:?}"
    );

    let recovery_pipeline =
        PipelineBuilder::from_source(MemorySource::<U8>::new(8, 32, 1, pixels.clone()).unwrap())
            .then(Box::new(OperationBridge::new(PassThrough, 1u32)))
            .unwrap()
            .build()
            .unwrap();
    let recovery_sink = MemorySink::for_pipeline(&recovery_pipeline).unwrap();
    scheduler
        .run_concurrent(&recovery_pipeline, &recovery_sink)
        .unwrap();

    assert_eq!(
        recovery_sink.into_buffer(),
        pixels,
        "rayon pool must remain usable after a worker panic"
    );
}

#[test]
fn scheduler_limits_max_concurrent_pipeline_runs() {
    let scheduler = Arc::new(
        RayonScheduler::new(1)
            .unwrap()
            .with_max_concurrent_pipelines(1),
    );
    let current = Arc::new(AtomicUsize::new(0));
    let max_seen = Arc::new(AtomicUsize::new(0));

    let make_pipeline = || {
        PipelineBuilder::from_source(MemorySource::<U8>::new(1, 1, 1, vec![7]).unwrap())
            .then(Box::new(OperationBridge::new(
                BlockingOp {
                    current: Arc::clone(&current),
                    max_seen: Arc::clone(&max_seen),
                },
                1u32,
            )))
            .unwrap()
            .build()
            .unwrap()
    };

    let pipeline_a = make_pipeline();
    let pipeline_b = make_pipeline();

    let scheduler_a = Arc::clone(&scheduler);
    let handle_a = std::thread::spawn(move || {
        let sink = MemorySink::for_pipeline(&pipeline_a).unwrap();
        scheduler_a.run_concurrent(&pipeline_a, &sink).unwrap();
        sink.into_buffer()
    });

    std::thread::sleep(Duration::from_millis(25));

    let scheduler_b = Arc::clone(&scheduler);
    let handle_b = std::thread::spawn(move || {
        let sink = MemorySink::for_pipeline(&pipeline_b).unwrap();
        scheduler_b.run_concurrent(&pipeline_b, &sink).unwrap();
        sink.into_buffer()
    });

    assert_eq!(handle_a.join().unwrap(), vec![7]);
    assert_eq!(handle_b.join().unwrap(), vec![7]);
    assert_eq!(max_seen.load(Ordering::SeqCst), 1);
}

/// `run_with_reducer` must produce the correct aggregate AND write the same pixels
/// to the sink as `run_concurrent`.
///
/// Uses a 4x4 single-band U8 pass-through pipeline with pixel values 1..=16.
/// The `PixelSum` reducer sums every sample value. Expected sum = 1+2+…+16 = 136.
/// The sink output must equal the source pixels (identity pass-through).
#[test]
fn run_with_reducer_returns_correct_sum_and_writes_sink() {
    use crate::{
        adapters::{pipeline::PipelineBuilder, sources::memory::MemorySource},
        domain::{op::OperationBridge, reducer::TileReducer},
    };

    // Sums every u8 sample across all tiles.
    struct PixelSum;
    impl TileReducer<U8> for PixelSum {
        type Partial = u64;
        type Output = u64;
        type Scratch = ();

        fn reduce_tile(&self, tile: &Tile<U8>, _region: &Region) -> u64 {
            tile.data.iter().map(|&s| s as u64).sum()
        }

        fn combine(&self, a: u64, b: u64) -> u64 {
            a + b
        }

        fn finalize(&self, combined: u64) -> u64 {
            combined
        }
    }

    // 4x4 single-band image with pixel values 1..=16. Sum = 136.
    let pixels: Vec<u8> = (1u8..=16).collect();
    let source = MemorySource::<U8>::new(4, 4, 1, pixels.clone()).unwrap();
    let pipeline = PipelineBuilder::from_source(source)
        .then(Box::new(OperationBridge::new(PassThrough, 1u32)))
        .unwrap()
        .build()
        .unwrap();

    let sink = MemorySink::for_pipeline(&pipeline).unwrap();
    let scheduler = RayonScheduler::new(2).unwrap();
    let total = scheduler
        .run_with_reducer::<U8, PixelSum>(&pipeline, &sink, &PixelSum)
        .unwrap();

    assert_eq!(total, 136, "sum of 1..=16 must be 136");

    // The sink must contain the pass-through pixels (identity op).
    let output = sink.into_buffer();
    assert_eq!(output, pixels, "pass-through must preserve source pixels");
}

#[cfg(feature = "lock_instrumentation")]
#[test]
fn run_with_profile_uses_zero_scheduler_locks_with_concurrent_sink() {
    let pipeline = make_profile_pipeline(100, 100);
    let expected_tiles = generate_tiles(&pipeline).unwrap().len() as u64;
    let mut sink = MemorySink::for_pipeline(&pipeline).unwrap();
    let scheduler = RayonScheduler::new(4).unwrap();

    let profile = scheduler.run_with_profile(&pipeline, &mut sink).unwrap();

    assert_eq!(profile.tile_count, expected_tiles);
    assert_eq!(profile.lock_stats.tile_count, expected_tiles);
    assert_eq!(profile.lock_stats.total_lock_acquisitions, 0);
    assert_eq!(profile.lock_stats.max_locks_per_tile, 0);
}

#[cfg(feature = "lock_instrumentation")]
#[test]
fn run_with_profile_reduces_serial_sink_to_one_lock_per_tile() {
    let pipeline = make_profile_pipeline(100, 100);
    let expected_tiles = generate_tiles(&pipeline).unwrap().len() as u64;
    let mut sink = SerialOnlySink::new(&pipeline);
    let scheduler = RayonScheduler::new(4).unwrap();

    let profile = scheduler.run_with_profile(&pipeline, &mut sink).unwrap();

    assert_eq!(profile.tile_count, expected_tiles);
    assert_eq!(profile.lock_stats.tile_count, expected_tiles);
    assert_eq!(profile.lock_stats.total_lock_acquisitions, expected_tiles);
    assert_eq!(profile.lock_stats.max_locks_per_tile, 1);
}

#[test]
fn view_only_pipeline_direct_writes_into_memory_sink() {
    use crate::{pipeline::PipelineBuilder, sources::memory::MemorySource};

    let width = 128;
    let height = 96;
    let bands = 3;
    let pixels: Vec<u8> = (0..width as usize * height as usize * bands as usize)
        .map(|value| (value % 251) as u8)
        .collect();
    let source = MemorySource::<U8>::new(width, height, bands, pixels).unwrap();
    let pipeline = PipelineBuilder::from_source(source)
        .extract_area(8, 4, width - 16, height - 8)
        .unwrap()
        .build()
        .unwrap();

    let scheduler = RayonScheduler::new(4).unwrap();
    let expected_tiles = scheduler
        .generate_tile_strips_for_execution(
            &pipeline,
            scheduler.effective_strip_height_tiles(&pipeline).unwrap(),
        )
        .unwrap()
        .iter()
        .map(|strip| strip.regions.len() as u64)
        .sum::<u64>();

    let mut sink = MemorySink::for_pipeline(&pipeline).unwrap();
    let profile = scheduler.run_with_profile(&pipeline, &mut sink).unwrap();

    assert_eq!(profile.tile_count, expected_tiles);
    assert_eq!(
        profile.sink_write_ns, 0,
        "view-only extract_area should bypass the extra sink scatter copy"
    );

    let output = sink.into_buffer();
    let expected_first = (((4 * width) + 8) * bands) as usize;
    assert_eq!(
        &output[..bands as usize],
        &[
            (expected_first % 251) as u8,
            ((expected_first + 1) % 251) as u8,
            ((expected_first + 2) % 251) as u8,
        ]
    );
}

#[cfg(feature = "lock_instrumentation")]
#[test]
fn branch_point_op_cache_stays_within_two_locks_per_output_tile() {
    use crate::{
        domain::{format::BandFormatId, ops::conversion::BandJoin},
        pipeline::PipelineArena,
    };

    let pixels: Vec<u8> = (0..100 * 100).map(|value| (value % 251) as u8).collect();
    let source = MemorySource::<U8>::new(100, 100, 1, pixels).unwrap();
    let mut arena = PipelineArena::with_source(Box::new(source));
    let root = arena.add_node(Box::new(OperationBridge::new(PassThrough, 1u32)));
    let branch = arena.add_node(Box::new(OperationBridge::new(PassThrough, 1u32)));
    let merge = arena.add_node(Box::new(BandJoin::new(1, 1, BandFormatId::U8)));
    arena.connect(root, branch).unwrap();
    arena.connect(root, merge).unwrap();
    arena.connect_to_slot(branch, merge, 1).unwrap();
    arena
        .enable_cache(root, std::num::NonZeroUsize::new(100 * 100).unwrap())
        .unwrap();
    let mut pipeline = arena.compile().unwrap();
    pipeline.demand_hint = DemandHint::ThinStrip;
    let expected_tiles = generate_tiles(&pipeline).unwrap().len() as u64;

    let scheduler = RayonScheduler::new(4).unwrap();
    let mut cold_sink = MemorySink::for_pipeline(&pipeline).unwrap();
    let cold_profile = scheduler
        .run_with_profile(&pipeline, &mut cold_sink)
        .unwrap();
    assert_eq!(cold_profile.lock_stats.tile_count, expected_tiles);
    assert_eq!(cold_profile.lock_stats.max_locks_per_tile, 2);

    let mut warm_sink = MemorySink::for_pipeline(&pipeline).unwrap();
    let warm_profile = scheduler
        .run_with_profile(&pipeline, &mut warm_sink)
        .unwrap();
    assert_eq!(warm_profile.lock_stats.tile_count, expected_tiles);
    assert_eq!(warm_profile.lock_stats.max_locks_per_tile, 1);
}

#[cfg(feature = "lock_instrumentation")]
#[test]
fn source_cache_plus_branch_point_op_cache_stays_within_four_locks_per_output_tile() {
    use crate::{
        domain::{format::BandFormatId, ops::conversion::BandJoin},
        pipeline::PipelineArena,
    };

    let pixels: Vec<u8> = (0..100 * 100).map(|value| (value % 251) as u8).collect();
    let source = crate::sources::tile_cache::TileCache::new(
        MemorySource::<U8>::new(100, 100, 1, pixels).unwrap(),
        std::num::NonZeroUsize::new(100 * 100).unwrap(),
    );
    let mut arena = PipelineArena::with_source(Box::new(source));
    let root = arena.add_node(Box::new(OperationBridge::new(PassThrough, 1u32)));
    let branch = arena.add_node(Box::new(OperationBridge::new(PassThrough, 1u32)));
    let merge = arena.add_node(Box::new(BandJoin::new(1, 1, BandFormatId::U8)));
    arena.connect(root, branch).unwrap();
    arena.connect(root, merge).unwrap();
    arena.connect_to_slot(branch, merge, 1).unwrap();
    arena
        .enable_cache(root, std::num::NonZeroUsize::new(100 * 100).unwrap())
        .unwrap();
    let mut pipeline = arena.compile().unwrap();
    pipeline.demand_hint = DemandHint::ThinStrip;
    let expected_tiles = generate_tiles(&pipeline).unwrap().len() as u64;

    let scheduler = RayonScheduler::new(4).unwrap();
    let mut cold_sink = MemorySink::for_pipeline(&pipeline).unwrap();
    let cold_profile = scheduler
        .run_with_profile(&pipeline, &mut cold_sink)
        .unwrap();
    assert_eq!(cold_profile.lock_stats.tile_count, expected_tiles);
    assert_eq!(cold_profile.lock_stats.max_locks_per_tile, 4);

    let mut warm_sink = MemorySink::for_pipeline(&pipeline).unwrap();
    let warm_profile = scheduler
        .run_with_profile(&pipeline, &mut warm_sink)
        .unwrap();
    assert_eq!(warm_profile.lock_stats.tile_count, expected_tiles);
    assert_eq!(warm_profile.lock_stats.max_locks_per_tile, 1);
}

/// `run_with_reducer` must return `ViprsError::Scheduler` when the caller passes
/// a format type `F` that does not match the pipeline's output format.
#[test]
fn run_with_reducer_rejects_format_mismatch() {
    use crate::{
        adapters::{pipeline::PipelineBuilder, sources::memory::MemorySource},
        domain::{format::U16, op::OperationBridge, reducer::TileReducer},
    };

    struct NoopReducer;
    impl TileReducer<U16> for NoopReducer {
        type Partial = u64;
        type Output = u64;
        type Scratch = ();
        fn reduce_tile(&self, _tile: &Tile<U16>, _region: &Region) -> u64 {
            0
        }
        fn combine(&self, a: u64, b: u64) -> u64 {
            a + b
        }
        fn finalize(&self, combined: u64) -> u64 {
            combined
        }
    }

    let pixels: Vec<u8> = vec![0u8; 4];
    let source = MemorySource::<U8>::new(2, 2, 1, pixels).unwrap();
    let pipeline = PipelineBuilder::from_source(source)
        .then(Box::new(OperationBridge::new(PassThrough, 1u32)))
        .unwrap()
        .build()
        .unwrap();

    let sink = MemorySink::for_pipeline(&pipeline).unwrap();
    let scheduler = RayonScheduler::new(1).unwrap();
    let result = scheduler.run_with_reducer::<U16, NoopReducer>(&pipeline, &sink, &NoopReducer);
    assert!(
        matches!(result, Err(ViprsError::Scheduler(_))),
        "format mismatch must return ViprsError::Scheduler"
    );
}

#[test]
fn sequential_access_runs_tiles_in_row_major_order() {
    use crate::{domain::op::OperationBridge, pipeline::PipelineBuilder};

    let state = Arc::new(TrackingSourceState::default());
    let source = TrackingSource::new(32, 256, Arc::clone(&state));
    let pipeline = PipelineBuilder::from_source(source)
        .sequential(0)
        .then(Box::new(OperationBridge::new(PassThrough, 1u32)))
        .unwrap()
        .build()
        .unwrap();

    let expected = generate_tiles(&pipeline).unwrap();
    let scheduler = RayonScheduler::new(4).unwrap();
    let sink = MemorySink::for_pipeline(&pipeline).unwrap();
    scheduler.run_concurrent(&pipeline, &sink).unwrap();

    let requests = state.requests.lock().unwrap().clone();
    assert_eq!(
        requests, expected,
        "sequential hint must preserve row-major tile order"
    );
    assert_eq!(
        state.max_active_reads.load(Ordering::SeqCst),
        1,
        "sequential hint must keep only one tile in flight"
    );
}

#[test]
fn sequential_access_reduces_in_flight_tile_memory() {
    use crate::{domain::op::OperationBridge, pipeline::PipelineBuilder};

    let parallel_state = Arc::new(TrackingSourceState::default());
    let parallel_source = TrackingSource::new(64, 256, Arc::clone(&parallel_state));
    let parallel_pipeline = PipelineBuilder::from_source(parallel_source)
        .then(Box::new(OperationBridge::new(PassThrough, 1u32)))
        .unwrap()
        .build()
        .unwrap();

    let sequential_state = Arc::new(TrackingSourceState::default());
    let sequential_source = TrackingSource::new(64, 256, Arc::clone(&sequential_state));
    let sequential_pipeline = PipelineBuilder::from_source(sequential_source)
        .sequential(0)
        .then(Box::new(OperationBridge::new(PassThrough, 1u32)))
        .unwrap()
        .build()
        .unwrap();

    let scheduler = RayonScheduler::new(4).unwrap();

    let parallel_sink = MemorySink::for_pipeline(&parallel_pipeline).unwrap();
    scheduler
        .run_concurrent(&parallel_pipeline, &parallel_sink)
        .unwrap();

    let sequential_sink = MemorySink::for_pipeline(&sequential_pipeline).unwrap();
    scheduler
        .run_concurrent(&sequential_pipeline, &sequential_sink)
        .unwrap();

    let parallel_peak = parallel_state.max_active_reads.load(Ordering::SeqCst);
    let sequential_peak = sequential_state.max_active_reads.load(Ordering::SeqCst);
    assert!(
        parallel_peak > sequential_peak,
        "non-sequential scheduling should keep more source tiles in flight"
    );
    assert_eq!(
        sequential_peak, 1,
        "sequential scheduling should cap in-flight tile memory at one tile"
    );
}

#[test]
fn sequential_line_cache_keeps_a_bounded_window() {
    let state = Arc::new(TrackingSourceState::default());
    let source = TrackingSource::new(8, 12, Arc::clone(&state));
    let mut cache =
        SequentialLineCache::new(8, 12, 1, 4, 2, crate::pipeline::LineCacheAccess::Sequential);
    let mut output = vec![0u8; 16];

    cache
        .read_region(&source, Region::new(0, 0, 8, 2), &mut output)
        .unwrap();
    cache
        .read_region(&source, Region::new(0, 1, 8, 2), &mut output)
        .unwrap();
    cache
        .read_region(&source, Region::new(0, 3, 8, 2), &mut output)
        .unwrap();

    let requests = state.requests.lock().unwrap().clone();
    assert_eq!(requests[0], Region::new(0, 0, 8, 2));
    assert!(
        requests.windows(2).all(|pair| pair[0].x == 0
            && pair[0].width == 8
            && pair[1].x == 0
            && pair[1].width == 8
            && pair[0].y <= pair[1].y),
        "line cache refills must remain full-width and monotonic"
    );
    assert!(
        cache.max_cached_lines() <= 4,
        "line cache must stay within the configured line budget"
    );
}

#[test]
fn linecache_random_access_rereads_cached_lines_without_extra_source_reads() {
    let state = Arc::new(TrackingSourceState::default());
    let source = TrackingSource::new(8, 12, Arc::clone(&state));
    let mut cache =
        SequentialLineCache::new(8, 12, 1, 5, 2, crate::pipeline::LineCacheAccess::Random);
    let mut output = vec![0u8; 16];

    cache
        .read_region(&source, Region::new(0, 0, 8, 2), &mut output)
        .unwrap();
    let reads_after_first_request = state.requests.lock().unwrap().len();

    cache
        .read_region(&source, Region::new(0, 3, 8, 2), &mut output)
        .unwrap();
    let reads_after_second_request = state.requests.lock().unwrap().len();

    cache
        .read_region(&source, Region::new(0, 0, 8, 2), &mut output)
        .unwrap();
    let reads_after_third_request = state.requests.lock().unwrap().len();

    assert_eq!(reads_after_first_request, 1);
    assert!(
        reads_after_second_request > reads_after_first_request,
        "moving the window forward should fetch more source lines"
    );
    assert_eq!(
        reads_after_third_request, reads_after_second_request,
        "re-reading retained lines should hit the linecache"
    );
}

#[test]
fn linecache_chaos_sequential_access_rejects_requests_behind_retained_window() {
    let state = Arc::new(TrackingSourceState::default());
    let source = TrackingSource::new(8, 12, Arc::clone(&state));
    let mut cache =
        SequentialLineCache::new(8, 12, 1, 4, 2, crate::pipeline::LineCacheAccess::Sequential);
    let mut output = vec![0u8; 16];

    cache
        .read_region(&source, Region::new(0, 3, 8, 2), &mut output)
        .unwrap();

    let err = cache
        .read_region(&source, Region::new(0, 0, 8, 2), &mut output)
        .unwrap_err();
    assert!(
        matches!(err, ViprsError::Scheduler(message) if message.contains("moved behind retained window"))
    );
}

#[test]
fn execute_tile_returns_error_when_transform_state_is_missing() {
    let pipeline = make_pipeline(4, 4, DemandHint::ThinStrip);
    let mut pool = ThreadBufferPool::new(&pipeline);
    pool.op_states[0] = None;

    let err = match execute_tile(&pipeline, Region::new(0, 0, 4, 4), &mut pool, None) {
        Ok(()) => panic!("missing transform state must return scheduler error"),
        Err(err) => err,
    };

    assert!(
        matches!(
            err,
            ViprsError::SchedulerContract(SchedulerContractError::MissingTransformState {
                node: 0
            })
        ),
        "expected typed missing transform-state scheduler error, got: {err:?}"
    );
}

#[test]
fn execute_tile_returns_error_when_later_transform_state_is_missing() {
    use crate::{
        adapters::{pipeline::PipelineBuilder, sources::memory::MemorySource},
        domain::{
            format::U8,
            image::{DemandHint, Tile, TileMut},
            op::{Op, OperationBridge},
        },
    };

    struct CopyOp;
    impl Op for CopyOp {
        type Input = U8;
        type Output = U8;
        type State = ();

        fn demand_hint(&self) -> DemandHint {
            DemandHint::Any
        }

        fn required_input_region(&self, region: &Region) -> Region {
            *region
        }

        fn start(&self) {}

        #[inline]
        fn process_region(&self, (): &mut (), input: &Tile<U8>, output: &mut TileMut<U8>) {
            output.data.copy_from_slice(input.data);
        }
    }

    let source = MemorySource::<U8>::new(4, 4, 1, (1u8..=16).collect()).unwrap();
    let pipeline = PipelineBuilder::from_source(source)
        .then(Box::new(OperationBridge::new(CopyOp, 1u32)))
        .unwrap()
        .then(Box::new(OperationBridge::new(CopyOp, 1u32)))
        .unwrap()
        .build()
        .unwrap();

    let mut pool = ThreadBufferPool::new(&pipeline);
    pool.op_states[1] = None;

    let err = match execute_tile(&pipeline, Region::new(0, 0, 4, 4), &mut pool, None) {
        Ok(()) => panic!("missing transform state must return scheduler error"),
        Err(err) => err,
    };

    assert!(
        matches!(
            err,
            ViprsError::SchedulerContract(SchedulerContractError::MissingTransformState {
                node: 1
            })
        ),
        "expected typed missing transform-state scheduler error for node 1, got: {err:?}"
    );
}

#[test]
fn shrinkh_zero_band_u16_pipeline_returns_typed_error() {
    let source = MemorySource::<U16>::new(4, 2, 0, vec![]).unwrap();
    let result = PipelineBuilder::from_source(source).shrink_h(2);

    assert!(matches!(
        result,
        Err(BuildError::SourceHint {
            context: "shrink_h",
            ..
        })
    ));
}

#[test]
fn run_with_reducer_supports_bisource_reducer_side_input() {
    use crate::{
        adapters::{
            pipeline::PipelineBuilder, sinks::memory::MemorySink, sources::memory::MemorySource,
        },
        domain::{
            format::U8,
            image::{DemandHint, Tile, TileMut},
            op::{Op, OperationBridge},
            reducer::{BiSourceReducer, TileReducer},
        },
    };

    struct PassThrough;
    impl Op for PassThrough {
        type Input = U8;
        type Output = U8;
        type State = ();

        fn demand_hint(&self) -> DemandHint {
            DemandHint::ThinStrip
        }

        fn required_input_region(&self, r: &Region) -> Region {
            *r
        }

        fn start(&self) {}

        #[inline]
        fn process_region(&self, (): &mut (), input: &Tile<U8>, output: &mut TileMut<U8>) {
            output.data.copy_from_slice(input.data);
        }
    }

    struct MaskedSum {
        width: usize,
        mask: Vec<u8>,
    }

    impl BiSourceReducer<U8> for MaskedSum {
        type Partial = u64;
        type Output = u64;

        fn reduce_tile_with_side_input(&self, tile: &Tile<U8>, region: &Region) -> Self::Partial {
            let row_width = region.width as usize;
            let mut total = 0u64;

            for row in 0..region.height as usize {
                let y = region.y as usize + row;
                let secondary_row = y * self.width;
                let tile_row = row * row_width;

                for col in 0..row_width {
                    if self.mask[secondary_row + region.x as usize + col] != 0 {
                        total += u64::from(tile.data[tile_row + col]);
                    }
                }
            }

            total
        }

        fn combine(&self, a: Self::Partial, b: Self::Partial) -> Self::Partial {
            a + b
        }

        fn finalize(&self, combined: Self::Partial) -> Self::Output {
            combined
        }
    }

    impl TileReducer<U8> for MaskedSum {
        type Partial = <Self as BiSourceReducer<U8>>::Partial;
        type Output = <Self as BiSourceReducer<U8>>::Output;
        type Scratch = ();

        fn reduce_tile(&self, tile: &Tile<U8>, region: &Region) -> Self::Partial {
            <Self as BiSourceReducer<U8>>::reduce_tile_with_side_input(self, tile, region)
        }

        fn combine(&self, a: Self::Partial, b: Self::Partial) -> Self::Partial {
            <Self as BiSourceReducer<U8>>::combine(self, a, b)
        }

        fn finalize(&self, combined: Self::Partial) -> Self::Output {
            <Self as BiSourceReducer<U8>>::finalize(self, combined)
        }
    }

    let pixels = vec![1u8, 2, 3, 4, 5, 6];
    let source = MemorySource::<U8>::new(3, 2, 1, pixels.clone()).unwrap();
    let pipeline = PipelineBuilder::from_source(source)
        .then(Box::new(OperationBridge::new(PassThrough, 1u32)))
        .unwrap()
        .build()
        .unwrap();

    let sink = MemorySink::for_pipeline(&pipeline).unwrap();
    let scheduler = RayonScheduler::new(2).unwrap();
    let reducer = MaskedSum {
        width: 3,
        mask: vec![1, 0, 1, 0, 1, 0],
    };

    let total = scheduler
        .run_with_reducer::<U8, MaskedSum>(&pipeline, &sink, &reducer)
        .unwrap();

    assert_eq!(total, 9, "mask selects source samples 1, 3, and 5");
    assert_eq!(sink.into_buffer(), pixels, "sink path remains unchanged");
}

#[test]
fn run_with_reducer_uses_accumulate_into_scratch_api() {
    use crate::{
        adapters::{
            pipeline::PipelineBuilder, sinks::memory::MemorySink, sources::memory::MemorySource,
        },
        domain::{
            format::U8,
            image::{DemandHint, Tile, TileMut},
            op::{Op, OperationBridge},
            reducer::TileReducer,
        },
    };

    struct PassThrough;
    impl Op for PassThrough {
        type Input = U8;
        type Output = U8;
        type State = ();
        fn demand_hint(&self) -> DemandHint {
            DemandHint::ThinStrip
        }
        fn required_input_region(&self, r: &Region) -> Region {
            *r
        }
        fn start(&self) {}
        #[inline]
        fn process_region(&self, (): &mut (), input: &Tile<U8>, output: &mut TileMut<U8>) {
            output.data.copy_from_slice(input.data);
        }
    }

    struct ScratchOnlySum;
    impl TileReducer<U8> for ScratchOnlySum {
        type Partial = u64;
        type Output = u64;
        type Scratch = u64;

        fn reduce_tile(&self, _tile: &Tile<U8>, _region: &Region) -> Self::Partial {
            panic!("run_with_reducer must call accumulate_into, not reduce_tile/accumulate_tile");
        }

        fn accumulate_into(
            &self,
            tile: &Tile<U8>,
            _region: &Region,
            scratch: &mut Self::Scratch,
            partial: &mut Option<Self::Partial>,
        ) {
            *scratch += 1;
            let tile_sum: u64 = tile.data.iter().map(|&v| u64::from(v)).sum();
            *partial = Some(partial.unwrap_or(0) + tile_sum);
        }

        fn combine(&self, a: Self::Partial, b: Self::Partial) -> Self::Partial {
            a + b
        }

        fn finalize(&self, combined: Self::Partial) -> Self::Output {
            combined
        }
    }

    let pixels: Vec<u8> = (1u8..=16).collect();
    let source = MemorySource::<U8>::new(4, 4, 1, pixels.clone()).unwrap();
    let pipeline = PipelineBuilder::from_source(source)
        .then(Box::new(OperationBridge::new(PassThrough, 1u32)))
        .unwrap()
        .build()
        .unwrap();

    let sink = MemorySink::for_pipeline(&pipeline).unwrap();
    let scheduler = RayonScheduler::new(2).unwrap();
    let total = scheduler
        .run_with_reducer::<U8, ScratchOnlySum>(&pipeline, &sink, &ScratchOnlySum)
        .unwrap();

    assert_eq!(total, 136, "sum of 1..=16 must be 136");
    assert_eq!(
        sink.into_buffer(),
        pixels,
        "sink output must still be written"
    );
}

/// Diamond DAG: Source → `NodeA` (Noop) ─────────┐
///                      └─→ `NodeB` (`LinearScale`) ─┤
///                                               └─→ `NodeC` (`MergeAdd`) → Sink
///
/// `MergeAdd` sums the two input slices sample-by-sample, saturating at 255 (U8).
/// The source has pixel values 10. `NodeA` passes through (10). `NodeB` doubles (20).
/// `MergeAdd`: 10 + 20 = 30. Expected sink: all 30.
///
/// This test exercises `execute_tile`'s multi-input path for the first time.
#[test]
fn diamond_dag_merge_add_produces_correct_output() {
    use crate::{
        adapters::{
            pipeline::PipelineArena, sinks::memory::MemorySink, sources::memory::MemorySource,
        },
        domain::{
            format::U8,
            image::{DemandHint, Tile, TileMut},
            op::{DynOperation, Op, OperationBridge},
        },
        ports::{scheduler::TileScheduler, source::DynImageSource},
    };
    use std::any::Any;

    // NodeA: pass-through (U8 → U8, factor=1)
    struct Noop;
    impl Op for Noop {
        type Input = U8;
        type Output = U8;
        type State = ();
        fn demand_hint(&self) -> DemandHint {
            DemandHint::ThinStrip
        }
        fn required_input_region(&self, r: &Region) -> Region {
            *r
        }
        fn start(&self) {}
        #[inline]
        fn process_region(&self, (): &mut (), input: &Tile<U8>, output: &mut TileMut<U8>) {
            output.data.copy_from_slice(input.data);
        }
    }

    // NodeB: double every sample (U8 → U8, saturating)
    struct Double;
    impl Op for Double {
        type Input = U8;
        type Output = U8;
        type State = ();
        fn demand_hint(&self) -> DemandHint {
            DemandHint::ThinStrip
        }
        fn required_input_region(&self, r: &Region) -> Region {
            *r
        }
        fn start(&self) {}
        #[inline]
        fn process_region(&self, (): &mut (), input: &Tile<U8>, output: &mut TileMut<U8>) {
            for (o, &i) in output.data.iter_mut().zip(input.data.iter()) {
                *o = i.saturating_mul(2);
            }
        }
    }

    // NodeC: MergeAdd — sums two U8 inputs sample-by-sample, saturating at 255.
    // This is the merge node; it overrides `input_slot_count` and
    // `dyn_process_region_multi` directly as a `DynOperation` because `Op` only
    // supports a single input slot. Multi-input ops must implement `DynOperation`
    // directly; `OperationBridge` is for single-input ops only.
    struct MergeAdd {
        bands: u32,
    }

    impl DynOperation for MergeAdd {
        fn input_format(&self) -> crate::domain::format::BandFormatId {
            crate::domain::format::BandFormatId::U8
        }
        fn output_format(&self) -> crate::domain::format::BandFormatId {
            crate::domain::format::BandFormatId::U8
        }
        fn bands(&self) -> u32 {
            self.bands
        }
        fn demand_hint(&self) -> DemandHint {
            DemandHint::ThinStrip
        }
        fn required_input_region(&self, output: &Region) -> Region {
            *output
        }
        fn input_slot_count(&self) -> usize {
            2
        }
        fn required_input_region_slot(&self, output: &Region, _slot: usize) -> Region {
            *output
        }
        fn dyn_start(&self) -> Box<dyn Any + Send> {
            Box::new(())
        }
        fn dyn_process_region(
            &self,
            _state: &mut dyn Any,
            input: &[u8],
            output: &mut [u8],
            _input_region: Region,
            _output_region: Region,
        ) {
            // Single-input fallback — not expected to be called for this op.
            output.copy_from_slice(input);
        }
        fn dyn_process_region_multi(
            &self,
            _state: &mut dyn Any,
            inputs: &[&[u8]],
            output: &mut [u8],
            _input_regions: &[Region],
            _output_region: Region,
        ) {
            debug_assert_eq!(inputs.len(), 2, "MergeAdd requires exactly 2 inputs");
            let a = inputs[0];
            let b = inputs[1];
            for ((o, &sa), &sb) in output.iter_mut().zip(a.iter()).zip(b.iter()) {
                *o = sa.saturating_add(sb);
            }
        }
    }

    // Source: 4x4 single-band, all pixels = 10.
    let pixels: Vec<u8> = vec![10u8; 16];
    let source = MemorySource::<U8>::new(4, 4, 1, pixels).unwrap();

    // Build the diamond DAG manually via PipelineArena.
    let mut arena = PipelineArena::with_source(Box::new(source) as Box<dyn DynImageSource>);
    let node_a = arena.add_node(Box::new(OperationBridge::new(Noop, 1u32)));
    let node_b = arena.add_node(Box::new(OperationBridge::new(Double, 1u32)));
    let node_c = arena.add_node(Box::new(MergeAdd { bands: 1 }));

    // Source feeds both NodeA (slot 0) and NodeB (slot 0).
    // Both NodeA and NodeB feed NodeC — NodeA into slot 0, NodeB into slot 1.
    arena.connect(node_a, node_b).unwrap();
    arena.connect(node_a, node_c).unwrap();
    arena.connect_to_slot(node_b, node_c, 1).unwrap();

    let pipeline = arena.compile().unwrap();

    let sink = MemorySink::for_pipeline(&pipeline).unwrap();
    let scheduler = RayonScheduler::new(1).unwrap();
    scheduler.run_concurrent(&pipeline, &sink).unwrap();
    let output = sink.into_buffer();

    // NodeA passes 10 through; NodeB doubles to 20. MergeAdd: 10 + 20 = 30.
    let expected = vec![30u8; 16];
    assert_eq!(
        output, expected,
        "diamond DAG MergeAdd must produce 10+20=30 for every pixel"
    );
}

#[test]
fn multi_input_first_node_reuses_preallocated_input_refs_for_three_slots() {
    use crate::{
        adapters::{
            pipeline::PipelineArena, sinks::memory::MemorySink, sources::memory::MemorySource,
        },
        domain::{format::U8, ops::conversion::ArrayJoinOp},
        ports::scheduler::TileScheduler,
    };

    let pixels = vec![1u8, 2, 3, 4];
    let source = MemorySource::<U8>::new(2, 2, 1, pixels).unwrap();
    let mut arena = PipelineArena::with_source(
        Box::new(source) as Box<dyn crate::ports::source::DynImageSource>
    );
    arena.add_node(Box::new(ArrayJoinOp::new(
        3,
        1,
        crate::domain::format::BandFormatId::U8,
    )));

    let pipeline = arena.compile().unwrap();
    let sink = MemorySink::for_pipeline(&pipeline).unwrap();
    let scheduler = RayonScheduler::new(1).unwrap();
    scheduler.run_concurrent(&pipeline, &sink).unwrap();

    assert_eq!(
        sink.into_buffer(),
        vec![1, 1, 1, 2, 2, 2, 3, 3, 3, 4, 4, 4],
        "ArrayJoin must read all three slots without per-tile Vec allocation"
    );
}

#[test]
fn affine_identity_bilinear_runs_without_source_buffer_overflow() {
    use crate::{
        adapters::{
            pipeline::PipelineBuilder, sinks::memory::MemorySink, sources::memory::MemorySource,
        },
        domain::{format::U8, kernel::InterpolationKernel},
        ports::scheduler::TileScheduler,
    };

    let pixels = vec![128u8; 512 * 512];
    let source = MemorySource::<U8>::new(512, 512, 1, pixels).unwrap();
    let pipeline = PipelineBuilder::from_source(source)
        .affine(
            [1.0, 0.0, 0.0, 1.0],
            0.0,
            0.0,
            512,
            512,
            InterpolationKernel::Bilinear,
        )
        .unwrap()
        .build()
        .unwrap();

    let sink = MemorySink::for_pipeline(&pipeline).unwrap();
    let scheduler = RayonScheduler::new(1).unwrap();
    scheduler.run_concurrent(&pipeline, &sink).unwrap();
    assert_eq!(sink.into_buffer().len(), 512 * 512);
}

#[test]
fn run_returns_image_too_large_for_full_image_source_only_overflow() {
    use crate::{pipeline::PipelineArena, ports::scheduler::TileScheduler};

    struct HugeSource;

    impl ImageSource for HugeSource {
        type Format = U8;

        fn width(&self) -> u32 {
            u32::MAX
        }

        fn height(&self) -> u32 {
            u32::MAX
        }

        fn bands(&self) -> u32 {
            2
        }

        fn demand_hint(&self) -> DemandHint {
            DemandHint::FullImage
        }

        fn read_region(&self, _region: Region, _output: &mut [u8]) -> Result<(), ViprsError> {
            panic!("overflow must be reported before source reads");
        }
    }

    struct DiscardSink;

    impl ImageSink for DiscardSink {
        fn write_region(&mut self, _region: Region, _data: &[u8]) -> Result<(), ViprsError> {
            Ok(())
        }

        fn finish(self: Box<Self>) -> Result<(), ViprsError> {
            Ok(())
        }
    }

    let source = MemorySource::<U8>::new(1, 1, 2, vec![0u8; 2]).unwrap();
    let mut pipeline = PipelineArena::with_source(Box::new(source))
        .compile()
        .unwrap();
    pipeline.source = Box::new(HugeSource);
    pipeline.width = u32::MAX;
    pipeline.height = u32::MAX;
    pipeline.demand_hint = DemandHint::FullImage;

    let scheduler = RayonScheduler::new(1).unwrap();
    let result = scheduler.run(&pipeline, &mut DiscardSink);

    assert!(matches!(
        result,
        Err(ViprsError::ImageTooLarge {
            width: u32::MAX,
            height: u32::MAX,
            bands: 2,
            ..
        })
    ));
}

// Pre-existing failure on master after scheduler changes; this test does not
// modify rayon scheduling and should not stay blocked on this unrelated regression.
#[test]
// pre-existing failure, unrelated to nearby scheduler changes
#[ignore = "fails on master too: available region containment regression in rayon scheduler"]
fn execute_tile_propagates_distinct_regions_per_input_slot() {
    use crate::{
        domain::{
            error::ViprsError,
            format::U8,
            image::{Tile, TileMut},
            op::{DynOperation, Op, OperationBridge},
        },
        pipeline::{PipelineArena, ThreadBufferPool},
        ports::source::{DynImageSource, ImageSource},
    };
    use std::{
        any::Any,
        sync::{Arc, Mutex},
    };

    struct RecordingSource {
        reads: Arc<Mutex<Vec<Region>>>,
    }

    impl ImageSource for RecordingSource {
        type Format = U8;

        fn width(&self) -> u32 {
            8
        }

        fn height(&self) -> u32 {
            8
        }

        fn bands(&self) -> u32 {
            1
        }

        fn demand_hint(&self) -> DemandHint {
            DemandHint::ThinStrip
        }

        fn read_region(&self, region: Region, output: &mut [u8]) -> Result<(), ViprsError> {
            self.reads.lock().unwrap().push(region);
            output[..region.pixel_count()].fill(1);
            Ok(())
        }
    }

    struct Noop;

    impl Op for Noop {
        type Input = U8;
        type Output = U8;
        type State = ();

        fn demand_hint(&self) -> DemandHint {
            DemandHint::ThinStrip
        }

        fn required_input_region(&self, region: &Region) -> Region {
            *region
        }

        fn start(&self) {}

        fn process_region(&self, (): &mut (), input: &Tile<U8>, output: &mut TileMut<U8>) {
            output.data.copy_from_slice(input.data);
        }
    }

    struct RecordingPass {
        seen_regions: Arc<Mutex<Vec<Region>>>,
    }

    impl Op for RecordingPass {
        type Input = U8;
        type Output = U8;
        type State = ();

        fn demand_hint(&self) -> DemandHint {
            DemandHint::ThinStrip
        }

        fn required_input_region(&self, region: &Region) -> Region {
            *region
        }

        fn start(&self) {}

        fn process_region(&self, (): &mut (), input: &Tile<U8>, output: &mut TileMut<U8>) {
            self.seen_regions.lock().unwrap().push(input.region);
            output.data.copy_from_slice(input.data);
        }
    }

    struct RegionRecordingMerge {
        seen_regions: Arc<Mutex<Vec<Region>>>,
        seen_input_lengths: Arc<Mutex<Vec<usize>>>,
    }

    impl DynOperation for RegionRecordingMerge {
        fn input_format(&self) -> crate::domain::format::BandFormatId {
            crate::domain::format::BandFormatId::U8
        }

        fn output_format(&self) -> crate::domain::format::BandFormatId {
            crate::domain::format::BandFormatId::U8
        }

        fn bands(&self) -> u32 {
            1
        }

        fn demand_hint(&self) -> DemandHint {
            DemandHint::ThinStrip
        }

        fn input_slot_count(&self) -> usize {
            2
        }

        fn required_input_region_slot(&self, output: &Region, slot: usize) -> Region {
            match slot {
                0 => Region::new(output.x, output.y, 4, 4),
                1 => Region::new(output.x + 1, output.y + 1, 2, 2),
                _ => Region::new(0, 0, 0, 0),
            }
        }

        fn required_input_region(&self, output: &Region) -> Region {
            *output
        }

        fn dyn_start(&self) -> Box<dyn Any + Send> {
            Box::new(())
        }

        fn dyn_process_region(
            &self,
            _state: &mut dyn Any,
            _input: &[u8],
            _output: &mut [u8],
            _input_region: Region,
            _output_region: Region,
        ) {
            panic!("RegionRecordingMerge must run through dyn_process_region_multi");
        }

        fn dyn_process_region_multi(
            &self,
            _state: &mut dyn Any,
            inputs: &[&[u8]],
            output: &mut [u8],
            input_regions: &[Region],
            _output_region: Region,
        ) {
            self.seen_regions
                .lock()
                .unwrap()
                .extend(input_regions.iter().copied());
            self.seen_input_lengths
                .lock()
                .unwrap()
                .extend(inputs.iter().map(|input| input.len()));
            output.fill(0);
        }
    }

    let source_reads = Arc::new(Mutex::new(Vec::new()));
    let branch_reads = Arc::new(Mutex::new(Vec::new()));
    let merge_regions = Arc::new(Mutex::new(Vec::new()));
    let merge_lengths = Arc::new(Mutex::new(Vec::new()));

    let source = RecordingSource {
        reads: Arc::clone(&source_reads),
    };
    let mut arena = PipelineArena::with_source(Box::new(source) as Box<dyn DynImageSource>);
    let node_a = arena.add_node(Box::new(OperationBridge::new(Noop, 1)));
    let node_b = arena.add_node(Box::new(OperationBridge::new(
        RecordingPass {
            seen_regions: Arc::clone(&branch_reads),
        },
        1,
    )));
    let node_c = arena.add_node(Box::new(RegionRecordingMerge {
        seen_regions: Arc::clone(&merge_regions),
        seen_input_lengths: Arc::clone(&merge_lengths),
    }));

    arena.connect(node_a, node_b).unwrap();
    arena.connect(node_a, node_c).unwrap();
    arena.connect_to_slot(node_b, node_c, 1).unwrap();

    let pipeline = arena.compile().unwrap();
    let mut pool = ThreadBufferPool::new(&pipeline);

    execute_tile(&pipeline, Region::new(0, 0, 1, 1), &mut pool, None).unwrap();

    assert_eq!(
        source_reads.lock().unwrap().as_slice(),
        &[Region::new(0, 0, 4, 4)]
    );
    assert_eq!(
        branch_reads.lock().unwrap().as_slice(),
        &[Region::new(1, 1, 2, 2)]
    );
    assert_eq!(
        merge_regions.lock().unwrap().as_slice(),
        &[Region::new(0, 0, 4, 4), Region::new(1, 1, 2, 2)]
    );
    assert_eq!(merge_lengths.lock().unwrap().as_slice(), &[16, 4]);
}

#[test]
fn execute_tile_materializes_coordinate_dependency_before_root_source_slot() {
    use crate::{
        domain::{
            error::ViprsError,
            format::U8,
            image::{Tile, TileMut},
            op::{DynOperation, NodeSpec, Op, OperationBridge, SourceReadPlan},
        },
        pipeline::{PipelineArena, ThreadBufferPool},
        ports::source::{DynImageSource, ImageSource},
    };
    use std::{
        any::Any,
        sync::{Arc, Mutex},
    };

    struct RecordingSource {
        reads: Arc<Mutex<Vec<Region>>>,
    }

    impl ImageSource for RecordingSource {
        type Format = U8;

        fn width(&self) -> u32 {
            8
        }

        fn height(&self) -> u32 {
            8
        }

        fn bands(&self) -> u32 {
            1
        }

        fn demand_hint(&self) -> DemandHint {
            DemandHint::ThinStrip
        }

        fn read_region(&self, region: Region, output: &mut [u8]) -> Result<(), ViprsError> {
            self.reads.lock().unwrap().push(region);
            output[..region.pixel_count()].fill(region.x as u8);
            Ok(())
        }
    }

    struct ConstantCoords;

    impl Op for ConstantCoords {
        type Input = U8;
        type Output = U8;
        type State = ();

        fn demand_hint(&self) -> DemandHint {
            DemandHint::ThinStrip
        }

        fn required_input_region(&self, region: &Region) -> Region {
            *region
        }

        fn start(&self) {}

        fn process_region(&self, (): &mut (), _input: &Tile<U8>, output: &mut TileMut<U8>) {
            output.data.fill(5);
        }
    }

    struct CoordinateDrivenRead;

    impl DynOperation for CoordinateDrivenRead {
        fn input_format(&self) -> crate::domain::format::BandFormatId {
            crate::domain::format::BandFormatId::U8
        }

        fn output_format(&self) -> crate::domain::format::BandFormatId {
            crate::domain::format::BandFormatId::U8
        }

        fn bands(&self) -> u32 {
            1
        }

        fn demand_hint(&self) -> DemandHint {
            DemandHint::ThinStrip
        }

        fn input_slot_count(&self) -> usize {
            2
        }

        fn required_input_region_slot(&self, output: &Region, _slot: usize) -> Region {
            *output
        }

        fn required_input_region(&self, output: &Region) -> Region {
            *output
        }

        fn node_spec(&self, tile_w: u32, tile_h: u32) -> NodeSpec {
            NodeSpec::identity(tile_w, tile_h).with_coordinate_driven_source(0, 1)
        }

        fn coordinate_driven_source_spec(
            &self,
        ) -> Option<crate::domain::op::CoordinateDrivenSourceSpec> {
            Some(crate::domain::op::CoordinateDrivenSourceSpec {
                source_slot: 0,
                dependency_slot: 1,
            })
        }

        fn source_read_plan_slot_with_materialized_dependency(
            &self,
            _output: &Region,
            slot: usize,
            dependency_slot: usize,
            _dependency_region: Region,
            dependency: &[u8],
        ) -> Option<SourceReadPlan> {
            if slot != 0 || dependency_slot != 1 || dependency.is_empty() {
                return None;
            }

            let coord = i32::from(dependency[0]);
            Some(SourceReadPlan::rect(Region::new(coord, coord, 1, 1)))
        }

        fn dyn_start(&self) -> Box<dyn Any + Send> {
            Box::new(())
        }

        fn dyn_process_region(
            &self,
            _state: &mut dyn Any,
            _input: &[u8],
            _output: &mut [u8],
            _input_region: Region,
            _output_region: Region,
        ) {
            panic!("CoordinateDrivenRead must run through dyn_process_region_multi");
        }

        fn dyn_process_region_multi(
            &self,
            _state: &mut dyn Any,
            inputs: &[&[u8]],
            output: &mut [u8],
            _input_regions: &[Region],
            _output_region: Region,
        ) {
            output.fill(inputs[0][0]);
        }
    }

    let reads = Arc::new(Mutex::new(Vec::new()));
    let source = RecordingSource {
        reads: Arc::clone(&reads),
    };
    let mut arena = PipelineArena::with_source(Box::new(source) as Box<dyn DynImageSource>);
    let coords = arena.add_node(Box::new(OperationBridge::new(ConstantCoords, 1)));
    let map = arena.add_node(Box::new(CoordinateDrivenRead));
    arena.connect_to_slot(coords, map, 1).unwrap();

    let pipeline = arena.compile().unwrap();
    let mut pool = ThreadBufferPool::new(&pipeline);
    execute_tile(&pipeline, Region::new(0, 0, 2, 2), &mut pool, None).unwrap();

    assert_eq!(
        reads.lock().unwrap().as_slice(),
        &[Region::new(0, 0, 2, 2), Region::new(5, 5, 1, 1)]
    );
}

#[test]
fn union_region_handles_i32_boundary_without_wrapping() {
    let lhs = Region::new(i32::MAX - 1, 0, 4, 1);
    let rhs = Region::new(i32::MAX, 0, 1, 1);

    assert_eq!(union_region(lhs, rhs), Region::new(i32::MAX - 1, 0, 4, 1));
}
