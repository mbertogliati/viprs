#![allow(missing_docs)]
/// Benchmark: ShrinkH<U8> — horizontal integer shrink by factor 2.
///
/// Measures the full pipeline path: MemorySource → ShrinkH → MemorySink via
/// RayonScheduler. process_region averages `factor` consecutive pixels per output
/// pixel — O(width/factor * height) work with no heap allocation per tile.
///
/// Factor-sweep group (`shrinkh_u8_rgb_factor_sweep`) covers factors 3–50 at
/// 8192×8192 to validate NEON dispatch thresholds across the full range. The
/// dispatch routes factor < 16 → vld1_u8 NEON, factor ≥ 16 → chunked vld3q_u8.
/// A visible throughput drop at factor=16 would indicate the wrong dispatch choice.
use criterion::{BenchmarkId, Criterion, black_box, criterion_group, criterion_main};
use viprs::domain::ops::resample::shrinkh::ShrinkH;
use viprs::{
    adapters::{
        pipeline::ImagePipeline, scheduler::rayon_scheduler::RayonScheduler,
        sinks::memory::MemorySink, sources::memory::MemorySource,
    },
    domain::format::U8,
    domain::op::OperationBridge,
    ports::scheduler::TileScheduler,
};

fn bench_shrinkh(c: &mut Criterion) {
    let mut group = c.benchmark_group("shrinkh_u8_factor2");

    for &size in &[512u32, 2048, 8192] {
        let pixels = vec![128u8; size as usize * size as usize];

        group.bench_with_input(BenchmarkId::from_parameter(size), &size, |b, &size| {
            b.iter(|| {
                let source = MemorySource::<U8>::new(size, size, 1, pixels.clone()).unwrap();
                let op = Box::new(OperationBridge::new(ShrinkH::<U8>::new(2).unwrap(), 1u32));
                let pipeline = ImagePipeline::from_source(source)
                    .then(op)
                    .unwrap()
                    .build()
                    .unwrap();
                let mut sink = MemorySink::for_pipeline(&pipeline).unwrap();
                RayonScheduler::new(RayonScheduler::default_threads())
                    .unwrap()
                    .run(&pipeline, &mut sink)
                    .unwrap();
                black_box(sink.into_buffer())
            });
        });
    }

    group.finish();

    let mut rgb_group = c.benchmark_group("shrinkh_u8_rgb_factor2");

    for &size in &[512u32, 2048, 8192] {
        let pixels = vec![128u8; size as usize * size as usize * 3];

        rgb_group.bench_with_input(BenchmarkId::from_parameter(size), &size, |b, &size| {
            b.iter(|| {
                let source = MemorySource::<U8>::new(size, size, 3, pixels.clone()).unwrap();
                let op = Box::new(OperationBridge::new(ShrinkH::<U8>::new(2).unwrap(), 3u32));
                let pipeline = ImagePipeline::from_source(source)
                    .then(op)
                    .unwrap()
                    .build()
                    .unwrap();
                let mut sink = MemorySink::for_pipeline(&pipeline).unwrap();
                RayonScheduler::new(RayonScheduler::default_threads())
                    .unwrap()
                    .run(&pipeline, &mut sink)
                    .unwrap();
                black_box(sink.into_buffer())
            });
        });
    }

    rgb_group.finish();
}

/// Sweep factors 3–50 at 8192×8192 RGB to validate the NEON dispatch threshold.
///
/// Factors tested include:
///   - 3–15  (vld1_u8 NEON path)
///   - 10    (thumbnail 800 from 8192: factor = floor(8192/800) = 10)
///   - 14–17 (straddles dispatch threshold at 16)
///   - 19    (thumbnail 400 from 8192)
///   - 39    (thumbnail 200 from 8192)
///   - 50    (extreme downscale)
///
/// Look for: monotonic throughput decrease as factor grows (expected — more work
/// per output pixel); and specifically NO discontinuity at factor=15→16 (the
/// dispatch boundary). A step-change there means the wrong path is chosen.
fn bench_shrinkh_factor_sweep(c: &mut Criterion) {
    const SIZE: u32 = 8192;
    const BANDS: usize = 3;
    let pixels = vec![128u8; SIZE as usize * SIZE as usize * BANDS];

    let sweep_factors: &[u32] = &[
        3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 25, 30, 39, 50,
    ];

    let mut group = c.benchmark_group("shrinkh_u8_rgb_factor_sweep");
    group.sample_size(10);

    for &factor in sweep_factors {
        group.bench_with_input(
            BenchmarkId::from_parameter(factor),
            &factor,
            |b, &factor| {
                b.iter(|| {
                    let source =
                        MemorySource::<U8>::new(SIZE, SIZE, BANDS as u32, pixels.clone()).unwrap();
                    let op = Box::new(OperationBridge::new(
                        ShrinkH::<U8>::new(factor).unwrap(),
                        BANDS as u32,
                    ));
                    let pipeline = ImagePipeline::from_source(source)
                        .then(op)
                        .unwrap()
                        .build()
                        .unwrap();
                    let mut sink = MemorySink::for_pipeline(&pipeline).unwrap();
                    RayonScheduler::new(RayonScheduler::default_threads())
                        .unwrap()
                        .run(&pipeline, &mut sink)
                        .unwrap();
                    black_box(sink.into_buffer())
                });
            },
        );
    }

    group.finish();
}

criterion_group!(benches, bench_shrinkh, bench_shrinkh_factor_sweep);
criterion_main!(benches);
