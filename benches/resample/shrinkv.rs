/// Benchmark: ShrinkV<U8> — vertical integer shrink by factor 2.
///
/// Measures the full pipeline path: MemorySource → ShrinkV → MemorySink via
/// RayonScheduler. process_region averages `factor` consecutive rows per output
/// row — O(width * height/factor) work with no heap allocation per tile.
use criterion::{BenchmarkId, Criterion, black_box, criterion_group, criterion_main};
use viprs::domain::ops::resample::shrinkv::ShrinkV;
use viprs::{
    adapters::{
        pipeline::PipelineBuilder, scheduler::rayon_scheduler::RayonScheduler,
        sinks::memory::MemorySink, sources::memory::MemorySource,
    },
    domain::format::U8,
    domain::op::OperationBridge,
    ports::scheduler::TileScheduler,
};

fn bench_shrinkv(c: &mut Criterion) {
    let mut group = c.benchmark_group("shrinkv_u8_factor2");

    for &size in &[512u32, 2048, 8192] {
        let pixels = vec![128u8; size as usize * size as usize];

        group.bench_with_input(BenchmarkId::from_parameter(size), &size, |b, &size| {
            b.iter(|| {
                let source = MemorySource::<U8>::new(size, size, 1, pixels.clone()).unwrap();
                let op = Box::new(OperationBridge::new(ShrinkV::<U8>::new(2).unwrap(), 1u32));
                let pipeline = PipelineBuilder::from_source(source)
                    .then(op)
                    .unwrap()
                    .build()
                    .unwrap();
                let mut sink = MemorySink::for_pipeline(&pipeline);
                RayonScheduler::new(RayonScheduler::default_threads())
                    .unwrap()
                    .run(&pipeline, &mut sink)
                    .unwrap();
                black_box(sink.into_buffer())
            });
        });
    }

    group.finish();

    let mut rgb_group = c.benchmark_group("shrinkv_u8_rgb_factor2");

    for &size in &[512u32, 2048, 8192] {
        let pixels = vec![128u8; size as usize * size as usize * 3];

        rgb_group.bench_with_input(BenchmarkId::from_parameter(size), &size, |b, &size| {
            b.iter(|| {
                let source = MemorySource::<U8>::new(size, size, 3, pixels.clone()).unwrap();
                let op = Box::new(OperationBridge::new(ShrinkV::<U8>::new(2).unwrap(), 3u32));
                let pipeline = PipelineBuilder::from_source(source)
                    .then(op)
                    .unwrap()
                    .build()
                    .unwrap();
                let mut sink = MemorySink::for_pipeline(&pipeline);
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

criterion_group!(benches, bench_shrinkv);
criterion_main!(benches);
