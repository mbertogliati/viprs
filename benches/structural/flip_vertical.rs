#![allow(missing_docs)]
/// Benchmark: FlipVertical<U8> — mirror the image top-to-bottom.
///
/// Measures the full pipeline path: MemorySource → FlipVertical → MemorySink via
/// RayonScheduler. process_region reverses row order within each tile — O(pixels)
/// work with no extra allocation.
///
/// Note: process_region currently copies rows in reverse order into the output buffer
/// (see B-021). When B-021 (zero-copy structural ops) is resolved, this
/// benchmark provides the baseline to quantify the improvement.
use criterion::{BenchmarkId, Criterion, black_box, criterion_group, criterion_main};
use viprs::{
    adapters::{
      pipeline::ImagePipeline, scheduler::rayon_scheduler::RayonScheduler,
      sinks::memory::MemorySink, sources::memory::MemorySource,
    },
    domain::format::U8,
    ports::scheduler::TileScheduler,
};

fn bench_flip_vertical(c: &mut Criterion) {
    let mut group = c.benchmark_group("flip_vertical_u8");

    for &size in &[512u32, 2048, 8192] {
        let pixel_count = (size as usize) * (size as usize);
        let pixels = vec![128u8; pixel_count];

        group.bench_with_input(BenchmarkId::from_parameter(size), &size, |b, &size| {
            b.iter(|| {
                let source = MemorySource::<U8>::new(size, size, 1, pixels.clone()).unwrap();
                let pipeline = ImagePipeline::from_source(source)
                    .flip_vertical()
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
}

criterion_group!(benches, bench_flip_vertical);
criterion_main!(benches);
