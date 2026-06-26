#![allow(missing_docs)]
/// Benchmark: Linear<U8> — per-sample `output = input * scale + offset`.
///
/// Measures the full pipeline path: MemorySource → Linear → MemorySink via RayonScheduler.
/// Linear is built via PipelineBuilder::linear, which dispatches over the pipeline format.
use criterion::{BenchmarkId, Criterion, black_box, criterion_group, criterion_main};
use viprs::{
    adapters::{
        pipeline::ImagePipeline, scheduler::rayon_scheduler::RayonScheduler,
        sinks::memory::MemorySink, sources::memory::MemorySource,
    },
    domain::format::U8,
    ports::scheduler::TileScheduler,
};

fn bench_linear(c: &mut Criterion) {
    let mut group = c.benchmark_group("linear_u8");

    for &size in &[512u32, 2048, 8192] {
        let pixel_count = (size as usize) * (size as usize);
        let pixels = vec![100u8; pixel_count];

        group.bench_with_input(BenchmarkId::from_parameter(size), &size, |b, &size| {
            b.iter(|| {
                let source = MemorySource::<U8>::new(size, size, 1, pixels.clone()).unwrap();
                // scale=1, offset=1 in U8 arithmetic (truncating cast from f64).
                let pipeline = ImagePipeline::from_source(source)
                    .linear(1.0, 1.0)
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

    let mut rgb_group = c.benchmark_group("linear_u8_rgb");

    for &size in &[512u32, 2048, 8192] {
        let pixel_count = (size as usize) * (size as usize);
        let pixels = vec![100u8; pixel_count * 3];

        rgb_group.bench_with_input(BenchmarkId::from_parameter(size), &size, |b, &size| {
            b.iter(|| {
                let source = MemorySource::<U8>::new(size, size, 3, pixels.clone()).unwrap();
                let pipeline = ImagePipeline::from_source(source)
                    .linear(1.0, 1.0)
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

criterion_group!(benches, bench_linear);
criterion_main!(benches);
