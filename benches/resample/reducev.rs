/// Benchmark: ReduceV<U8> — vertical downscale via Lanczos3.
use criterion::{BenchmarkId, Criterion, black_box, criterion_group, criterion_main};
use viprs::{
    adapters::{
        pipeline::PipelineBuilder, scheduler::rayon_scheduler::RayonScheduler,
        sinks::memory::MemorySink, sources::memory::MemorySource,
    },
    domain::{format::U8, kernel::InterpolationKernel},
    ports::scheduler::TileScheduler,
};

fn bench_reducev(c: &mut Criterion) {
    let mut group = c.benchmark_group("reducev_u8_factor2");

    for &size in &[512u32, 2048, 8192] {
        let pixels = vec![128u8; size as usize * size as usize];

        group.bench_with_input(BenchmarkId::from_parameter(size), &size, |b, &size| {
            b.iter(|| {
                let source = MemorySource::<U8>::new(size, size, 1, pixels.clone()).unwrap();
                let pipeline = PipelineBuilder::from_source(source)
                    .reduce_v(2.0, InterpolationKernel::Lanczos3)
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
}

fn bench_reducev_thumbnail_residual(c: &mut Criterion) {
    let mut group = c.benchmark_group("reducev_u8_thumbnail_residual");
    let input_w = 400u32;
    let input_h = 409u32;
    let bands = 3u32;
    let factor = 409.0 / 400.0;
    let pixels = vec![128u8; input_w as usize * input_h as usize * bands as usize];

    group.bench_function("400x409_rgb_to_400x400", |b| {
        b.iter(|| {
            let source = MemorySource::<U8>::new(input_w, input_h, bands, pixels.clone()).unwrap();
            let pipeline = PipelineBuilder::from_source(source)
                .reduce_v(factor, InterpolationKernel::Lanczos3)
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

    group.finish();
}

criterion_group!(benches, bench_reducev, bench_reducev_thumbnail_residual);
criterion_main!(benches);
