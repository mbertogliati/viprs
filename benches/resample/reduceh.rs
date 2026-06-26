#![allow(missing_docs)]
/// Benchmark: ReduceH<U8/F32> — horizontal downscale via Lanczos3.
use criterion::{BenchmarkId, Criterion, black_box, criterion_group, criterion_main};
use viprs::{
    adapters::{
        pipeline::internal::PipelinePlan, scheduler::rayon_scheduler::RayonScheduler,
        sinks::memory::MemorySink, sources::memory::MemorySource,
    },
    domain::{
        format::{F32, U8},
        kernel::InterpolationKernel,
    },
    ports::scheduler::TileScheduler,
};

fn bench_reduceh_u8(c: &mut Criterion) {
    let mut group = c.benchmark_group("reduceh_u8_factor2");

    for &size in &[512u32, 2048, 8192] {
        let pixels = vec![128u8; size as usize * size as usize];

        group.bench_with_input(BenchmarkId::from_parameter(size), &size, |b, &size| {
            b.iter(|| {
                let source = MemorySource::<U8>::new(size, size, 1, pixels.clone()).unwrap();
                let pipeline = PipelinePlan::from_source(source)
                    .plan_reduce_h(2.0, InterpolationKernel::Lanczos3)
                    .unwrap()
                    .compile()
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

fn bench_reduceh_f32(c: &mut Criterion) {
    let mut group = c.benchmark_group("reduceh_f32_factor2");

    for &size in &[512u32, 2048, 8192] {
        let pixels = vec![0.5f32; size as usize * size as usize];

        group.bench_with_input(BenchmarkId::from_parameter(size), &size, |b, &size| {
            b.iter(|| {
                let source = MemorySource::<F32>::new(size, size, 1, pixels.clone()).unwrap();
                let pipeline = PipelinePlan::from_source(source)
                    .plan_reduce_h(2.0, InterpolationKernel::Lanczos3)
                    .unwrap()
                    .compile()
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

fn bench_reduceh_thumbnail_residual(c: &mut Criterion) {
    let mut group = c.benchmark_group("reduceh_u8_thumbnail_residual");
    let input_w = 409u32;
    let input_h = 409u32;
    let bands = 3u32;
    let factor = 409.0 / 400.0;
    let pixels = vec![128u8; input_w as usize * input_h as usize * bands as usize];

    group.bench_function("409x409_rgb_to_400x409", |b| {
        b.iter(|| {
            let source = MemorySource::<U8>::new(input_w, input_h, bands, pixels.clone()).unwrap();
            let pipeline = PipelinePlan::from_source(source)
                .plan_reduce_h(factor, InterpolationKernel::Lanczos3)
                .unwrap()
                .compile()
                .unwrap();
            let mut sink = MemorySink::for_pipeline(&pipeline).unwrap();
            RayonScheduler::new(RayonScheduler::default_threads())
                .unwrap()
                .run(&pipeline, &mut sink)
                .unwrap();
            black_box(sink.into_buffer())
        });
    });

    group.finish();
}

criterion_group!(
    benches,
    bench_reduceh_u8,
    bench_reduceh_f32,
    bench_reduceh_thumbnail_residual
);
criterion_main!(benches);
