#![allow(missing_docs)]
use criterion::{BenchmarkId, Criterion, black_box, criterion_group, criterion_main};
use viprs::{
    adapters::{
        pipeline::internal::PipelinePlan, scheduler::rayon_scheduler::RayonScheduler,
        sinks::memory::MemorySink, sources::memory::MemorySource,
    },
    domain::{format::U8, kernel::InterpolationKernel, ops::resample::resize::Resize},
    ports::scheduler::TileScheduler,
};

fn bench_resize(c: &mut Criterion) {
    let mut group = c.benchmark_group("resize_u8");

    for &(src_size, dst_size) in &[(512u32, 256u32), (512, 1024), (2048, 512)] {
        let pixels = vec![128u8; src_size as usize * src_size as usize];

        group.bench_with_input(
            BenchmarkId::new("resize", format!("{src_size}_to_{dst_size}")),
            &(src_size, dst_size),
            |b, &(src_size, dst_size)| {
                b.iter(|| {
                    let source =
                        MemorySource::<U8>::new(src_size, src_size, 1, pixels.clone()).unwrap();
                    let scale = dst_size as f64 / src_size as f64;
                    let pipeline = PipelinePlan::from_source(source)
                        .plan_resize(Resize::new(scale, scale, InterpolationKernel::Lanczos3))
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
            },
        );
    }

    group.finish();

    let mut rgb_group = c.benchmark_group("resize_u8_rgb");

    for &size in &[512u32, 2048, 8192] {
        let dst_size = (size / 2).max(1);
        let pixels = vec![128u8; size as usize * size as usize * 3];

        rgb_group.bench_with_input(BenchmarkId::new("resize", size), &size, |b, &size| {
            b.iter(|| {
                let source = MemorySource::<U8>::new(size, size, 3, pixels.clone()).unwrap();
                let scale = dst_size as f64 / size as f64;
                let pipeline = PipelinePlan::from_source(source)
                    .plan_resize(Resize::new(scale, scale, InterpolationKernel::Lanczos3))
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

    rgb_group.finish();
}

criterion_group!(benches, bench_resize);
criterion_main!(benches);
