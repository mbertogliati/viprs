#[path = "../common/mod.rs"]
mod common;

use criterion::{BenchmarkId, Criterion, black_box, criterion_group, criterion_main};
use viprs::{
    adapters::{
        pipeline::PipelineBuilder, scheduler::rayon_scheduler::RayonScheduler,
        sinks::memory::MemorySink, sources::memory::MemorySource,
    },
    domain::{
        format::U8,
        kernel::InterpolationKernel,
        ops::resample::thumbnail::{Thumbnail, ThumbnailTarget},
    },
    ports::scheduler::TileScheduler,
};

fn bench_thumbnail(c: &mut Criterion) {
    let mut group = c.benchmark_group("thumbnail_u8_rgba");

    for &size in &common::STANDARD_SIZES {
        let pixels = vec![128u8; size as usize * size as usize * 4];
        let target_width = (size / 4).max(1);

        group.bench_with_input(BenchmarkId::from_parameter(size), &size, |b, &size| {
            b.iter(|| {
                let source = MemorySource::<U8>::new(size, size, 4, pixels.clone()).unwrap();
                let pipeline = PipelineBuilder::from_source(source)
                    .thumbnail(Thumbnail::new(
                        ThumbnailTarget::Width(target_width),
                        InterpolationKernel::Lanczos3,
                    ))
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

    let mut rgb_group = c.benchmark_group("thumbnail_u8_rgb");

    for &size in &common::STANDARD_SIZES {
        let pixels = vec![128u8; size as usize * size as usize * 3];
        let target_width = (size / 4).max(1);

        rgb_group.bench_with_input(BenchmarkId::from_parameter(size), &size, |b, &size| {
            b.iter(|| {
                let source = MemorySource::<U8>::new(size, size, 3, pixels.clone()).unwrap();
                let pipeline = PipelineBuilder::from_source(source)
                    .thumbnail(Thumbnail::new(
                        ThumbnailTarget::Width(target_width),
                        InterpolationKernel::Lanczos3,
                    ))
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

criterion_group!(benches, bench_thumbnail);
criterion_main!(benches);
