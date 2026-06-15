use criterion::{BenchmarkId, Criterion, black_box, criterion_group, criterion_main};
use std::num::NonZeroUsize;
use viprs::{
    adapters::{
        pipeline::{CompiledPipeline, PipelineBuilder},
        scheduler::rayon_scheduler::RayonScheduler,
        sinks::memory::MemorySink,
        sources::memory::MemorySource,
    },
    domain::{
        format::U8,
        kernel::InterpolationKernel,
        ops::resample::thumbnail::{Thumbnail, ThumbnailTarget},
    },
    ports::scheduler::TileScheduler,
};

const STANDARD_SIZES: [u32; 3] = [512, 2048, 8192];
const TARGET_WIDTH: u32 = 128;
const RUNS_PER_SAMPLE: usize = 10;

fn thumbnail_pipeline(
    size: u32,
    pixels: Vec<u8>,
    cache_tiles: Option<NonZeroUsize>,
) -> CompiledPipeline {
    let source = MemorySource::<U8>::new(size, size, 1, pixels).unwrap();
    let builder = PipelineBuilder::from_source(source)
        .thumbnail(Thumbnail::new(
            ThumbnailTarget::Width(TARGET_WIDTH),
            InterpolationKernel::Lanczos3,
        ))
        .unwrap();
    let builder = if let Some(cache_tiles) = cache_tiles {
        builder.cache_last_op(cache_tiles).unwrap()
    } else {
        builder
    };
    builder.build().unwrap()
}

fn run_thumbnail(pipeline: &CompiledPipeline, scheduler: &RayonScheduler) -> Vec<u8> {
    let mut sink = MemorySink::for_pipeline(pipeline);
    scheduler.run(pipeline, &mut sink).unwrap();
    sink.into_buffer()
}

fn bench_cache_vs_nocache(c: &mut Criterion) {
    let mut group = c.benchmark_group("cache_vs_nocache");

    for &size in &STANDARD_SIZES {
        let pixels = vec![128u8; size as usize * size as usize];

        group.bench_with_input(
            BenchmarkId::new("thumbnail_no_cache", size),
            &size,
            |b, &size| {
                let scheduler = RayonScheduler::new(1).unwrap();
                b.iter(|| {
                    let mut last = Vec::new();
                    for _ in 0..RUNS_PER_SAMPLE {
                        let pipeline = thumbnail_pipeline(size, pixels.clone(), None);
                        last = run_thumbnail(&pipeline, &scheduler);
                    }
                    black_box(last)
                });
            },
        );

        group.bench_with_input(
            BenchmarkId::new("thumbnail_with_cache", size),
            &size,
            |b, &size| {
                let scheduler = RayonScheduler::new(1).unwrap();
                let pipeline =
                    thumbnail_pipeline(size, pixels.clone(), Some(NonZeroUsize::new(64).unwrap()));
                b.iter(|| {
                    pipeline.clear_tile_cache().unwrap();
                    let mut last = Vec::new();
                    for _ in 0..RUNS_PER_SAMPLE {
                        last = run_thumbnail(&pipeline, &scheduler);
                    }
                    black_box(last)
                });
            },
        );
    }

    group.finish();
}

criterion_group!(benches, bench_cache_vs_nocache);
criterion_main!(benches);
