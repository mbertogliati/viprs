#![allow(missing_docs)]
use criterion::{BenchmarkId, Criterion, black_box, criterion_group, criterion_main};
use viprs::{
    adapters::{
      pipeline::ImagePipeline, scheduler::rayon_scheduler::RayonScheduler,
      sinks::memory::MemorySink, sources::memory::MemorySource,
    },
    domain::format::U8,
    ports::scheduler::TileScheduler,
};

fn must<T, E: std::fmt::Display>(result: Result<T, E>, context: &str) -> T {
    match result {
        Ok(value) => value,
        Err(error) => panic!("{context}: {error}"),
    }
}

fn bench_flip(c: &mut Criterion) {
    let mut group = c.benchmark_group("conversion_flip_u8");

    for &size in &[512u32, 2048, 8192] {
        let pixels = vec![128u8; size as usize * size as usize];

        group.bench_with_input(BenchmarkId::new("horizontal", size), &size, |b, &size| {
            b.iter(|| {
                let source = must(
                    MemorySource::<U8>::new(size, size, 1, pixels.clone()),
                    "create memory source",
                );
                let builder = must(
                  ImagePipeline::from_source(source).flip_horizontal(),
                  "add horizontal flip operation",
                );
                let pipeline = must(builder.build(), "build pipeline");
                let mut sink = MemorySink::for_pipeline(&pipeline).unwrap();
                let scheduler = must(
                    RayonScheduler::new(RayonScheduler::default_threads()),
                    "create rayon scheduler",
                );
                must(scheduler.run(&pipeline, &mut sink), "run pipeline");
                black_box(sink.into_buffer())
            });
        });

        group.bench_with_input(BenchmarkId::new("vertical", size), &size, |b, &size| {
            b.iter(|| {
                let source = must(
                    MemorySource::<U8>::new(size, size, 1, pixels.clone()),
                    "create memory source",
                );
                let builder = must(
                  ImagePipeline::from_source(source).flip_vertical(),
                  "add vertical flip operation",
                );
                let pipeline = must(builder.build(), "build pipeline");
                let mut sink = MemorySink::for_pipeline(&pipeline).unwrap();
                let scheduler = must(
                    RayonScheduler::new(RayonScheduler::default_threads()),
                    "create rayon scheduler",
                );
                must(scheduler.run(&pipeline, &mut sink), "run pipeline");
                black_box(sink.into_buffer())
            });
        });
    }

    group.finish();
}

criterion_group!(benches, bench_flip);
criterion_main!(benches);
