use criterion::{BenchmarkId, Criterion, black_box, criterion_group, criterion_main};
use viprs::{
    adapters::{
        pipeline::PipelineBuilder, scheduler::rayon_scheduler::RayonScheduler,
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

fn bench_wrap(c: &mut Criterion) {
    let mut group = c.benchmark_group("conversion_wrap_center_u8");

    for &size in &[512u32, 2048, 8192] {
        let pixels = vec![128u8; size as usize * size as usize];
        let displacement = (size / 2) as i32;

        group.bench_with_input(BenchmarkId::from_parameter(size), &size, |b, &size| {
            b.iter(|| {
                let source = must(
                    MemorySource::<U8>::new(size, size, 1, pixels.clone()),
                    "create memory source",
                );
                let builder = must(
                    PipelineBuilder::from_source(source).wrap(displacement, displacement),
                    "add wrap operation",
                );
                let pipeline = must(builder.build(), "build pipeline");
                let mut sink = MemorySink::for_pipeline(&pipeline);
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

criterion_group!(benches, bench_wrap);
criterion_main!(benches);
