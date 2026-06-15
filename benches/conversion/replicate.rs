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

fn bench_replicate(c: &mut Criterion) {
    let mut group = c.benchmark_group("conversion_replicate_2x2_u8");

    for &dst_size in &[512u32, 2048, 8192] {
        let src_size = dst_size / 2;
        let pixels = vec![128u8; src_size as usize * src_size as usize];

        group.bench_with_input(
            BenchmarkId::from_parameter(dst_size),
            &src_size,
            |b, &src_size| {
                b.iter(|| {
                    let source = must(
                        MemorySource::<U8>::new(src_size, src_size, 1, pixels.clone()),
                        "create memory source",
                    );
                    let builder = must(
                        PipelineBuilder::from_source(source).replicate(2, 2),
                        "add replicate operation",
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
            },
        );
    }

    group.finish();
}

criterion_group!(benches, bench_replicate);
criterion_main!(benches);
