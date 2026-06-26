#![allow(missing_docs)]
use criterion::{BenchmarkId, Criterion, black_box, criterion_group, criterion_main};
use viprs::{
    adapters::{
        pipeline::internal::PipelinePlan, scheduler::rayon_scheduler::RayonScheduler,
        sinks::memory::MemorySink, sources::memory::MemorySource,
    },
    domain::{format::U8, ops::conversion::embed::ExtendMode},
    ports::scheduler::TileScheduler,
};

fn must<T, E: std::fmt::Display>(result: Result<T, E>, context: &str) -> T {
    match result {
        Ok(value) => value,
        Err(error) => panic!("{context}: {error}"),
    }
}

fn bench_embed(c: &mut Criterion) {
    let mut group = c.benchmark_group("conversion_embed_black_u8");

    for &src_size in &[512u32, 2048, 8192] {
        let dst_size = src_size * 2;
        let offset = src_size / 2;
        let pixels = vec![128u8; src_size as usize * src_size as usize * 3];

        group.bench_with_input(BenchmarkId::from_parameter(src_size), &src_size, |b, _| {
            b.iter(|| {
                let source = must(
                    MemorySource::<U8>::new(src_size, src_size, 3, pixels.clone()),
                    "create memory source",
                );
                let builder = must(
                    PipelinePlan::from_source(source).plan_embed(
                        dst_size,
                        dst_size,
                        offset,
                        offset,
                        src_size,
                        src_size,
                        ExtendMode::Black,
                    ),
                    "add embed operation",
                );
                let pipeline = must(builder.compile(), "build pipeline");
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

criterion_group!(benches, bench_embed);
criterion_main!(benches);
