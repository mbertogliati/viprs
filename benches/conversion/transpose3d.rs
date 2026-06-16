#![allow(missing_docs)]
use criterion::{BenchmarkId, Criterion, black_box, criterion_group, criterion_main};
use viprs::{
    pipeline::PipelineBuilder,
    adapters::{
        scheduler::rayon_scheduler::RayonScheduler, sinks::memory::MemorySink,
        sources::memory::MemorySource,
    },
    domain::{format::U8, op::OperationBridge, ops::conversion::Transpose3dOp},
    ports::scheduler::TileScheduler,
};

fn must<T, E: std::fmt::Display>(result: Result<T, E>, context: &str) -> T {
    match result {
        Ok(value) => value,
        Err(error) => panic!("{context}: {error}"),
    }
}

fn bench_transpose3d(c: &mut Criterion) {
    let mut group = c.benchmark_group("transpose3d_u8");

    for &size in &[512u32, 2048, 8192] {
        let page_height = size / 8;
        let pixels = vec![127u8; (size * size) as usize];

        group.bench_with_input(BenchmarkId::from_parameter(size), &size, |b, &size| {
            b.iter(|| {
                let source = must(
                    MemorySource::<U8>::new(size, size, 1, pixels.clone()),
                    "create memory source",
                );
                let pipeline = must(
                    PipelineBuilder::from_source(source).then(Box::new(OperationBridge::new(
                        Transpose3dOp::<U8>::new(size, page_height),
                        1,
                    ))),
                    "add transpose3d operation",
                )
                .build()
                .unwrap();
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

criterion_group!(benches, bench_transpose3d);
criterion_main!(benches);
