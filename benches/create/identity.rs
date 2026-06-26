#![allow(missing_docs)]
use criterion::{BenchmarkId, Criterion, black_box, criterion_group, criterion_main};
use viprs::{
    adapters::{
        pipeline::internal::PipelinePlan, scheduler::rayon_scheduler::RayonScheduler,
        sinks::memory::MemorySink, sources::memory::MemorySource,
    },
    domain::{
        format::{U8, U16},
        op::OperationBridge,
        ops::create::IdentityOp,
    },
    ports::scheduler::TileScheduler,
};

fn bench_identity(c: &mut Criterion) {
    let mut group = c.benchmark_group("create_identity");
    let scheduler = RayonScheduler::new(RayonScheduler::default_threads()).unwrap();

    group.bench_with_input(BenchmarkId::new("u8", 256u32), &256u32, |b, &size| {
        let pixels = vec![0u8; size as usize];
        b.iter(|| {
            let source = MemorySource::<U8>::new(size, 1, 1, pixels.clone()).unwrap();
            let pipeline = PipelinePlan::from_source(source)
                .append_dyn_op(Box::new(OperationBridge::new_pixel_local(
                    IdentityOp::<U8>::new(false),
                    1,
                )))
                .unwrap()
                .compile()
                .unwrap();
            let mut sink = MemorySink::for_pipeline(&pipeline).unwrap();
            scheduler.run(&pipeline, &mut sink).unwrap();
            black_box(sink.into_buffer())
        });
    });

    group.bench_with_input(
        BenchmarkId::new("u16", 65_536u32),
        &65_536u32,
        |b, &size| {
            let pixels = vec![0u16; size as usize];
            b.iter(|| {
                let source = MemorySource::<U16>::new(size, 1, 1, pixels.clone()).unwrap();
                let pipeline = PipelinePlan::from_source(source)
                    .append_dyn_op(Box::new(OperationBridge::new_pixel_local(
                        IdentityOp::<U16>::new(true),
                        1,
                    )))
                    .unwrap()
                    .compile()
                    .unwrap();
                let mut sink = MemorySink::for_pipeline(&pipeline).unwrap();
                scheduler.run(&pipeline, &mut sink).unwrap();
                black_box(sink.into_buffer())
            });
        },
    );

    group.finish();
}

criterion_group!(benches, bench_identity);
criterion_main!(benches);
