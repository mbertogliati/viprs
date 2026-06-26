#![allow(missing_docs)]
use criterion::{BenchmarkId, Criterion, black_box, criterion_group, criterion_main};
use viprs::{
    adapters::{
        pipeline::internal::PipelinePlan, scheduler::rayon_scheduler::RayonScheduler,
        sinks::memory::MemorySink, sources::memory::MemorySource,
    },
    domain::{format::U16, op::OperationBridge, ops::conversion::byteswap::ByteswapOp},
    ports::scheduler::TileScheduler,
};

fn bench_byteswap(c: &mut Criterion) {
    let mut group = c.benchmark_group("byteswap_u16_grey");

    for &size in &[512u32, 2048, 8192] {
        let pixels = vec![0x1234u16; size as usize * size as usize];

        group.bench_with_input(BenchmarkId::from_parameter(size), &size, |b, &size| {
            b.iter(|| {
                let source = MemorySource::<U16>::new(size, size, 1, pixels.clone()).unwrap();
                let op = ByteswapOp::<U16>::new();
                let pipeline = PipelinePlan::from_source(source)
                    .append_dyn_op(Box::new(OperationBridge::new_pixel_local(op, 1)))
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

criterion_group!(benches, bench_byteswap);
criterion_main!(benches);
