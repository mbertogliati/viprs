#![allow(missing_docs)]
use criterion::{BenchmarkId, Criterion, black_box, criterion_group, criterion_main};
use viprs::domain::ops::conversion::copy::CopyOp;
use viprs::{
    adapters::{
        pipeline::internal::PipelinePlan, scheduler::rayon_scheduler::RayonScheduler,
        sinks::memory::MemorySink, sources::memory::MemorySource,
    },
    domain::format::U8,
    domain::op::OperationBridge,
    ports::scheduler::TileScheduler,
};

fn bench_copy(c: &mut Criterion) {
    let mut group = c.benchmark_group("copy_u8_rgba");

    for &size in &[512u32, 2048, 8192] {
        let pixels = vec![128u8; size as usize * size as usize * 4];

        group.bench_with_input(BenchmarkId::from_parameter(size), &size, |b, &size| {
            b.iter(|| {
                let source = MemorySource::<U8>::new(size, size, 4, pixels.clone()).unwrap();
                let op = CopyOp::<U8>::default();
                let dyn_op = Box::new(OperationBridge::new_pixel_local(op, 4));
                let pipeline = PipelinePlan::from_source(source)
                    .append_dyn_op(dyn_op)
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

criterion_group!(benches, bench_copy);
criterion_main!(benches);
