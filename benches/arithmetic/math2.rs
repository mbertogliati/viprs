#![allow(missing_docs)]
use criterion::{BenchmarkId, Criterion, black_box, criterion_group, criterion_main};
use viprs::domain::ops::arithmetic::math2::{Math2, Math2Mode};
use viprs::{
    adapters::{
        pipeline::internal::PipelinePlan, scheduler::rayon_scheduler::RayonScheduler,
        sinks::memory::MemorySink, sources::memory::MemorySource,
    },
    domain::{format::F32, op::OperationBridge},
    ports::scheduler::TileScheduler,
};

fn bench_math2(c: &mut Criterion) {
    let mut group = c.benchmark_group("math2_pow_f32");

    for &size in &[512u32, 2048, 8192] {
        let pixel_count = (size as usize) * (size as usize);
        let pixels = vec![0.75f32; pixel_count];
        let rhs = vec![2.0f32; pixel_count];

        group.bench_with_input(BenchmarkId::from_parameter(size), &size, |b, &size| {
            b.iter(|| {
                let source = MemorySource::<F32>::new(size, size, 1, pixels.clone()).unwrap();
                let op = Box::new(OperationBridge::new(
                    Math2::<F32>::new(rhs.clone(), size, 1, Math2Mode::Pow),
                    1u32,
                ));
                let pipeline = PipelinePlan::from_source(source)
                    .append_dyn_op(op)
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

criterion_group!(benches, bench_math2);
criterion_main!(benches);
