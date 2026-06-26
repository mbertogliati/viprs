#![allow(missing_docs)]
use criterion::{BenchmarkId, Criterion, black_box, criterion_group, criterion_main};
use viprs::{
    adapters::{
        pipeline::internal::PipelinePlan, scheduler::rayon_scheduler::RayonScheduler,
        sinks::memory::MemorySink, sources::memory::MemorySource,
    },
    domain::{format::F32, op::OperationBridge, ops::morphology::Median},
    ports::scheduler::TileScheduler,
};

fn bench_median_blur(c: &mut Criterion) {
    let mut group = c.benchmark_group("median_blur_f32_3x3");

    for &size in &[512u32, 2048, 8192] {
        let pixels = vec![1.0f32; size as usize * size as usize];

        group.bench_with_input(BenchmarkId::from_parameter(size), &size, |b, &size| {
            b.iter(|| {
                let source = MemorySource::<F32>::new(size, size, 1, pixels.clone()).unwrap();
                let op = Median::<F32>::new(3, 3).unwrap();
                let pipeline = PipelinePlan::from_source(source)
                    .append_dyn_op(Box::new(OperationBridge::new(op, 1)))
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

criterion_group!(benches, bench_median_blur);
criterion_main!(benches);
