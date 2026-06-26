#![allow(missing_docs)]
/// Benchmark: Cos<F32> — element-wise cosine.
///
/// Measures the full pipeline path: MemorySource → Cos → MemorySink via RayonScheduler.
use criterion::{BenchmarkId, Criterion, black_box, criterion_group, criterion_main};
use viprs::domain::ops::arithmetic::cos::Cos;
use viprs::{
    adapters::{
        pipeline::internal::PipelinePlan, scheduler::rayon_scheduler::RayonScheduler,
        sinks::memory::MemorySink, sources::memory::MemorySource,
    },
    domain::format::F32,
    domain::op::OperationBridge,
    ports::scheduler::TileScheduler,
};
fn bench_cos(c: &mut Criterion) {
    let mut group = c.benchmark_group("cos_f32");
    for &size in &[512u32, 2048, 8192] {
        let pixels = vec![0.5f32; size as usize * size as usize];
        group.bench_with_input(BenchmarkId::from_parameter(size), &size, |b, &size| {
            b.iter(|| {
                let source = MemorySource::<F32>::new(size, size, 1, pixels.clone()).unwrap();
                let op = Box::new(OperationBridge::new(Cos::<F32>::new(), 1u32));
                let pipeline = PipelinePlan::from_source(source)
                    .append_dyn_op(op)
                    .unwrap()
                    .compile()
                    .unwrap();
                let mut sink = MemorySink::for_pipeline(&pipeline).unwrap();
                RayonScheduler::new(1)
                    .unwrap()
                    .run(&pipeline, &mut sink)
                    .unwrap();
                black_box(sink.into_buffer())
            });
        });
    }
    group.finish();
}
criterion_group!(benches, bench_cos);
criterion_main!(benches);
