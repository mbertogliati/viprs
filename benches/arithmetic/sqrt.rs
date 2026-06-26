#![allow(missing_docs)]
/// Benchmark: Sqrt<F32> — element-wise square root.
///
/// Measures the full pipeline path: MemorySource → Sqrt → MemorySink via RayonScheduler.
/// Pixels are set to 4.0 (positive input, sqrt well-defined).
use criterion::{BenchmarkId, Criterion, black_box, criterion_group, criterion_main};
use viprs::domain::ops::arithmetic::sqrt::Sqrt;
use viprs::{
    adapters::{
        pipeline::internal::PipelinePlan, scheduler::rayon_scheduler::RayonScheduler,
        sinks::memory::MemorySink, sources::memory::MemorySource,
    },
    domain::format::F32,
    domain::op::OperationBridge,
    ports::scheduler::TileScheduler,
};
fn bench_sqrt(c: &mut Criterion) {
    let mut group = c.benchmark_group("sqrt_f32");
    for &size in &[512u32, 2048, 8192] {
        let pixels = vec![4.0f32; size as usize * size as usize];
        group.bench_with_input(BenchmarkId::from_parameter(size), &size, |b, &size| {
            b.iter(|| {
                let source = MemorySource::<F32>::new(size, size, 1, pixels.clone()).unwrap();
                let op = Box::new(OperationBridge::new(Sqrt::<F32>::new(), 1u32));
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
criterion_group!(benches, bench_sqrt);
criterion_main!(benches);
