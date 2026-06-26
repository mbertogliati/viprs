#![allow(missing_docs)]
/// Benchmark: LessEq<F32> — element-wise less-than-or-equal comparison against a scalar.
///
/// Measures the full pipeline path: MemorySource → LessEq → MemorySink via RayonScheduler.
use criterion::{BenchmarkId, Criterion, black_box, criterion_group, criterion_main};
use viprs::domain::ops::relational::LessEq;
use viprs::{
    adapters::{
      pipeline::ImagePipeline, scheduler::rayon_scheduler::RayonScheduler,
      sinks::memory::MemorySink, sources::memory::MemorySource,
    },
    domain::format::F32,
    domain::op::OperationBridge,
    ports::scheduler::TileScheduler,
};

fn bench_less_eq(c: &mut Criterion) {
    let mut group = c.benchmark_group("less_eq_f32");
    for &size in &[512u32, 2048, 8192] {
        let pixels = vec![0.5f32; size as usize * size as usize];
        group.bench_with_input(BenchmarkId::from_parameter(size), &size, |b, &size| {
            b.iter(|| {
                let source = MemorySource::<F32>::new(size, size, 1, pixels.clone()).unwrap();
                let op = Box::new(OperationBridge::new(LessEq::<F32>::new(0.5), 1u32));
                let pipeline = ImagePipeline::from_source(source)
                    .then(op)
                    .unwrap()
                    .build()
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

criterion_group!(benches, bench_less_eq);
criterion_main!(benches);
