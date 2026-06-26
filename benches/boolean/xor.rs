#![allow(missing_docs)]
/// Benchmark: Xor<U8> — bitwise XOR of each pixel sample with a constant mask.
///
/// Measures the full pipeline path: MemorySource → Xor → MemorySink via RayonScheduler.
/// The mask 0xAA is representative: alternating bits ensure no sample is a trivial no-op.
use criterion::{BenchmarkId, Criterion, black_box, criterion_group, criterion_main};
use viprs::domain::ops::boolean::xor::Xor;
use viprs::{
    adapters::{
        pipeline::internal::PipelineBuilder, scheduler::rayon_scheduler::RayonScheduler,
        sinks::memory::MemorySink, sources::memory::MemorySource,
    },
    domain::format::U8,
    domain::op::OperationBridge,
    ports::scheduler::TileScheduler,
};

fn bench_xor(c: &mut Criterion) {
    let mut group = c.benchmark_group("xor_u8");

    for &size in &[512u32, 2048, 8192] {
        let pixels = vec![255u8; size as usize * size as usize];

        group.bench_with_input(BenchmarkId::from_parameter(size), &size, |b, &size| {
            b.iter(|| {
                let source = MemorySource::<U8>::new(size, size, 1, pixels.clone()).unwrap();
                let op = Box::new(OperationBridge::new(Xor::<U8>::new(0xAAu8), 1u32));
                let pipeline = PipelineBuilder::from_source(source)
                    .then(op)
                    .unwrap()
                    .build()
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

criterion_group!(benches, bench_xor);
criterion_main!(benches);
