/// Benchmark: And<U8> — bitwise AND of each pixel sample with a constant mask.
///
/// Measures the full pipeline path: MemorySource → And → MemorySink via RayonScheduler.
/// The mask 0xF0 is representative: it forces work on every sample without being a no-op.
use criterion::{BenchmarkId, Criterion, black_box, criterion_group, criterion_main};
use viprs::domain::ops::boolean::and::And;
use viprs::{
    adapters::{
        pipeline::PipelineBuilder, scheduler::rayon_scheduler::RayonScheduler,
        sinks::memory::MemorySink, sources::memory::MemorySource,
    },
    domain::format::U8,
    domain::op::OperationBridge,
    ports::scheduler::TileScheduler,
};

fn bench_and(c: &mut Criterion) {
    let mut group = c.benchmark_group("and_u8");

    for &size in &[512u32, 2048, 8192] {
        let pixels = vec![255u8; size as usize * size as usize];

        group.bench_with_input(BenchmarkId::from_parameter(size), &size, |b, &size| {
            b.iter(|| {
                let source = MemorySource::<U8>::new(size, size, 1, pixels.clone()).unwrap();
                let op = Box::new(OperationBridge::new(And::<U8>::new(0xF0u8), 1u32));
                let pipeline = PipelineBuilder::from_source(source)
                    .then(op)
                    .unwrap()
                    .build()
                    .unwrap();
                let mut sink = MemorySink::for_pipeline(&pipeline);
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

criterion_group!(benches, bench_and);
criterion_main!(benches);
