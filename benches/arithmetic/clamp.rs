#![allow(missing_docs)]
/// Benchmark: ClampOp<U8> — clamp every sample to `[16, 240]`.
///
/// Measures the full pipeline path: MemorySource → ClampOp → MemorySink via RayonScheduler.
use criterion::{BenchmarkId, Criterion, black_box, criterion_group, criterion_main};
use viprs::domain::ops::arithmetic::clamp::ClampOp;
use viprs::{
    adapters::{
        pipeline::internal::PipelineBuilder, scheduler::rayon_scheduler::RayonScheduler,
        sinks::memory::MemorySink, sources::memory::MemorySource,
    },
    domain::format::U8,
    domain::op::OperationBridge,
    ports::scheduler::TileScheduler,
};

fn bench_clamp(c: &mut Criterion) {
    let mut group = c.benchmark_group("clamp_u8");

    for &size in &[512u32, 2048, 8192] {
        let pixel_count = (size as usize) * (size as usize);
        let pixels = vec![128u8; pixel_count];

        group.bench_with_input(BenchmarkId::from_parameter(size), &size, |b, &size| {
            b.iter(|| {
                let source = MemorySource::<U8>::new(size, size, 1, pixels.clone()).unwrap();
                let clamp = ClampOp::<U8>::new(16.0, 240.0);
                let pipeline = PipelineBuilder::from_source(source)
                    .then(Box::new(OperationBridge::new_pixel_local(clamp, 1)))
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

criterion_group!(benches, bench_clamp);
criterion_main!(benches);
