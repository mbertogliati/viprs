#![allow(missing_docs)]
/// Benchmark: RShift<U8> — right-shift each pixel sample by a constant number of bits.
///
/// Measures the full pipeline path: MemorySource → RShift → MemorySink via RayonScheduler.
/// A shift of 1 is representative: it exercises the shift path on every sample.
use criterion::{BenchmarkId, Criterion, black_box, criterion_group, criterion_main};
use viprs::domain::ops::boolean::rshift::RShift;
use viprs::{
    adapters::{
        pipeline::ImagePipeline, scheduler::rayon_scheduler::RayonScheduler,
        sinks::memory::MemorySink, sources::memory::MemorySource,
    },
    domain::format::U8,
    domain::op::OperationBridge,
    ports::scheduler::TileScheduler,
};

fn bench_rshift(c: &mut Criterion) {
    let mut group = c.benchmark_group("rshift_u8");

    for &size in &[512u32, 2048, 8192] {
        let pixels = vec![255u8; size as usize * size as usize];

        group.bench_with_input(BenchmarkId::from_parameter(size), &size, |b, &size| {
            b.iter(|| {
                let source = MemorySource::<U8>::new(size, size, 1, pixels.clone()).unwrap();
                let op = Box::new(OperationBridge::new(RShift::<U8>::new(1u32), 1u32));
                let pipeline = ImagePipeline::from_source(source)
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

criterion_group!(benches, bench_rshift);
criterion_main!(benches);
