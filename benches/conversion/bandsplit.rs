/// Benchmark: BandSplit<U8> — extract one band from a 4-band RGBA image.
///
/// Measures the full pipeline path: MemorySource → BandSplit → MemorySink via RayonScheduler.
/// Band 0 (Red) is extracted; output is a 1-band image of the same dimensions.
use criterion::{BenchmarkId, Criterion, black_box, criterion_group, criterion_main};
use viprs::domain::ops::conversion::bandsplit::BandSplit;
use viprs::{
    adapters::{
        pipeline::PipelineBuilder, scheduler::rayon_scheduler::RayonScheduler,
        sinks::memory::MemorySink, sources::memory::MemorySource,
    },
    domain::format::U8,
    domain::op::OperationBridge,
    ports::scheduler::TileScheduler,
};

fn bench_bandsplit(c: &mut Criterion) {
    let mut group = c.benchmark_group("bandsplit_u8_rgba");

    for &size in &[512u32, 2048, 8192] {
        // 4-band RGBA image; extract band 0 (Red).
        let pixels = vec![128u8; size as usize * size as usize * 4];

        group.bench_with_input(BenchmarkId::from_parameter(size), &size, |b, &size| {
            b.iter(|| {
                let source = MemorySource::<U8>::new(size, size, 4, pixels.clone()).unwrap();
                let op = BandSplit::<U8>::new(0, 4);
                // Output has 1 band; OperationBridge::bands must reflect that — see B-50.
                let dyn_op = Box::new(OperationBridge::new(op, 1u32));
                let pipeline = PipelineBuilder::from_source(source)
                    .then(dyn_op)
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

criterion_group!(benches, bench_bandsplit);
criterion_main!(benches);
