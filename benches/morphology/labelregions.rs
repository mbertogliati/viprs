#![allow(missing_docs)]
use criterion::{BenchmarkId, Criterion, black_box, criterion_group, criterion_main};
use viprs::{
    adapters::{
        pipeline::PipelineBuilder, scheduler::rayon_scheduler::RayonScheduler,
        sinks::memory::MemorySink, sources::memory::MemorySource,
    },
    domain::{format::U8, ops::morphology::LabelRegionsOp},
    pipeline::OperationBridge,
    ports::scheduler::TileScheduler,
};

fn bench_labelregions(c: &mut Criterion) {
    let mut group = c.benchmark_group("labelregions_u8");

    for &size in &[512u32, 2048, 8192] {
        let pixels: Vec<u8> = (0..size as usize * size as usize)
            .map(|idx| u8::from((idx / 32 + (idx % size as usize) / 64).is_multiple_of(2)))
            .collect();

        group.bench_with_input(BenchmarkId::from_parameter(size), &size, |b, &size| {
            b.iter(|| {
                let source = MemorySource::<U8>::new(size, size, 1, pixels.clone()).unwrap();
                let pipeline = PipelineBuilder::from_source(source)
                    .then(Box::new(OperationBridge::new(LabelRegionsOp::default(), 1)))
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

criterion_group!(benches, bench_labelregions);
criterion_main!(benches);
