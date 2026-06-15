use criterion::{BenchmarkId, Criterion, black_box, criterion_group, criterion_main};
use viprs::{
    OperationBridge,
    adapters::{
        pipeline::PipelineBuilder, scheduler::rayon_scheduler::RayonScheduler,
        sinks::memory::MemorySink, sources::memory::MemorySource,
    },
    domain::{format::U8, ops::morphology::RankOp},
    ports::scheduler::TileScheduler,
};

fn bench_rank(c: &mut Criterion) {
    let mut group = c.benchmark_group("rank_u8_15x15");

    for &size in &[512u32, 2048, 8192] {
        let pixel_count = size as usize * size as usize;
        let pixels = vec![127u8; pixel_count];

        group.bench_with_input(BenchmarkId::from_parameter(size), &size, |b, &size| {
            b.iter(|| {
                let source = MemorySource::<U8>::new(size, size, 1, pixels.clone()).unwrap();
                let pipeline = PipelineBuilder::from_source(source)
                    .then(Box::new(OperationBridge::new(
                        RankOp::<U8>::new(15, 15, 112).unwrap(),
                        1,
                    )))
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

criterion_group!(benches, bench_rank);
criterion_main!(benches);
