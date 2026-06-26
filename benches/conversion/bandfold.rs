#![allow(missing_docs)]
use criterion::{BenchmarkId, Criterion, black_box, criterion_group, criterion_main};
use viprs::{
    adapters::{
        pipeline::internal::PipelineBuilder, scheduler::rayon_scheduler::RayonScheduler,
        sinks::memory::MemorySink, sources::memory::MemorySource,
    },
    domain::{format::U8, ops::conversion::bandfold::BandfoldBridge},
    ports::scheduler::TileScheduler,
};

fn bench_bandfold(c: &mut Criterion) {
    let mut group = c.benchmark_group("bandfold_u8_rg");

    for &size in &[512u32, 2048, 8192] {
        let width = size * 2;
        let pixels = vec![128u8; width as usize * size as usize * 2];

        group.bench_with_input(BenchmarkId::from_parameter(size), &size, |b, &size| {
            b.iter(|| {
                let source = MemorySource::<U8>::new(width, size, 2, pixels.clone()).unwrap();
                let pipeline = PipelineBuilder::from_source(source)
                    .then(Box::new(BandfoldBridge::<U8>::new(2, width, 2)))
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

criterion_group!(benches, bench_bandfold);
criterion_main!(benches);
