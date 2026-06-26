#![allow(missing_docs)]
use criterion::{BenchmarkId, Criterion, black_box, criterion_group, criterion_main};
use viprs::{
    adapters::{
      pipeline::ImagePipeline, scheduler::rayon_scheduler::RayonScheduler,
      sinks::memory::MemorySink, sources::memory::MemorySource,
    },
    domain::{format::U8, ops::conversion::falsecolour::FalsecolourOp},
    ports::scheduler::TileScheduler,
};

fn bench_falsecolour(c: &mut Criterion) {
    let mut group = c.benchmark_group("falsecolour_u8_grey");

    for &size in &[512u32, 2048, 8192] {
        let pixels = vec![128u8; size as usize * size as usize];

        group.bench_with_input(BenchmarkId::from_parameter(size), &size, |b, &size| {
            b.iter(|| {
                let source = MemorySource::<U8>::new(size, size, 1, pixels.clone()).unwrap();
                let pipeline = ImagePipeline::from_source(source)
                    .then(Box::new(FalsecolourOp::<U8>::new().into_bridge(1)))
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

criterion_group!(benches, bench_falsecolour);
criterion_main!(benches);
