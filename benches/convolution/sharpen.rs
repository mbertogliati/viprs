#![allow(missing_docs)]
use criterion::{BenchmarkId, Criterion, black_box, criterion_group, criterion_main};
use viprs::{
    adapters::{
        pipeline::ImagePipeline, scheduler::rayon_scheduler::RayonScheduler,
        sinks::memory::MemorySink, sources::memory::MemorySource,
    },
    domain::{format::F32, op::OperationBridge, ops::convolution::Sharpen},
    ports::scheduler::TileScheduler,
};

fn bench_sharpen(c: &mut Criterion) {
    let mut group = c.benchmark_group("sharpen_f32");

    for &size in &[512u32, 2048, 8192] {
        let pixel_count = (size as usize) * (size as usize);
        let pixels = vec![1.0f32; pixel_count];

        group.bench_with_input(BenchmarkId::from_parameter(size), &size, |b, &size| {
            b.iter(|| {
                let source = MemorySource::<F32>::new(size, size, 1, pixels.clone()).unwrap();
                let bridge = OperationBridge::new(Sharpen::<F32>::new(1.0, 1.5), 1u32);
                let pipeline = ImagePipeline::from_source(source)
                    .then(Box::new(bridge))
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

criterion_group!(benches, bench_sharpen);
criterion_main!(benches);
