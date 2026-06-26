#![allow(missing_docs)]
use criterion::{BenchmarkId, Criterion, black_box, criterion_group, criterion_main};
use viprs::{
    adapters::{
        pipeline::ImagePipeline, scheduler::rayon_scheduler::RayonScheduler,
        sinks::memory::MemorySink, sources::memory::MemorySource,
    },
    domain::{format::F32, op::OperationBridge, ops::convolution::EdgeOp},
    ports::scheduler::TileScheduler,
};

fn bench_edge(c: &mut Criterion) {
    let mut group = c.benchmark_group("edge_f32_sobel");

    for &size in &[512u32, 2048, 8192] {
        let pixels = vec![1.0f32; size as usize * size as usize];

        group.bench_with_input(BenchmarkId::from_parameter(size), &size, |b, &size| {
            b.iter(|| {
                let source = MemorySource::<F32>::new(size, size, 1, pixels.clone()).unwrap();
                let pipeline = ImagePipeline::from_source(source)
                    .then(Box::new(OperationBridge::new(EdgeOp::<F32>::sobel(), 1)))
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

criterion_group!(benches, bench_edge);
criterion_main!(benches);
