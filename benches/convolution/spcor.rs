#![allow(missing_docs)]
use criterion::{BenchmarkId, Criterion, black_box, criterion_group, criterion_main};
use viprs::{
    adapters::{
        scheduler::rayon_scheduler::RayonScheduler, sinks::memory::MemorySink,
        sources::memory::MemorySource,
    },
    domain::ops::convolution::SpcorOp,
    domain::{format::F32, image::Image},
    pipeline::OperationBridge,
    ports::scheduler::TileScheduler,
};
use viprs_runtime::pipeline::internal::PipelineBuilder;

fn make_pixels(size: u32) -> Vec<f32> {
    let pixel_count = size as usize * size as usize;
    (0..pixel_count)
        .map(|idx| ((idx * 13 + idx / size as usize * 7) % 251) as f32 / 255.0)
        .collect()
}

fn reference_patch() -> Image<F32> {
    Image::from_buffer(
        5,
        5,
        1,
        vec![
            0.0, 0.1, 0.3, 0.1, 0.0, 0.1, 0.4, 0.8, 0.4, 0.1, 0.2, 0.7, 1.0, 0.7, 0.2, 0.1, 0.4,
            0.8, 0.4, 0.1, 0.0, 0.1, 0.3, 0.1, 0.0,
        ],
    )
    .unwrap()
}

fn bench_spcor(c: &mut Criterion) {
    let mut group = c.benchmark_group("spcor_f32");
    let reference = reference_patch();

    for &size in &[512u32, 2048, 8192] {
        let pixels = make_pixels(size);

        group.bench_with_input(BenchmarkId::from_parameter(size), &size, |b, &size| {
            b.iter(|| {
                let source = MemorySource::<F32>::new(size, size, 1, pixels.clone()).unwrap();
                let op = SpcorOp::<F32>::new(reference.clone()).unwrap();
                let pipeline = PipelineBuilder::from_source(source)
                    .then(Box::new(OperationBridge::new(op, 1)))
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

criterion_group!(benches, bench_spcor);
criterion_main!(benches);
