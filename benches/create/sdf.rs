#![allow(missing_docs)]
use criterion::{BenchmarkId, Criterion, black_box, criterion_group, criterion_main};
use viprs::{
    adapters::{
        pipeline::internal::PipelineBuilder, scheduler::rayon_scheduler::RayonScheduler,
        sinks::memory::MemorySink, sources::memory::MemorySource,
    },
    domain::{
        format::F32,
        op::OperationBridge,
        ops::create::{SdfOp, SdfShape},
    },
    ports::scheduler::TileScheduler,
};

fn bench_sdf(c: &mut Criterion) {
    let mut group = c.benchmark_group("create_sdf_f32");
    let scheduler = RayonScheduler::new(RayonScheduler::default_threads()).unwrap();

    for &size in &[512u32, 2048, 8192] {
        group.bench_with_input(BenchmarkId::from_parameter(size), &size, |b, &size| {
            let pixels = vec![0.0f32; size as usize * size as usize];
            b.iter(|| {
                let source = MemorySource::<F32>::new(size, size, 1, pixels.clone()).unwrap();
                let pipeline = PipelineBuilder::from_source(source)
                    .then(Box::new(OperationBridge::new_pixel_local(
                        SdfOp::new(
                            size,
                            size,
                            SdfShape::Circle {
                                center: [size as f32 * 0.5, size as f32 * 0.5],
                                radius: size as f32 * 0.25,
                            },
                        )
                        .unwrap(),
                        1,
                    )))
                    .unwrap()
                    .build()
                    .unwrap();
                let mut sink = MemorySink::for_pipeline(&pipeline).unwrap();
                scheduler.run(&pipeline, &mut sink).unwrap();
                black_box(sink.into_buffer())
            });
        });
    }

    group.finish();
}

criterion_group!(benches, bench_sdf);
criterion_main!(benches);
