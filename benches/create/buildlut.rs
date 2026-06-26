#![allow(missing_docs)]
use criterion::{BenchmarkId, Criterion, black_box, criterion_group, criterion_main};
use viprs::{
    adapters::{
        pipeline::internal::PipelineBuilder, scheduler::rayon_scheduler::RayonScheduler,
        sinks::memory::MemorySink, sources::memory::MemorySource,
    },
    domain::{format::F32, op::OperationBridge, ops::create::BuildlutOp},
    ports::scheduler::TileScheduler,
};

fn bench_buildlut(c: &mut Criterion) {
    let mut group = c.benchmark_group("create_buildlut_f32");
    let scheduler = RayonScheduler::new(RayonScheduler::default_threads()).unwrap();

    for &size in &[512u32, 2048, 8192] {
        group.bench_with_input(BenchmarkId::from_parameter(size), &size, |b, &size| {
            let pixels = vec![0.0f32; size as usize];
            let points = vec![
                (0.0, 0.0),
                (f64::from(size) * 0.25, 32.0),
                (f64::from(size) * 0.5, 160.0),
                (f64::from(size) - 1.0, 255.0),
            ];
            b.iter(|| {
                let source = MemorySource::<F32>::new(size, 1, 1, pixels.clone()).unwrap();
                let pipeline = PipelineBuilder::from_source(source)
                    .then(Box::new(OperationBridge::new_pixel_local(
                        BuildlutOp::<F32>::new(points.clone(), size as usize).unwrap(),
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

criterion_group!(benches, bench_buildlut);
criterion_main!(benches);
