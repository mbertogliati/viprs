#![allow(missing_docs)]
use criterion::{BenchmarkId, Criterion, black_box, criterion_group, criterion_main};
use viprs::{
    adapters::{
        pipeline::ImagePipeline, scheduler::rayon_scheduler::RayonScheduler,
        sinks::memory::MemorySink, sources::memory::MemorySource,
    },
    domain::{format::U8, op::OperationBridge, ops::arithmetic::Linear, reducers::ProjectOp},
    ports::scheduler::ReducingScheduler,
};

fn make_pixels(size: u32) -> Vec<u8> {
    let sample_count = size as usize * size as usize;
    (0..sample_count)
        .map(|idx| ((idx * 13) % 251) as u8)
        .collect()
}

fn bench_project(c: &mut Criterion) {
    let mut group = c.benchmark_group("project_reducer");
    let scheduler = RayonScheduler::new(RayonScheduler::default_threads()).unwrap();

    for &size in &[512u32, 2048, 8192] {
        let pixels = make_pixels(size);
        group.bench_with_input(BenchmarkId::from_parameter(size), &size, |b, &size| {
            b.iter(|| {
                let source = MemorySource::<U8>::new(size, size, 1, pixels.clone()).unwrap();
                let pipeline = ImagePipeline::from_source(source)
                    .then(Box::new(OperationBridge::new_pixel_local(
                        Linear::<U8>::new(1, 0).unwrap(),
                        1,
                    )))
                    .unwrap()
                    .build()
                    .unwrap();
                let sink = MemorySink::for_pipeline(&pipeline).unwrap();
                let project = scheduler
                    .run_with_reducer::<U8, ProjectOp>(
                        &pipeline,
                        &sink,
                        &ProjectOp::new(size, size, 1),
                    )
                    .unwrap();
                black_box(project);
                black_box(sink.into_buffer());
            });
        });
    }

    group.finish();
}

criterion_group!(benches, bench_project);
criterion_main!(benches);
