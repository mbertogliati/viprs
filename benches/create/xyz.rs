use criterion::{BenchmarkId, Criterion, black_box, criterion_group, criterion_main};
use viprs::{
    adapters::{
        pipeline::PipelineBuilder, scheduler::rayon_scheduler::RayonScheduler,
        sinks::memory::MemorySink, sources::memory::MemorySource,
    },
    domain::{format::U16, op::OperationBridge, ops::create::XyzOp},
    ports::scheduler::TileScheduler,
};

fn bench_xyz(c: &mut Criterion) {
    let mut group = c.benchmark_group("create_xyz_u16");
    let scheduler = RayonScheduler::new(RayonScheduler::default_threads()).unwrap();

    for &size in &[512u32, 2048, 8192] {
        group.bench_with_input(BenchmarkId::from_parameter(size), &size, |b, &size| {
            let pixels = vec![0u16; size as usize * size as usize * 3];
            b.iter(|| {
                let source = MemorySource::<U16>::new(size, size, 3, pixels.clone()).unwrap();
                let pipeline = PipelineBuilder::from_source(source)
                    .then(Box::new(OperationBridge::new_pixel_local(
                        XyzOp::<U16>::new(size, size),
                        3,
                    )))
                    .unwrap()
                    .build()
                    .unwrap();
                let mut sink = MemorySink::for_pipeline(&pipeline);
                scheduler.run(&pipeline, &mut sink).unwrap();
                black_box(sink.into_buffer())
            });
        });
    }

    group.finish();
}

criterion_group!(benches, bench_xyz);
criterion_main!(benches);
