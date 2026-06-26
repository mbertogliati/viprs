#![allow(missing_docs)]
use criterion::{BenchmarkId, Criterion, black_box, criterion_group, criterion_main};
use viprs::{
    adapters::{
        pipeline::internal::PipelinePlan, scheduler::rayon_scheduler::RayonScheduler,
        sinks::memory::MemorySink, sources::memory::MemorySource,
    },
    domain::{format::U8, op::OperationBridge, ops::arithmetic::Linear, reducers::ProfileOp},
    ports::scheduler::ReducingScheduler,
};

fn make_pixels(size: u32) -> Vec<u8> {
    let sample_count = size as usize * size as usize;
    (0..sample_count)
        .map(|idx| if idx % 17 == 0 { 255 } else { 0 })
        .collect()
}

fn bench_profile(c: &mut Criterion) {
    let mut group = c.benchmark_group("profile_reducer");
    let scheduler = RayonScheduler::new(RayonScheduler::default_threads()).unwrap();

    for &size in &[512u32, 2048, 8192] {
        let pixels = make_pixels(size);
        group.bench_with_input(BenchmarkId::from_parameter(size), &size, |b, &size| {
            b.iter(|| {
                let source = MemorySource::<U8>::new(size, size, 1, pixels.clone()).unwrap();
                let pipeline = PipelinePlan::from_source(source)
                    .append_dyn_op(Box::new(OperationBridge::new_pixel_local(
                        Linear::<U8>::new(1, 0).unwrap(),
                        1,
                    )))
                    .unwrap()
                    .compile()
                    .unwrap();
                let sink = MemorySink::for_pipeline(&pipeline).unwrap();
                let profile = scheduler
                    .run_with_reducer::<U8, ProfileOp>(
                        &pipeline,
                        &sink,
                        &ProfileOp::new(size, size, 1),
                    )
                    .unwrap();
                black_box(profile);
                black_box(sink.into_buffer());
            });
        });
    }

    group.finish();
}

criterion_group!(benches, bench_profile);
criterion_main!(benches);
