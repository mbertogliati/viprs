#![allow(missing_docs)]
use criterion::{BenchmarkId, Criterion, black_box, criterion_group, criterion_main};
use viprs::{
    adapters::{
        pipeline::internal::PipelineBuilder, scheduler::rayon_scheduler::RayonScheduler,
        sinks::memory::MemorySink, sources::memory::MemorySource,
    },
    domain::{
        format::U8,
        op::OperationBridge,
        ops::arithmetic::{Linear, StatsOp},
    },
    ports::scheduler::ReducingScheduler,
};

fn make_pixels(size: u32) -> Vec<u8> {
    let sample_count = size as usize * size as usize * 3;
    (0..sample_count)
        .map(|idx| ((idx * 17 + idx / 3) % 256) as u8)
        .collect()
}

fn bench_stats(c: &mut Criterion) {
    let mut group = c.benchmark_group("stats_reducer");
    let scheduler = RayonScheduler::new(RayonScheduler::default_threads()).unwrap();

    for &size in &[512u32, 2048, 8192] {
        let pixels = make_pixels(size);
        group.bench_with_input(BenchmarkId::from_parameter(size), &size, |b, &size| {
            b.iter(|| {
                let source = MemorySource::<U8>::new(size, size, 3, pixels.clone()).unwrap();
                let pipeline = PipelineBuilder::from_source(source)
                    .then(Box::new(OperationBridge::new_pixel_local(
                        Linear::<U8>::new(1, 0).unwrap(),
                        3,
                    )))
                    .unwrap()
                    .build()
                    .unwrap();
                let sink = MemorySink::for_pipeline(&pipeline).unwrap();
                let stats = scheduler
                    .run_with_reducer::<U8, StatsOp>(&pipeline, &sink, &StatsOp::new(3))
                    .unwrap();
                black_box(stats);
                black_box(sink.into_buffer());
            });
        });
    }

    group.finish();
}

criterion_group!(benches, bench_stats);
criterion_main!(benches);
