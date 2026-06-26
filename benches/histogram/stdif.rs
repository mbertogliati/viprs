#![allow(missing_docs)]
use criterion::{BenchmarkId, Criterion, black_box, criterion_group, criterion_main};
use viprs::{
    adapters::{
        pipeline::internal::PipelinePlan, scheduler::rayon_scheduler::RayonScheduler,
        sinks::memory::MemorySink, sources::memory::MemorySource,
    },
    domain::{format::U8, op::OperationBridge, ops::histogram::StdifOp},
    ports::scheduler::TileScheduler,
};

fn make_pixels(size: u32) -> Vec<u8> {
    let pixel_count = (size as usize) * (size as usize);
    (0..pixel_count)
        .map(|idx| (((idx * 19) + (idx / size as usize) * 23) % 256) as u8)
        .collect()
}

fn bench_stdif(c: &mut Criterion) {
    let mut group = c.benchmark_group("stdif");

    for &size in &[512u32, 2048, 8192] {
        let scheduler = RayonScheduler::new(RayonScheduler::default_threads()).unwrap();
        let pixels = make_pixels(size);
        group.bench_with_input(BenchmarkId::from_parameter(size), &size, |b, &size| {
            b.iter(|| {
                let source = MemorySource::<U8>::new(size, size, 1, pixels.clone()).unwrap();
                let pipeline = PipelinePlan::from_source(source)
                    .append_dyn_op(Box::new(OperationBridge::new(
                        StdifOp::<U8>::new(11, 11, 0.5, 128.0, 0.5, 50.0).unwrap(),
                        1,
                    )))
                    .unwrap()
                    .compile()
                    .unwrap();
                let mut sink = MemorySink::for_pipeline(&pipeline).unwrap();
                scheduler.run(&pipeline, &mut sink).unwrap();
                black_box(sink.into_buffer());
            });
        });
    }

    group.finish();
}

criterion_group!(benches, bench_stdif);
criterion_main!(benches);
