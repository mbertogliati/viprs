use std::sync::Arc;

use criterion::{BenchmarkId, Criterion, black_box, criterion_group, criterion_main};
use viprs::{
    adapters::{
        pipeline::PipelineBuilder, scheduler::rayon_scheduler::RayonScheduler,
        sinks::memory::MemorySink, sources::memory::MemorySource,
    },
    domain::{
        format::U8,
        op::OperationBridge,
        ops::{arithmetic::Linear, histogram::HistFindIndexedOp},
    },
    ports::scheduler::ReducingScheduler,
};

fn make_pixels(size: u32) -> Vec<u8> {
    let sample_count = size as usize * size as usize * 3;
    (0..sample_count)
        .map(|idx| ((idx * 31 + idx / 7) % 256) as u8)
        .collect()
}

fn make_index_pixels(size: u32) -> Vec<u8> {
    let sample_count = size as usize * size as usize;
    (0..sample_count)
        .map(|idx| ((idx * 13 + idx / 5) % 256) as u8)
        .collect()
}

fn bench_hist_find_indexed(c: &mut Criterion) {
    let mut group = c.benchmark_group("hist_find_indexed");
    let scheduler = RayonScheduler::new(RayonScheduler::default_threads()).unwrap();

    for &size in &[512u32, 2048, 8192] {
        let pixels = make_pixels(size);
        let index_pixels = make_index_pixels(size);
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
                let sink = MemorySink::for_pipeline(&pipeline);
                let index =
                    Arc::new(MemorySource::<U8>::new(size, size, 1, index_pixels.clone()).unwrap());
                let reducer = HistFindIndexedOp::new(size, size, 3, u8::MAX as u32, index).unwrap();
                let hist = scheduler
                    .run_with_reducer::<U8, HistFindIndexedOp>(&pipeline, &sink, &reducer)
                    .unwrap();
                black_box(hist);
                black_box(sink.into_buffer());
            });
        });
    }

    group.finish();
}

criterion_group!(benches, bench_hist_find_indexed);
criterion_main!(benches);
