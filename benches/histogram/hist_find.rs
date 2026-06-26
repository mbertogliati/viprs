#![allow(missing_docs)]
use criterion::{BenchmarkId, Criterion, black_box, criterion_group, criterion_main};
use viprs::{
    adapters::{
      pipeline::ImagePipeline, scheduler::rayon_scheduler::RayonScheduler,
      sinks::memory::MemorySink, sources::memory::MemorySource,
    },
    domain::{
        format::U8,
        op::OperationBridge,
        ops::{arithmetic::Linear, histogram::HistFindOp},
    },
    ports::scheduler::ReducingScheduler,
};

fn make_pixels(size: u32) -> Vec<u8> {
    let sample_count = size as usize * size as usize * 3;
    (0..sample_count)
        .map(|idx| ((idx * 29 + idx / 11) % 256) as u8)
        .collect()
}

fn bench_hist_find(c: &mut Criterion) {
    let mut group = c.benchmark_group("hist_find");
    let scheduler = RayonScheduler::new(RayonScheduler::default_threads()).unwrap();

    for &size in &[512u32, 2048, 8192] {
        let pixels = make_pixels(size);
        let source = MemorySource::<U8>::new(size, size, 3, pixels).unwrap();
        let pipeline = ImagePipeline::from_source(source)
            .then(Box::new(OperationBridge::new_pixel_local(
                Linear::<U8>::new(1, 0).unwrap(),
                3,
            )))
            .unwrap()
            .build()
            .unwrap();
        let reducer = HistFindOp::for_format(3, None, u8::MAX as u32);

        group.bench_with_input(BenchmarkId::from_parameter(size), &size, |b, _| {
            b.iter(|| {
                let sink = MemorySink::for_pipeline(&pipeline).unwrap();
                let hist = scheduler
                    .run_with_reducer::<U8, HistFindOp>(&pipeline, &sink, &reducer)
                    .unwrap();
                black_box(hist);
            });
        });
    }

    group.finish();
}

criterion_group!(benches, bench_hist_find);
criterion_main!(benches);
