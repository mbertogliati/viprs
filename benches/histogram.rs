#![allow(missing_docs)]
use criterion::{BenchmarkId, Criterion, black_box, criterion_group, criterion_main};
use viprs::{
    adapters::{
      pipeline::ImagePipeline, scheduler::rayon_scheduler::RayonScheduler,
      sinks::memory::MemorySink, sources::memory::MemorySource,
    },
    domain::{
        format::{BandFormatId, F32, U8},
        op::OperationBridge,
        ops::{
            arithmetic::linear::Linear,
            histogram::{HistCumOp, hist_norm::HistNormTypedOp},
        },
        reducers::histogram::HistFindReducer,
    },
    ports::scheduler::{ReducingScheduler, TileScheduler},
};

fn make_pixels(size: u32, stride: usize, row_scale: usize) -> Vec<u8> {
    let pixel_count = (size as usize) * (size as usize);
    (0..pixel_count)
        .map(|idx| (((idx * stride) + (idx / size as usize) * row_scale) % 256) as u8)
        .collect()
}

fn histogram_bins_from_source(size: u32, pixels: &[u8], scheduler: &RayonScheduler) -> Vec<f32> {
    let source = MemorySource::<U8>::new(size, size, 1, pixels.to_vec()).unwrap();
    let pipeline = ImagePipeline::from_source(source)
        .then(Box::new(OperationBridge::new_pixel_local(
            Linear::<U8>::new(1, 0).unwrap(),
            1,
        )))
        .unwrap()
        .build()
        .unwrap();
    let sink = MemorySink::for_pipeline(&pipeline).unwrap();
    let hist = scheduler
        .run_with_reducer::<U8, HistFindReducer>(
            &pipeline,
            &sink,
            &HistFindReducer::new(0, 256, BandFormatId::U8),
        )
        .unwrap();
    hist.bins.into_iter().map(|count| count as f32).collect()
}

fn bench_hist_norm(c: &mut Criterion) {
    let mut group = c.benchmark_group("histogram_hist_norm");
    let scheduler = RayonScheduler::new(RayonScheduler::default_threads()).unwrap();

    for &size in &[512u32, 2048, 8192] {
        let pixels = make_pixels(size, 13, 7);
        group.bench_with_input(BenchmarkId::new("hist_norm", size), &size, |b, &size| {
            b.iter(|| {
                let hist_bins = histogram_bins_from_source(size, &pixels, &scheduler);
                let source = MemorySource::<F32>::new(256, 1, 1, hist_bins).unwrap();
                let pipeline = ImagePipeline::from_source(source)
                    .then(Box::new(OperationBridge::new(
                        HistNormTypedOp::<F32, U8>::new(),
                        1,
                    )))
                    .unwrap()
                    .build()
                    .unwrap();
                let mut sink = MemorySink::for_pipeline(&pipeline).unwrap();
                scheduler.run(&pipeline, &mut sink).unwrap();
                black_box(sink.into_buffer());
            });
        });
    }

    group.finish();
}

fn bench_hist_cum(c: &mut Criterion) {
    let mut group = c.benchmark_group("histogram_hist_cum");
    let scheduler = RayonScheduler::new(RayonScheduler::default_threads()).unwrap();

    for &size in &[512u32, 2048, 8192] {
        let pixels = make_pixels(size, 29, 11);
        group.bench_with_input(BenchmarkId::new("hist_cum", size), &size, |b, &size| {
            b.iter(|| {
                let hist_bins = histogram_bins_from_source(size, &pixels, &scheduler);
                let source = MemorySource::<F32>::new(256, 1, 1, hist_bins).unwrap();
                let pipeline = ImagePipeline::from_source(source)
                    .then(Box::new(OperationBridge::new(
                        HistNormTypedOp::<F32, U8>::new(),
                        1,
                    )))
                    .unwrap()
                    .then(Box::new(OperationBridge::new(HistCumOp::<U8>::new(), 1)))
                    .unwrap()
                    .build()
                    .unwrap();
                let mut sink = MemorySink::for_pipeline(&pipeline).unwrap();
                scheduler.run(&pipeline, &mut sink).unwrap();
                black_box(sink.into_buffer());
            });
        });
    }

    group.finish();
}

criterion_group!(benches, bench_hist_norm, bench_hist_cum);
criterion_main!(benches);
