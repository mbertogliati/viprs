use criterion::{BenchmarkId, Criterion, black_box, criterion_group, criterion_main};
use viprs::{
    BandFormatId, F32, OperationBridge, PipelineBuilder, U8,
    adapters::{
        scheduler::rayon_scheduler::RayonScheduler, sinks::memory::MemorySink,
        sources::memory::MemorySource,
    },
    domain::{
        ops::{arithmetic::Linear, histogram::HistPercentOp},
        reducers::histogram::HistFindReducer,
    },
    ports::scheduler::{ReducingScheduler, TileScheduler},
};

fn make_pixels(size: u32) -> Vec<u8> {
    let pixel_count = size as usize * size as usize;
    (0..pixel_count)
        .map(|idx| (((idx * 23) + (idx / size as usize) * 3) % 256) as u8)
        .collect()
}

fn cumulative_bins(size: u32, pixels: &[u8], scheduler: &RayonScheduler) -> Vec<f32> {
    let source = MemorySource::<U8>::new(size, size, 1, pixels.to_vec()).unwrap();
    let pipeline = PipelineBuilder::from_source(source)
        .then(Box::new(OperationBridge::new_pixel_local(
            Linear::<U8>::new(1, 0).unwrap(),
            1,
        )))
        .unwrap()
        .build()
        .unwrap();
    let sink = MemorySink::for_pipeline(&pipeline);
    let histogram = scheduler
        .run_with_reducer::<U8, HistFindReducer>(
            &pipeline,
            &sink,
            &HistFindReducer::new(0, 256, BandFormatId::U8),
        )
        .unwrap();

    let mut running = 0f32;
    histogram
        .bins
        .into_iter()
        .map(|count| {
            running += count as f32;
            running
        })
        .collect()
}

fn bench_hist_percent(c: &mut Criterion) {
    let mut group = c.benchmark_group("hist_percent");
    let scheduler = RayonScheduler::new(RayonScheduler::default_threads()).unwrap();

    for &size in &[512u32, 2048, 8192] {
        let pixels = make_pixels(size);

        group.bench_with_input(BenchmarkId::from_parameter(size), &size, |b, &size| {
            b.iter(|| {
                let cumulative = cumulative_bins(size, &pixels, &scheduler);
                let source = MemorySource::<F32>::new(256, 1, 1, cumulative).unwrap();
                let pipeline = PipelineBuilder::from_source(source)
                    .then(Box::new(OperationBridge::new(
                        HistPercentOp::<F32>::new(0.5).unwrap(),
                        1,
                    )))
                    .unwrap()
                    .build()
                    .unwrap();
                let mut sink = MemorySink::for_pipeline(&pipeline);
                scheduler.run(&pipeline, &mut sink).unwrap();
                black_box(sink.into_buffer());
            });
        });
    }

    group.finish();
}

criterion_group!(benches, bench_hist_percent);
criterion_main!(benches);
