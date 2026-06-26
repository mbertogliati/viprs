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
        ops::{arithmetic::linear::Linear, histogram::HistEqualOp},
        reducers::hist_equal::HistEqualReducer,
    },
    ports::scheduler::{ReducingScheduler, TileScheduler},
};

fn make_pixels(size: u32) -> Vec<u8> {
    let pixel_count = (size as usize) * (size as usize);
    (0..pixel_count)
        .map(|idx| (((idx * 17) + (idx / size as usize) * 5) % 256) as u8)
        .collect()
}

fn bench_hist_equal(c: &mut Criterion) {
    let mut group = c.benchmark_group("hist_equal");
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
                let lut = scheduler
                    .run_with_reducer::<U8, HistEqualReducer>(
                        &pipeline,
                        &sink,
                        &HistEqualReducer::new(1, 0, 256).unwrap(),
                    )
                    .unwrap();

                let source = MemorySource::<U8>::new(size, size, 1, pixels.clone()).unwrap();
                let op = HistEqualOp::<U8>::from_lut(lut).unwrap();
                let pipeline = ImagePipeline::from_source(source)
                    .then(Box::new(OperationBridge::new_pixel_local(op, 1)))
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

criterion_group!(benches, bench_hist_equal);
criterion_main!(benches);
