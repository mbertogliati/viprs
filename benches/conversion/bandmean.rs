#![allow(missing_docs)]
use criterion::{BenchmarkId, Criterion, black_box, criterion_group, criterion_main};
use viprs::domain::ops::conversion::bandmean::BandMean;
use viprs::{
    adapters::{
      pipeline::ImagePipeline, scheduler::rayon_scheduler::RayonScheduler,
      sinks::memory::MemorySink, sources::memory::MemorySource,
    },
    domain::format::U8,
    domain::op::OperationBridge,
    ports::scheduler::TileScheduler,
};

fn bench_bandmean(c: &mut Criterion) {
    let mut group = c.benchmark_group("bandmean_u8_rgba");

    for &size in &[512u32, 2048, 8192] {
        let pixels = vec![128u8; size as usize * size as usize * 4];

        group.bench_with_input(BenchmarkId::from_parameter(size), &size, |b, &size| {
            b.iter(|| {
                let source = MemorySource::<U8>::new(size, size, 4, pixels.clone()).unwrap();
                let op = BandMean::<U8>::new(4);
                let dyn_op = Box::new(OperationBridge::new_pixel_local(op, 4));
                let pipeline = ImagePipeline::from_source(source)
                    .then(dyn_op)
                    .unwrap()
                    .build()
                    .unwrap();
                let mut sink = MemorySink::for_pipeline(&pipeline).unwrap();
                RayonScheduler::new(RayonScheduler::default_threads())
                    .unwrap()
                    .run(&pipeline, &mut sink)
                    .unwrap();
                black_box(sink.into_buffer())
            });
        });
    }

    group.finish();

    let mut rgb_group = c.benchmark_group("bandmean_u8_rgb");

    for &size in &[512u32, 2048, 8192] {
        let pixels = vec![128u8; size as usize * size as usize * 3];

        rgb_group.bench_with_input(BenchmarkId::from_parameter(size), &size, |b, &size| {
            b.iter(|| {
                let source = MemorySource::<U8>::new(size, size, 3, pixels.clone()).unwrap();
                let op = BandMean::<U8>::new(3);
                let dyn_op = Box::new(OperationBridge::new_pixel_local(op, 3));
                let pipeline = ImagePipeline::from_source(source)
                    .then(dyn_op)
                    .unwrap()
                    .build()
                    .unwrap();
                let mut sink = MemorySink::for_pipeline(&pipeline).unwrap();
                RayonScheduler::new(RayonScheduler::default_threads())
                    .unwrap()
                    .run(&pipeline, &mut sink)
                    .unwrap();
                black_box(sink.into_buffer())
            });
        });
    }

    rgb_group.finish();
}

criterion_group!(benches, bench_bandmean);
criterion_main!(benches);
