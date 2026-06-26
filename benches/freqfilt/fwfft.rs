#![allow(missing_docs)]
#![cfg(feature = "fft")]

use criterion::{BenchmarkId, Criterion, black_box, criterion_group, criterion_main};
use viprs::{
    adapters::{
        pipeline::internal::PipelineBuilder, scheduler::rayon_scheduler::RayonScheduler,
        sinks::memory::MemorySink, sources::memory::MemorySource,
    },
    domain::{format::F32, op::OperationBridge, ops::freqfilt::FwFftOp},
    ports::scheduler::TileScheduler,
};

fn make_pixels(size: u32) -> Vec<f32> {
    let pixel_count = (size as usize) * (size as usize);
    (0..pixel_count)
        .map(|idx| {
            let x = (idx % size as usize) as f32;
            let y = (idx / size as usize) as f32;
            x.mul_add(0.5, y * 0.25)
        })
        .collect()
}

fn bench_fwfft(c: &mut Criterion) {
    let mut group = c.benchmark_group("fwfft");
    let scheduler = RayonScheduler::new(RayonScheduler::default_threads()).unwrap();

    for &size in &[512u32, 2048, 8192] {
        let pixels = make_pixels(size);
        group.bench_with_input(BenchmarkId::from_parameter(size), &size, |b, &size| {
            b.iter(|| {
                let source = MemorySource::<F32>::new(size, size, 1, pixels.clone()).unwrap();
                let pipeline = PipelineBuilder::from_source(source)
                    .then(Box::new(OperationBridge::new(
                        FwFftOp::<F32>::new(size, size).expect("FwFftOp should construct"),
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

criterion_group!(benches, bench_fwfft);
criterion_main!(benches);
