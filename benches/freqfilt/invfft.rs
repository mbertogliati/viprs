#![allow(missing_docs)]
#![cfg(feature = "fft")]

use criterion::{BenchmarkId, Criterion, black_box, criterion_group, criterion_main};
use viprs::{
    adapters::{
        pipeline::internal::PipelinePlan, scheduler::rayon_scheduler::RayonScheduler,
        sinks::memory::MemorySink, sources::memory::MemorySource,
    },
    domain::{
        format::F32,
        op::OperationBridge,
        ops::freqfilt::{FwFftOp, InvFftOp},
    },
    ports::scheduler::TileScheduler,
};

fn make_spatial_pixels(size: u32) -> Vec<f32> {
    let pixel_count = (size as usize) * (size as usize);
    (0..pixel_count)
        .map(|idx| {
            let x = (idx % size as usize) as f32;
            let y = (idx / size as usize) as f32;
            (x * 0.03125).sin() + (y * 0.015625).cos()
        })
        .collect()
}

fn make_frequency_pixels(size: u32, scheduler: &RayonScheduler) -> Vec<f32> {
    let source = MemorySource::<F32>::new(size, size, 1, make_spatial_pixels(size)).unwrap();
    let pipeline = PipelinePlan::from_source(source)
        .append_dyn_op(Box::new(OperationBridge::new(
            FwFftOp::<F32>::new(size, size).expect("FwFftOp should construct"),
            1,
        )))
        .unwrap()
        .compile()
        .unwrap();
    let mut sink = MemorySink::for_pipeline(&pipeline).unwrap();
    scheduler.run(&pipeline, &mut sink).unwrap();
    bytemuck::cast_slice::<u8, f32>(&sink.into_buffer()).to_vec()
}

fn bench_invfft(c: &mut Criterion) {
    let mut group = c.benchmark_group("invfft");
    let scheduler = RayonScheduler::new(RayonScheduler::default_threads()).unwrap();

    for &size in &[512u32, 2048, 8192] {
        let frequency_pixels = make_frequency_pixels(size, &scheduler);
        group.bench_with_input(BenchmarkId::from_parameter(size), &size, |b, &size| {
            b.iter(|| {
                let source =
                    MemorySource::<F32>::new(size, size, 2, frequency_pixels.clone()).unwrap();
                let pipeline = PipelinePlan::from_source(source)
                    .append_dyn_op(Box::new(OperationBridge::new(
                        InvFftOp::<F32>::new(size, size).expect("InvFftOp should construct"),
                        2,
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

criterion_group!(benches, bench_invfft);
criterion_main!(benches);
