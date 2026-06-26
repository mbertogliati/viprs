#![allow(missing_docs)]
// Benchmarks the standalone LabQ→SRgb parity op.
// see B-258
use criterion::{BenchmarkId, Criterion, black_box, criterion_group, criterion_main};
use viprs::{
    adapters::{
        pipeline::internal::PipelineBuilder, scheduler::rayon_scheduler::RayonScheduler,
        sinks::memory::MemorySink, sources::memory::MemorySource,
    },
    domain::{format::U8, op::OperationBridge, ops::colour::labq_to_srgb::LabQToSRgb},
    ports::scheduler::TileScheduler,
};

fn labq_pixels(pixel_count: usize) -> Vec<u8> {
    let mut pixels = Vec::with_capacity(pixel_count * 4);
    for index in 0..pixel_count {
        pixels.extend_from_slice(&[
            (index % 256) as u8,
            ((index * 3) % 256) as u8,
            ((index * 7) % 256) as u8,
            ((index * 13) % 256) as u8,
        ]);
    }
    pixels
}

fn bench_labq_to_srgb(c: &mut Criterion) {
    let mut group = c.benchmark_group("colour_labq_to_srgb");

    for &size in &[512_u32, 2048, 8192] {
        let pixel_count = size as usize * size as usize;
        let labq = labq_pixels(pixel_count);

        group.bench_with_input(BenchmarkId::from_parameter(size), &size, |b, &size| {
            b.iter(|| {
                let source = MemorySource::<U8>::new(size, size, 4, labq.clone()).unwrap();
                let pipeline = PipelineBuilder::from_source(source)
                    .then(Box::new(OperationBridge::new(LabQToSRgb, 4)))
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
}

criterion_group!(benches, bench_labq_to_srgb);
criterion_main!(benches);
