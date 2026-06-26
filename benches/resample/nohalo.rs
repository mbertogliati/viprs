#![allow(missing_docs)]
use criterion::{BenchmarkId, Criterion, black_box, criterion_group, criterion_main};
use viprs::{
    adapters::{
        pipeline::ImagePipeline, scheduler::rayon_scheduler::RayonScheduler,
        sinks::memory::MemorySink, sources::memory::MemorySource,
    },
    domain::{format::U8, kernel::InterpolationKernel},
    ports::scheduler::TileScheduler,
};

fn patterned_pixels(size: u32) -> Vec<u8> {
    (0..size as usize * size as usize)
        .map(|index| ((index * 17 + index / 31) % 251) as u8)
        .collect()
}

fn bench_nohalo_affine(c: &mut Criterion) {
    let mut group = c.benchmark_group("affine_u8_nohalo_fractional");

    for &size in &[512_u32, 2048, 8192] {
        let pixels = patterned_pixels(size);
        group.bench_with_input(BenchmarkId::from_parameter(size), &size, |b, &size| {
            b.iter(|| {
                let source = MemorySource::<U8>::new(size, size, 1, pixels.clone()).unwrap();
                let pipeline = ImagePipeline::from_source(source)
                    .affine(
                        [1.0, 0.0, 0.0, 1.0],
                        0.25,
                        0.25,
                        size,
                        size,
                        InterpolationKernel::Nohalo,
                    )
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

criterion_group!(benches, bench_nohalo_affine);
criterion_main!(benches);
