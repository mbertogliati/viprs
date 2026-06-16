#![allow(missing_docs)]
use criterion::{BenchmarkId, Criterion, black_box, criterion_group, criterion_main};
use viprs::{
    adapters::{
        pipeline::PipelineBuilder, scheduler::rayon_scheduler::RayonScheduler,
        sinks::memory::MemorySink, sources::memory::MemorySource,
    },
    domain::format::U8,
    ports::scheduler::TileScheduler,
};

fn rgba_pixels(pixel_count: usize) -> Vec<u8> {
    let mut pixels = Vec::with_capacity(pixel_count * 4);
    for index in 0..pixel_count {
        let base = (index % 256) as u8;
        pixels.extend_from_slice(&[
            base,
            base.wrapping_add(37),
            base.wrapping_add(91),
            base.wrapping_mul(17).wrapping_add(13),
        ]);
    }
    pixels
}

fn bench_flatten(c: &mut Criterion) {
    let mut group = c.benchmark_group("colour_flatten");

    for &size in &[512_u32, 2048, 8192] {
        let pixel_count = size as usize * size as usize;
        let rgba = rgba_pixels(pixel_count);

        group.bench_with_input(
            BenchmarkId::from_parameter(size),
            &size,
            |b, &image_size| {
                b.iter(|| {
                    let source =
                        MemorySource::<U8>::new(image_size, image_size, 4, rgba.clone()).unwrap();
                    let pipeline = PipelineBuilder::from_source(source)
                        .flatten([0.0, 0.0, 0.0, 1.0])
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
            },
        );
    }

    group.finish();
}

criterion_group!(benches, bench_flatten);
criterion_main!(benches);
