#![allow(missing_docs)]
// Benchmarks standalone Cmyk↔Xyz parity ops.
// see B-175
use criterion::{BenchmarkId, Criterion, black_box, criterion_group, criterion_main};
use viprs::{
    adapters::{
        pipeline::ImagePipeline, scheduler::rayon_scheduler::RayonScheduler,
        sinks::memory::MemorySink, sources::memory::MemorySource,
    },
    domain::{
        colorspace::{Cmyk, ColorspaceId, Xyz},
        format::{F32, U8},
    },
    ports::scheduler::TileScheduler,
};

fn cmyk_pixels(pixel_count: usize) -> Vec<u8> {
    let mut pixels = Vec::with_capacity(pixel_count * 4);
    for index in 0..pixel_count {
        pixels.extend_from_slice(&[
            (index % 256) as u8,
            ((index * 3) % 256) as u8,
            ((index * 5) % 256) as u8,
            ((index * 7) % 256) as u8,
        ]);
    }
    pixels
}

fn xyz_pixels(pixel_count: usize) -> Vec<f32> {
    let mut pixels = Vec::with_capacity(pixel_count * 3);
    for index in 0..pixel_count {
        let x = 0.02 + ((index % 37) as f32 * 0.018);
        let y = 0.02 + (((index * 3) % 29) as f32 * 0.022);
        let z = 0.02 + (((index * 5) % 31) as f32 * 0.027);
        pixels.extend_from_slice(&[x, y, z]);
    }
    pixels
}

fn bench_cmyk_xyz(c: &mut Criterion) {
    let mut group = c.benchmark_group("colour_cmyk_xyz");

    for &size in &[512_u32, 2048, 8192] {
        let pixel_count = size as usize * size as usize;
        let cmyk = cmyk_pixels(pixel_count);
        let xyz = xyz_pixels(pixel_count);

        group.bench_with_input(BenchmarkId::new("cmyk_to_xyz", size), &size, |b, &size| {
            b.iter(|| {
                let source = MemorySource::<U8>::new(size, size, 4, cmyk.clone()).unwrap();
                let pipeline = ImagePipeline::from_source(source)
                    .with_colorspace(ColorspaceId::Cmyk)
                    .colourspace::<Xyz>()
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

        group.bench_with_input(BenchmarkId::new("xyz_to_cmyk", size), &size, |b, &size| {
            b.iter(|| {
                let source = MemorySource::<F32>::new(size, size, 3, xyz.clone()).unwrap();
                let pipeline = ImagePipeline::from_source(source)
                    .with_colorspace(ColorspaceId::Xyz)
                    .colourspace::<Cmyk>()
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

criterion_group!(benches, bench_cmyk_xyz);
criterion_main!(benches);
