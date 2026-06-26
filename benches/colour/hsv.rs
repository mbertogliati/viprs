#![allow(missing_docs)]
use criterion::{BenchmarkId, Criterion, black_box, criterion_group, criterion_main};
use viprs::{
    adapters::{
        pipeline::internal::PipelineBuilder, scheduler::rayon_scheduler::RayonScheduler,
        sinks::memory::MemorySink, sources::memory::MemorySource,
    },
    domain::{
        colorspace::{Hsv, SRgb},
        format::{F32, U8},
        ops::colour::{ColourConvertBridge, HsvToSRgb, SRgbToHsv},
    },
    ports::scheduler::TileScheduler,
};

fn srgb_pixels(pixel_count: usize) -> Vec<u8> {
    let mut pixels = Vec::with_capacity(pixel_count * 3);
    for index in 0..pixel_count {
        let base = (index % 256) as u8;
        pixels.extend_from_slice(&[base, base.wrapping_add(64), base.wrapping_add(128)]);
    }
    pixels
}

fn hsv_pixels(pixel_count: usize) -> Vec<f32> {
    let mut pixels = Vec::with_capacity(pixel_count * 3);
    for index in 0..pixel_count {
        let hue = (index % 360) as f32;
        let saturation = match index % 3 {
            0 => 0.25,
            1 => 0.5,
            _ => 1.0,
        };
        let value = 0.2 + ((index % 5) as f32 * 0.15);
        pixels.extend_from_slice(&[hue, saturation, value.min(1.0)]);
    }
    pixels
}

fn bench_hsv(c: &mut Criterion) {
    let mut group = c.benchmark_group("colour_hsv");

    for &size in &[512_u32, 2048, 8192] {
        let pixel_count = size as usize * size as usize;
        let srgb = srgb_pixels(pixel_count);
        let hsv = hsv_pixels(pixel_count);

        group.bench_with_input(
            BenchmarkId::new("srgb_to_hsv", size),
            &size,
            |b, &image_size| {
                b.iter(|| {
                    let source =
                        MemorySource::<U8>::new(image_size, image_size, 3, srgb.clone()).unwrap();
                    let pipeline = PipelineBuilder::from_source(source)
                        .then(Box::new(ColourConvertBridge::<SRgbToHsv, SRgb, Hsv>::new(
                            SRgbToHsv, 3,
                        )))
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

        group.bench_with_input(
            BenchmarkId::new("hsv_to_srgb", size),
            &size,
            |b, &image_size| {
                b.iter(|| {
                    let source =
                        MemorySource::<F32>::new(image_size, image_size, 3, hsv.clone()).unwrap();
                    let pipeline = PipelineBuilder::from_source(source)
                        .then(Box::new(ColourConvertBridge::<HsvToSRgb, Hsv, SRgb>::new(
                            HsvToSRgb, 3,
                        )))
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

criterion_group!(benches, bench_hsv);
criterion_main!(benches);
