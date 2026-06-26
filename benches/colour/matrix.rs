#![allow(missing_docs)]
use std::time::Duration;

use criterion::{BenchmarkId, Criterion, SamplingMode, black_box, criterion_group, criterion_main};
use viprs::{
    adapters::{
        pipeline::internal::PipelineBuilder, scheduler::rayon_scheduler::RayonScheduler,
        sinks::memory::MemorySink, sources::memory::MemorySource,
    },
    domain::{
        colorspace::{Colorspace, ColorspaceId, Hsv, Lab, SRgb, ScRgb},
        format::{F32, U8},
    },
    ports::scheduler::TileScheduler,
};

const STANDARD_SIZES: &[u32] = &[512, 2048, 8192];

fn srgb_pixels(pixel_count: usize, bands: u32) -> Vec<u8> {
    let mut pixels = Vec::with_capacity(pixel_count * bands as usize);
    for index in 0..pixel_count {
        let base = (index % 256) as u8;
        pixels.extend_from_slice(&[
            base,
            base.wrapping_mul(3).wrapping_add(17),
            base.wrapping_mul(5).wrapping_add(29),
        ]);
        if bands == 4 {
            pixels.push(base.wrapping_mul(7).wrapping_add(61));
        }
    }
    pixels
}

fn lab_pixels(pixel_count: usize, bands: u32) -> Vec<f32> {
    let mut pixels = Vec::with_capacity(pixel_count * bands as usize);
    for index in 0..pixel_count {
        let lightness = 5.0 + ((index % 91) as f32);
        let a = ((index % 171) as f32) - 85.0;
        let b = (((index * 7) % 191) as f32) - 95.0;
        pixels.extend_from_slice(&[lightness, a, b]);
        if bands == 4 {
            pixels.push((index % 256) as f32);
        }
    }
    pixels
}

fn hsv_pixels(pixel_count: usize, bands: u32) -> Vec<f32> {
    let mut pixels = Vec::with_capacity(pixel_count * bands as usize);
    for index in 0..pixel_count {
        let hue = ((index * 37) % 360) as f32;
        let saturation = 0.15 + (((index * 11) % 80) as f32 / 100.0);
        let value = 0.2 + (((index * 13) % 75) as f32 / 100.0);
        pixels.extend_from_slice(&[hue, saturation.min(1.0), value.min(1.0)]);
        if bands == 4 {
            pixels.push((index % 256) as f32);
        }
    }
    pixels
}

fn scrgb_pixels(pixel_count: usize, bands: u32) -> Vec<f32> {
    let mut pixels = Vec::with_capacity(pixel_count * bands as usize);
    for index in 0..pixel_count {
        let red = ((index * 3) % 256) as f32 / 255.0;
        let green = ((index * 5 + 17) % 256) as f32 / 255.0;
        let blue = ((index * 7 + 29) % 256) as f32 / 255.0;
        pixels.extend_from_slice(&[red, green, blue]);
        if bands == 4 {
            pixels.push(((index * 11 + 61) % 256) as f32 / 255.0);
        }
    }
    pixels
}

fn run_u8_pipeline<To: Colorspace>(
    size: u32,
    bands: u32,
    source_colorspace: ColorspaceId,
    pixels: Vec<u8>,
) {
    let source = MemorySource::<U8>::new(size, size, bands, pixels).unwrap();
    let pipeline = PipelineBuilder::from_source(source)
        .with_colorspace(source_colorspace)
        .colourspace::<To>()
        .unwrap()
        .build()
        .unwrap();
    let mut sink = MemorySink::for_pipeline(&pipeline).unwrap();
    RayonScheduler::new(RayonScheduler::default_threads())
        .unwrap()
        .run(&pipeline, &mut sink)
        .unwrap();
    black_box(sink.into_buffer());
}

fn run_f32_pipeline<To: Colorspace>(
    size: u32,
    bands: u32,
    source_colorspace: ColorspaceId,
    pixels: Vec<f32>,
) {
    let source = MemorySource::<F32>::new(size, size, bands, pixels).unwrap();
    let pipeline = PipelineBuilder::from_source(source)
        .with_colorspace(source_colorspace)
        .colourspace::<To>()
        .unwrap()
        .build()
        .unwrap();
    let mut sink = MemorySink::for_pipeline(&pipeline).unwrap();
    RayonScheduler::new(RayonScheduler::default_threads())
        .unwrap()
        .run(&pipeline, &mut sink)
        .unwrap();
    black_box(sink.into_buffer());
}

fn bench_colour_matrix(c: &mut Criterion) {
    let mut group = c.benchmark_group("colourspace_matrix");
    group.sample_size(10);
    group.sampling_mode(SamplingMode::Flat);
    group.warm_up_time(Duration::from_secs(1));
    group.measurement_time(Duration::from_secs(3));

    for &size in STANDARD_SIZES {
        let pixel_count = size as usize * size as usize;

        let srgb_rgb = srgb_pixels(pixel_count, 3);
        group.bench_with_input(
            BenchmarkId::new("srgb_to_lab_rgb", size),
            &size,
            |b, &image_size| {
                b.iter(|| {
                    run_u8_pipeline::<Lab>(image_size, 3, ColorspaceId::SRgb, srgb_rgb.clone())
                });
            },
        );
        group.bench_with_input(
            BenchmarkId::new("srgb_to_hsv_rgb", size),
            &size,
            |b, &image_size| {
                b.iter(|| {
                    run_u8_pipeline::<Hsv>(image_size, 3, ColorspaceId::SRgb, srgb_rgb.clone())
                });
            },
        );
        group.bench_with_input(
            BenchmarkId::new("srgb_to_scrgb_rgb", size),
            &size,
            |b, &image_size| {
                b.iter(|| {
                    run_u8_pipeline::<ScRgb>(image_size, 3, ColorspaceId::SRgb, srgb_rgb.clone())
                });
            },
        );

        let srgb_rgba = srgb_pixels(pixel_count, 4);
        group.bench_with_input(
            BenchmarkId::new("srgb_to_lab_rgba", size),
            &size,
            |b, &image_size| {
                b.iter(|| {
                    run_u8_pipeline::<Lab>(image_size, 4, ColorspaceId::SRgb, srgb_rgba.clone())
                });
            },
        );
        group.bench_with_input(
            BenchmarkId::new("srgb_to_hsv_rgba", size),
            &size,
            |b, &image_size| {
                b.iter(|| {
                    run_u8_pipeline::<Hsv>(image_size, 4, ColorspaceId::SRgb, srgb_rgba.clone())
                });
            },
        );

        let lab_rgb = lab_pixels(pixel_count, 3);
        group.bench_with_input(
            BenchmarkId::new("lab_to_srgb_rgb", size),
            &size,
            |b, &image_size| {
                b.iter(|| {
                    run_f32_pipeline::<SRgb>(image_size, 3, ColorspaceId::Lab, lab_rgb.clone())
                });
            },
        );

        let lab_rgba = lab_pixels(pixel_count, 4);
        group.bench_with_input(
            BenchmarkId::new("lab_to_srgb_rgba", size),
            &size,
            |b, &image_size| {
                b.iter(|| {
                    run_f32_pipeline::<SRgb>(image_size, 4, ColorspaceId::Lab, lab_rgba.clone())
                });
            },
        );

        let hsv_rgb = hsv_pixels(pixel_count, 3);
        group.bench_with_input(
            BenchmarkId::new("hsv_to_srgb_rgb", size),
            &size,
            |b, &image_size| {
                b.iter(|| {
                    run_f32_pipeline::<SRgb>(image_size, 3, ColorspaceId::Hsv, hsv_rgb.clone())
                });
            },
        );

        let hsv_rgba = hsv_pixels(pixel_count, 4);
        group.bench_with_input(
            BenchmarkId::new("hsv_to_srgb_rgba", size),
            &size,
            |b, &image_size| {
                b.iter(|| {
                    run_f32_pipeline::<SRgb>(image_size, 4, ColorspaceId::Hsv, hsv_rgba.clone())
                });
            },
        );

        let scrgb_rgb = scrgb_pixels(pixel_count, 3);
        group.bench_with_input(
            BenchmarkId::new("scrgb_to_srgb_rgb", size),
            &size,
            |b, &image_size| {
                b.iter(|| {
                    run_f32_pipeline::<SRgb>(image_size, 3, ColorspaceId::ScRgb, scrgb_rgb.clone())
                });
            },
        );
    }

    group.finish();
}

criterion_group!(benches, bench_colour_matrix);
criterion_main!(benches);
