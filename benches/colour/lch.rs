#![allow(missing_docs)]
use criterion::{BenchmarkId, Criterion, black_box, criterion_group, criterion_main};
use viprs::{
    adapters::{
      pipeline::ImagePipeline, scheduler::rayon_scheduler::RayonScheduler,
      sinks::memory::MemorySink, sources::memory::MemorySource,
    },
    domain::{
        colorspace::{Lab, Lch},
        format::F32,
        ops::colour::{ColourConvertBridge, LabToLch, LchToLab},
    },
    ports::scheduler::TileScheduler,
};

fn lab_pixels(pixel_count: usize) -> Vec<f32> {
    let mut pixels = Vec::with_capacity(pixel_count * 3);
    for index in 0..pixel_count {
        let lightness = (index % 101) as f32;
        let a = ((index % 257) as f32) - 128.0;
        let b = (((index * 3) % 257) as f32) - 128.0;
        pixels.extend_from_slice(&[lightness, a, b]);
    }
    pixels
}

fn lch_pixels(pixel_count: usize) -> Vec<f32> {
    let mut pixels = Vec::with_capacity(pixel_count * 3);
    for index in 0..pixel_count {
        let lightness = (index % 101) as f32;
        let chroma = ((index % 181) as f32) * 0.75;
        let hue = (index % 360) as f32;
        pixels.extend_from_slice(&[lightness, chroma, hue]);
    }
    pixels
}

fn bench_lch(c: &mut Criterion) {
    let mut group = c.benchmark_group("colour_lch");

    for &size in &[512_u32, 2048, 8192] {
        let pixel_count = size as usize * size as usize;
        let lab = lab_pixels(pixel_count);
        let lch = lch_pixels(pixel_count);

        group.bench_with_input(
            BenchmarkId::new("lab_to_lch", size),
            &size,
            |b, &image_size| {
                b.iter(|| {
                    let source =
                        MemorySource::<F32>::new(image_size, image_size, 3, lab.clone()).unwrap();
                    let pipeline = ImagePipeline::from_source(source)
                        .then(Box::new(ColourConvertBridge::<LabToLch, Lab, Lch>::new(
                            LabToLch, 3,
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
            BenchmarkId::new("lch_to_lab", size),
            &size,
            |b, &image_size| {
                b.iter(|| {
                    let source =
                        MemorySource::<F32>::new(image_size, image_size, 3, lch.clone()).unwrap();
                    let pipeline = ImagePipeline::from_source(source)
                        .then(Box::new(ColourConvertBridge::<LchToLab, Lch, Lab>::new(
                            LchToLab, 3,
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

criterion_group!(benches, bench_lch);
criterion_main!(benches);
