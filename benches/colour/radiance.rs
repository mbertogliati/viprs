#![allow(missing_docs)]
// Benchmarks standalone Radiance parity ops.
// see B-175
use criterion::{BenchmarkId, Criterion, black_box, criterion_group, criterion_main};
use viprs::{
    adapters::{
      pipeline::ImagePipeline, scheduler::rayon_scheduler::RayonScheduler,
      sinks::memory::MemorySink, sources::memory::MemorySource,
    },
    domain::{
        format::{F32, U8},
        op::OperationBridge,
        ops::colour::{FloatToRadiance, RadianceToFloat},
    },
    ports::scheduler::TileScheduler,
};

fn float_pixels(pixel_count: usize) -> Vec<f32> {
    let mut pixels = Vec::with_capacity(pixel_count * 3);
    for index in 0..pixel_count {
        let red = 0.125 + ((index % 29) as f32 * 0.25);
        let green = 0.0625 + (((index * 3) % 23) as f32 * 0.2);
        let blue = 0.03125 + (((index * 5) % 19) as f32 * 0.3);
        pixels.extend_from_slice(&[red, green, blue]);
    }
    pixels
}

fn radiance_pixels(pixel_count: usize) -> Vec<u8> {
    let floats = float_pixels(pixel_count);
    let mut pixels = Vec::with_capacity(pixel_count * 4);
    for pixel in floats.chunks_exact(3) {
        let input = [pixel[0], pixel[1], pixel[2]];
        let max_component = input[0].max(input[1]).max(input[2]);
        if max_component <= 1e-32 {
            pixels.extend_from_slice(&[0, 0, 0, 0]);
            continue;
        }
        let exponent = max_component.log2().floor() as i32 + 1;
        let scale = 255.9999_f32 * 2.0_f32.powi(-exponent);
        pixels.extend_from_slice(&[
            if input[0] > 0.0 {
                (input[0] * scale) as u8
            } else {
                0
            },
            if input[1] > 0.0 {
                (input[1] * scale) as u8
            } else {
                0
            },
            if input[2] > 0.0 {
                (input[2] * scale) as u8
            } else {
                0
            },
            (exponent + 128) as u8,
        ]);
    }
    pixels
}

fn bench_radiance(c: &mut Criterion) {
    let mut group = c.benchmark_group("colour_radiance");

    for &size in &[512_u32, 2048, 8192] {
        let pixel_count = size as usize * size as usize;
        let float_rgb = float_pixels(pixel_count);
        let radiance = radiance_pixels(pixel_count);

        group.bench_with_input(
            BenchmarkId::new("float_to_radiance", size),
            &size,
            |b, &size| {
                b.iter(|| {
                    let source =
                        MemorySource::<F32>::new(size, size, 3, float_rgb.clone()).unwrap();
                    let pipeline = ImagePipeline::from_source(source)
                        .then(Box::new(OperationBridge::new_pixel_local(
                            FloatToRadiance,
                            3,
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
            BenchmarkId::new("radiance_to_float", size),
            &size,
            |b, &size| {
                b.iter(|| {
                    let source = MemorySource::<U8>::new(size, size, 4, radiance.clone()).unwrap();
                    let pipeline = ImagePipeline::from_source(source)
                        .then(Box::new(OperationBridge::new_pixel_local(
                            RadianceToFloat,
                            4,
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

criterion_group!(benches, bench_radiance);
criterion_main!(benches);
