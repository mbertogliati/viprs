#![allow(missing_docs)]
/// Benchmark: `Affine<U8>` across the standard resample sizes.
///
/// Exercises the full pipeline path: `MemorySource → Affine → MemorySink` via
/// `RayonScheduler`. The bilinear cases use the same 2048→400 downscale ratio as
/// the thumbnail hot path that motivated B-283.
use criterion::{BenchmarkId, Criterion, black_box, criterion_group, criterion_main};
use viprs::{
    adapters::{
      pipeline::ImagePipeline, scheduler::rayon_scheduler::RayonScheduler,
      sinks::memory::MemorySink, sources::memory::MemorySource,
    },
    domain::{format::U8, kernel::InterpolationKernel},
    ports::scheduler::TileScheduler,
};

const STANDARD_SIZES: [u32; 3] = [512, 2048, 8192];
const THUMBNAIL_RATIO_NUMERATOR: u32 = 400;
const THUMBNAIL_RATIO_DENOMINATOR: u32 = 2048;

fn scaled_thumbnail_size(size: u32) -> u32 {
    (size * THUMBNAIL_RATIO_NUMERATOR) / THUMBNAIL_RATIO_DENOMINATOR
}

fn run_affine_pipeline(
    input_size: u32,
    output_size: u32,
    bands: u32,
    kernel: InterpolationKernel,
    pixels: &[u8],
) {
    let scale = input_size as f64 / output_size as f64;
    let source = MemorySource::<U8>::new(input_size, input_size, bands, pixels.to_vec()).unwrap();
    let pipeline = ImagePipeline::from_source(source)
        .affine(
            [scale, 0.0, 0.0, scale],
            0.0,
            0.0,
            output_size,
            output_size,
            kernel,
        )
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

fn bench_affine_nearest(c: &mut Criterion) {
    let mut group = c.benchmark_group("affine_u8_identity_nearest");
    for &size in &STANDARD_SIZES {
        let pixels = vec![128u8; size as usize * size as usize];
        group.bench_with_input(BenchmarkId::from_parameter(size), &size, |b, &size| {
            b.iter(|| {
                run_affine_pipeline(size, size, 1, InterpolationKernel::Nearest, &pixels);
            });
        });
    }
    group.finish();
}

fn bench_affine_bilinear(c: &mut Criterion) {
    let mut group = c.benchmark_group("affine_u8_downscale_bilinear");
    for &size in &STANDARD_SIZES {
        let pixels = vec![128u8; size as usize * size as usize];
        let output_size = scaled_thumbnail_size(size);
        group.bench_with_input(BenchmarkId::from_parameter(size), &size, |b, &size| {
            b.iter(|| {
                run_affine_pipeline(size, output_size, 1, InterpolationKernel::Bilinear, &pixels);
            });
        });
    }
    group.finish();
}

fn bench_affine_rgb_bilinear(c: &mut Criterion) {
    let mut group = c.benchmark_group("affine_u8_rgb_downscale_bilinear");
    for &size in &STANDARD_SIZES {
        let pixels = vec![128u8; size as usize * size as usize * 3];
        let output_size = scaled_thumbnail_size(size);
        group.bench_with_input(BenchmarkId::from_parameter(size), &size, |b, &size| {
            b.iter(|| {
                run_affine_pipeline(size, output_size, 3, InterpolationKernel::Bilinear, &pixels);
            });
        });
    }
    group.finish();
}

criterion_group!(
    benches,
    bench_affine_nearest,
    bench_affine_bilinear,
    bench_affine_rgb_bilinear
);
criterion_main!(benches);
