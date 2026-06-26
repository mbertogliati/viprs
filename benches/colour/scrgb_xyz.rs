#![allow(missing_docs)]
// Benchmarks standalone ScRgb↔Xyz parity ops.
// see B-175
use criterion::{BenchmarkId, Criterion, black_box, criterion_group, criterion_main};
use viprs::{
    adapters::{
        pipeline::internal::PipelinePlan, scheduler::rayon_scheduler::RayonScheduler,
        sinks::memory::MemorySink, sources::memory::MemorySource,
    },
    domain::{
        colorspace::{ColorspaceId, ScRgb, Xyz},
        format::F32,
    },
    ports::scheduler::TileScheduler,
};

fn scrgb_pixels(pixel_count: usize) -> Vec<f32> {
    let mut pixels = Vec::with_capacity(pixel_count * 3);
    for index in 0..pixel_count {
        let red = ((index % 19) as f32) * 0.15;
        let green = (((index * 3) % 23) as f32) * 0.12;
        let blue = (((index * 5) % 29) as f32) * 0.1;
        pixels.extend_from_slice(&[red, green, blue]);
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

fn bench_scrgb_xyz(c: &mut Criterion) {
    let mut group = c.benchmark_group("colour_scrgb_xyz");

    for &size in &[512_u32, 2048, 8192] {
        let pixel_count = size as usize * size as usize;
        let scrgb = scrgb_pixels(pixel_count);
        let xyz = xyz_pixels(pixel_count);

        group.bench_with_input(BenchmarkId::new("scrgb_to_xyz", size), &size, |b, &size| {
            b.iter(|| {
                let source = MemorySource::<F32>::new(size, size, 3, scrgb.clone()).unwrap();
                let pipeline = PipelinePlan::from_source(source)
                    .with_colorspace(ColorspaceId::ScRgb)
                    .plan_colourspace::<Xyz>()
                    .unwrap()
                    .compile()
                    .unwrap();
                let mut sink = MemorySink::for_pipeline(&pipeline).unwrap();
                RayonScheduler::new(RayonScheduler::default_threads())
                    .unwrap()
                    .run(&pipeline, &mut sink)
                    .unwrap();
                black_box(sink.into_buffer())
            });
        });

        group.bench_with_input(BenchmarkId::new("xyz_to_scrgb", size), &size, |b, &size| {
            b.iter(|| {
                let source = MemorySource::<F32>::new(size, size, 3, xyz.clone()).unwrap();
                let pipeline = PipelinePlan::from_source(source)
                    .with_colorspace(ColorspaceId::Xyz)
                    .plan_colourspace::<ScRgb>()
                    .unwrap()
                    .compile()
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

criterion_group!(benches, bench_scrgb_xyz);
criterion_main!(benches);
