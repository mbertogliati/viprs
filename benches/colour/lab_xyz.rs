#![allow(missing_docs)]
// Benchmarks standalone Lab↔Xyz parity ops.
// see B-175
use criterion::{BenchmarkId, Criterion, black_box, criterion_group, criterion_main};
use viprs::{
    adapters::{
        pipeline::internal::PipelinePlan, scheduler::rayon_scheduler::RayonScheduler,
        sinks::memory::MemorySink, sources::memory::MemorySource,
    },
    domain::{
        colorspace::{ColorspaceId, Lab, Xyz},
        format::F32,
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

fn xyz_pixels(pixel_count: usize) -> Vec<f32> {
    let mut pixels = Vec::with_capacity(pixel_count * 3);
    for index in 0..pixel_count {
        let x = 0.01 + ((index % 37) as f32 * 0.02);
        let y = 0.01 + (((index * 3) % 29) as f32 * 0.025);
        let z = 0.01 + (((index * 5) % 31) as f32 * 0.03);
        pixels.extend_from_slice(&[x, y, z]);
    }
    pixels
}

fn bench_lab_xyz(c: &mut Criterion) {
    let mut group = c.benchmark_group("colour_lab_xyz");

    for &size in &[512_u32, 2048, 8192] {
        let pixel_count = size as usize * size as usize;
        let lab = lab_pixels(pixel_count);
        let xyz = xyz_pixels(pixel_count);

        group.bench_with_input(BenchmarkId::new("lab_to_xyz", size), &size, |b, &size| {
            b.iter(|| {
                let source = MemorySource::<F32>::new(size, size, 3, lab.clone()).unwrap();
                let pipeline = PipelinePlan::from_source(source)
                    .with_colorspace(ColorspaceId::Lab)
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

        group.bench_with_input(BenchmarkId::new("xyz_to_lab", size), &size, |b, &size| {
            b.iter(|| {
                let source = MemorySource::<F32>::new(size, size, 3, xyz.clone()).unwrap();
                let pipeline = PipelinePlan::from_source(source)
                    .with_colorspace(ColorspaceId::Xyz)
                    .plan_colourspace::<Lab>()
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

criterion_group!(benches, bench_lab_xyz);
criterion_main!(benches);
