/// Benchmark: separable GaussBlur (GaussBlurH → GaussBlurV) on F32 images.
///
/// Measures the full pipeline path:
///   MemorySource<F32> → GaussBlurH<F32> → GaussBlurV<F32> → MemorySink
/// via RayonScheduler with `default_threads()`.
///
/// sigma = 3.0 → radius = 9 → kernel size = 19.
///
/// Complexity comparison vs Conv2d with an equivalent 2D Gaussian kernel:
///   Conv2d:             19 × 19 = 361 multiply-adds per pixel per band.
///   GaussBlurH + GaussBlurV: 2 × 19 =  38 multiply-adds per pixel per band.
/// Expected speedup: ~9× on throughput-bound workloads.
use criterion::{BenchmarkId, Criterion, black_box, criterion_group, criterion_main};
use viprs::{
    adapters::{
        pipeline::PipelineBuilder, scheduler::rayon_scheduler::RayonScheduler,
        sinks::memory::MemorySink, sources::memory::MemorySource,
    },
    domain::{
        format::{F32, U8},
        op::OperationBridge,
        ops::convolution::{GaussBlurH, GaussBlurV},
    },
    ports::scheduler::TileScheduler,
};

fn bench_gauss_blur(c: &mut Criterion) {
    let sigma = 3.0f32;
    let mut group = c.benchmark_group("gauss_blur_f32_sigma3");

    for &size in &[512u32, 2048, 8192] {
        let pixel_count = (size as usize) * (size as usize);
        let pixels = vec![1.0f32; pixel_count];

        group.bench_with_input(BenchmarkId::from_parameter(size), &size, |b, &size| {
            b.iter(|| {
                let source = MemorySource::<F32>::new(size, size, 1, pixels.clone()).unwrap();

                // Horizontal pass: F32 → F32
                let h_op = GaussBlurH::<F32>::new(sigma);
                let h_bridge = OperationBridge::new(h_op, 1u32);

                // Vertical pass: F32 → F32
                let v_op = GaussBlurV::<F32>::new(sigma);
                let v_bridge = OperationBridge::new(v_op, 1u32);

                let pipeline = PipelineBuilder::from_source(source)
                    .then(Box::new(h_bridge))
                    .unwrap()
                    .then(Box::new(v_bridge))
                    .unwrap()
                    .build()
                    .unwrap();

                let mut sink = MemorySink::for_pipeline(&pipeline);
                RayonScheduler::new(RayonScheduler::default_threads())
                    .unwrap()
                    .run(&pipeline, &mut sink)
                    .unwrap();
                black_box(sink.into_buffer())
            });
        });
    }

    group.finish();

    let mut rgb_group = c.benchmark_group("gauss_blur_u8_rgb_sigma3");

    for &size in &[512u32, 2048, 8192] {
        let pixel_count = (size as usize) * (size as usize);
        let pixels = vec![128u8; pixel_count * 3];

        rgb_group.bench_with_input(BenchmarkId::from_parameter(size), &size, |b, &size| {
            b.iter(|| {
                let source = MemorySource::<U8>::new(size, size, 3, pixels.clone()).unwrap();
                let pipeline = PipelineBuilder::from_source(source)
                    .gauss_blur(sigma)
                    .unwrap()
                    .build()
                    .unwrap();
                let mut sink = MemorySink::for_pipeline(&pipeline);
                RayonScheduler::new(RayonScheduler::default_threads())
                    .unwrap()
                    .run(&pipeline, &mut sink)
                    .unwrap();
                black_box(sink.into_buffer())
            });
        });
    }

    rgb_group.finish();
}

criterion_group!(benches, bench_gauss_blur);
criterion_main!(benches);
