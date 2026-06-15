use criterion::{BenchmarkId, Criterion, black_box, criterion_group, criterion_main};
use viprs::{
    adapters::{
        pipeline::PipelineBuilder, scheduler::rayon_scheduler::RayonScheduler,
        sinks::memory::MemorySink, sources::memory::MemorySource,
    },
    domain::{format::F32, op::OperationBridge, ops::convolution::ConvaOp},
    ports::scheduler::TileScheduler,
};

fn gaussian_kernel_2d(size: usize) -> Vec<Vec<f64>> {
    let radius = (size / 2) as i32;
    let sigma = size as f32 / 6.0;
    let sigma_sq = 2.0 * sigma * sigma;
    let mut kernel_1d = vec![0.0f32; size];
    let mut sum = 0.0f32;
    for (index, value) in kernel_1d.iter_mut().enumerate() {
        let x = index as i32 - radius;
        *value = (-(x * x) as f32 / sigma_sq).exp();
        sum += *value;
    }
    for value in &mut kernel_1d {
        *value /= sum;
    }

    let mut kernel = vec![vec![0.0; size]; size];
    for y in 0..size {
        for x in 0..size {
            kernel[y][x] = f64::from(kernel_1d[y]) * f64::from(kernel_1d[x]);
        }
    }
    kernel
}

fn bench_conva(c: &mut Criterion) {
    let mut group = c.benchmark_group("conva_vs_conv2d_gaussian_f32");

    for kernel_size in [5usize, 11, 21] {
        let kernel = gaussian_kernel_2d(kernel_size);

        for size in [512u32, 2048, 8192] {
            let pixel_count = size as usize * size as usize;
            let pixels = vec![1.0f32; pixel_count];

            group.bench_with_input(
                BenchmarkId::new(format!("conva_k{kernel_size}"), size),
                &size,
                |b, &image_size| {
                    b.iter(|| {
                        let source =
                            MemorySource::<F32>::new(image_size, image_size, 1, pixels.clone())
                                .unwrap();
                        let conva = ConvaOp::<F32>::new(kernel.clone()).unwrap();
                        let pipeline = PipelineBuilder::from_source(source)
                            .then(Box::new(OperationBridge::new(conva, 1)))
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
                },
            );

            group.bench_with_input(
                BenchmarkId::new(format!("conv2d_k{kernel_size}"), size),
                &size,
                |b, &image_size| {
                    b.iter(|| {
                        let source =
                            MemorySource::<F32>::new(image_size, image_size, 1, pixels.clone())
                                .unwrap();
                        let pipeline = PipelineBuilder::from_source(source)
                            .conv2d(kernel.clone())
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
                },
            );
        }
    }

    group.finish();
}

criterion_group!(benches, bench_conva);
criterion_main!(benches);
