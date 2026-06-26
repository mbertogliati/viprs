#![allow(missing_docs)]
use criterion::{BenchmarkId, Criterion, black_box, criterion_group, criterion_main};
use viprs::{
    adapters::{
        pipeline::internal::PipelineBuilder, scheduler::rayon_scheduler::RayonScheduler,
        sinks::memory::MemorySink, sources::memory::MemorySource,
    },
    domain::{
        format::U8,
        kernel::InterpolationKernel,
        ops::resample::{Quadratic, QuadraticCoefficients},
    },
    pipeline::OperationBridge,
    ports::scheduler::TileScheduler,
};

fn bench_quadratic(c: &mut Criterion) {
    let mut group = c.benchmark_group("quadratic_u8");
    let coeffs = QuadraticCoefficients::from_order3([
        [0.0, 0.0],
        [0.0, 0.0],
        [0.0, 0.0],
        [1.0e-6, -1.0e-6],
        [1.0e-6, -5.0e-7],
        [-1.0e-6, 5.0e-7],
    ]);

    for &size in &[512u32, 2048, 8192] {
        let pixel_count = (size as usize) * (size as usize);
        let pixels: Vec<u8> = (0..pixel_count).map(|idx| (idx % 251) as u8).collect();

        group.bench_with_input(BenchmarkId::from_parameter(size), &size, |b, &size| {
            b.iter(|| {
                let source = MemorySource::<U8>::new(size, size, 1, pixels.clone()).unwrap();
                let op = Quadratic::<U8>::new(coeffs, InterpolationKernel::Bilinear, size, size);
                let pipeline = PipelineBuilder::from_source(source)
                    .then(Box::new(OperationBridge::new(op, 1)))
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

criterion_group!(benches, bench_quadratic);
criterion_main!(benches);
