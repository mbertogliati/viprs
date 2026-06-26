#![allow(missing_docs)]
use criterion::{BenchmarkId, Criterion, black_box, criterion_group, criterion_main};
use viprs::{
    adapters::{
        pipeline::internal::PipelinePlan, scheduler::rayon_scheduler::RayonScheduler,
        sinks::memory::MemorySink, sources::memory::MemorySource,
    },
    domain::{format::U8, kernel::InterpolationKernel},
    ports::scheduler::TileScheduler,
};

const STANDARD_SIZES: [u32; 3] = [512, 2048, 8192];

fn bench_resize_kernel(c: &mut Criterion) {
    for &size in &STANDARD_SIZES {
        let mut group = c.benchmark_group(format!("resize_kernels_{size}_u8"));
        let pixels = vec![128u8; size as usize * size as usize];

        for kernel in [
            InterpolationKernel::Nearest,
            InterpolationKernel::Bilinear,
            InterpolationKernel::Bicubic,
            InterpolationKernel::Lanczos3,
        ] {
            group.bench_with_input(
                BenchmarkId::from_parameter(format!("{kernel:?}")),
                &kernel,
                |b, &kernel| {
                    b.iter(|| {
                        let source =
                            MemorySource::<U8>::new(size, size, 1, pixels.clone()).unwrap();
                        let pipeline = PipelinePlan::from_source(source)
                            .plan_resize(viprs::domain::ops::resample::resize::Resize::new(
                                0.75, 0.75, kernel,
                            ))
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
                },
            );
        }

        group.finish();
    }
}

criterion_group!(benches, bench_resize_kernel);
criterion_main!(benches);
