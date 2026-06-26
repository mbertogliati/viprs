#![allow(missing_docs)]
use criterion::{BenchmarkId, Criterion, black_box, criterion_group, criterion_main};
use viprs::{
    adapters::{
        pipeline::internal::PipelinePlan, scheduler::rayon_scheduler::RayonScheduler,
        sinks::memory::MemorySink, sources::memory::MemorySource,
    },
    domain::{format::F32, op::OperationBridge, ops::convolution::FastCorOp},
    ports::scheduler::TileScheduler,
};

fn make_pixels(size: u32) -> Vec<f32> {
    (0..size as usize * size as usize)
        .map(|idx| ((idx * 13 + idx / size as usize * 7) % 251) as f32 / 255.0)
        .collect()
}

fn reference_patch() -> Vec<f32> {
    vec![
        0.0, 0.1, 0.3, 0.1, 0.0, 0.1, 0.4, 0.8, 0.4, 0.1, 0.2, 0.7, 1.0, 0.7, 0.2, 0.1, 0.4, 0.8,
        0.4, 0.1, 0.0, 0.1, 0.3, 0.1, 0.0,
    ]
}

fn bench_fastcor(c: &mut Criterion) {
    let mut group = c.benchmark_group("fastcor_f32_ssd5x5");
    let reference = reference_patch();

    for &size in &[512u32, 2048, 8192] {
        let pixels = make_pixels(size);

        group.bench_with_input(BenchmarkId::from_parameter(size), &size, |b, &size| {
            b.iter(|| {
                let source = MemorySource::<F32>::new(size, size, 1, pixels.clone()).unwrap();
                let op = FastCorOp::<F32>::new(reference.clone(), 5, 5, 1).unwrap();
                let pipeline = PipelinePlan::from_source(source)
                    .append_dyn_op(Box::new(OperationBridge::new(op, 1)))
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

criterion_group!(benches, bench_fastcor);
criterion_main!(benches);
