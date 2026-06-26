#![allow(missing_docs)]
use criterion::{BenchmarkId, Criterion, black_box, criterion_group, criterion_main};
use viprs::{
    adapters::{
        pipeline::internal::PipelineBuilder, scheduler::rayon_scheduler::RayonScheduler,
        sinks::memory::MemorySink, sources::memory::MemorySource,
    },
    domain::{
        format::F32,
        op::OperationBridge,
        ops::create::{GaussmatOp, GaussmatPrecision},
    },
    ports::scheduler::TileScheduler,
};

fn op_for_target_size(target: u32) -> GaussmatOp<F32> {
    let mut low = 0.000_001f64;
    let mut high = target as f64;
    for _ in 0..48 {
        let sigma = (low + high) * 0.5;
        let op = GaussmatOp::<F32>::new(sigma, 0.1)
            .unwrap()
            .with_precision(GaussmatPrecision::Float);
        if op.width() < target {
            low = sigma;
        } else {
            high = sigma;
        }
    }
    GaussmatOp::<F32>::new(high, 0.1)
        .unwrap()
        .with_precision(GaussmatPrecision::Float)
}

fn bench_gaussmat(c: &mut Criterion) {
    let mut group = c.benchmark_group("create_gaussmat_f32");
    let scheduler = RayonScheduler::new(RayonScheduler::default_threads()).unwrap();

    for &target in &[512u32, 2048, 8192] {
        group.bench_with_input(
            BenchmarkId::from_parameter(target),
            &target,
            |b, &target| {
                let op = op_for_target_size(target);
                let pixels = vec![0.0f32; op.width() as usize * op.height() as usize];
                b.iter(|| {
                    let source =
                        MemorySource::<F32>::new(op.width(), op.height(), 1, pixels.clone())
                            .unwrap();
                    let pipeline = PipelineBuilder::from_source(source)
                        .then(Box::new(OperationBridge::new_pixel_local(op, 1)))
                        .unwrap()
                        .build()
                        .unwrap();
                    let mut sink = MemorySink::for_pipeline(&pipeline).unwrap();
                    scheduler.run(&pipeline, &mut sink).unwrap();
                    black_box(sink.into_buffer())
                });
            },
        );
    }

    group.finish();
}

criterion_group!(benches, bench_gaussmat);
criterion_main!(benches);
