#![allow(missing_docs)]
use criterion::{BenchmarkId, Criterion, black_box, criterion_group, criterion_main};
use viprs::{
    adapters::{
        scheduler::rayon_scheduler::RayonScheduler, sinks::memory::MemorySink,
        sources::memory::MemorySource,
    },
    domain::format::F32,
    domain::ops::arithmetic::RecombOp,
    domain::ops::arithmetic::recomb::Matrix,
    pipeline::OperationBridge,
    ports::scheduler::TileScheduler,
};
use viprs_runtime::pipeline::internal::PipelinePlan;

fn bench_recomb(c: &mut Criterion) {
    let mut group = c.benchmark_group("recomb_f32");

    #[rustfmt::skip]
    let matrix = Matrix::new(2, 3, vec![
        0.299, 0.587, 0.114,
        1.000, 0.000, -1.000,
    ]);

    for &size in &[512u32, 2048, 8192] {
        let pixel_count = size as usize * size as usize;
        let mut pixels = Vec::with_capacity(pixel_count * 3);
        for i in 0..pixel_count {
            pixels.push((i % 251) as f32 / 255.0);
            pixels.push(((i * 3) % 251) as f32 / 255.0);
            pixels.push(((i * 7) % 251) as f32 / 255.0);
        }

        group.bench_with_input(BenchmarkId::from_parameter(size), &size, |b, &size| {
            b.iter(|| {
                let source = MemorySource::<F32>::new(size, size, 3, pixels.clone()).unwrap();
                let pipeline = PipelinePlan::from_source(source)
                    .append_dyn_op(Box::new(OperationBridge::with_dynamic_bands_pixel_local(
                        RecombOp::<F32>::new(matrix.clone()),
                        3,
                        2,
                    )))
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

criterion_group!(benches, bench_recomb);
criterion_main!(benches);
