/// Benchmark: Divide<F32> — element-wise division by a per-pixel rhs buffer.
///
/// Measures the full pipeline path: MemorySource → Divide → MemorySink via RayonScheduler.
/// rhs covers the full image (pixel_count × 1 band) because Divide::process_region
/// zips input samples with rhs linearly across the full image extent.
use criterion::{BenchmarkId, Criterion, black_box, criterion_group, criterion_main};
use viprs::domain::ops::arithmetic::divide::Divide;
use viprs::{
    adapters::{
        pipeline::PipelineBuilder, scheduler::rayon_scheduler::RayonScheduler,
        sinks::memory::MemorySink, sources::memory::MemorySource,
    },
    domain::format::F32,
    domain::op::OperationBridge,
    ports::scheduler::TileScheduler,
};

fn bench_divide(c: &mut Criterion) {
    let mut group = c.benchmark_group("divide_f32");

    for &size in &[512u32, 2048, 8192] {
        let pixel_count = (size as usize) * (size as usize);
        let pixels = vec![4.0f32; pixel_count];

        group.bench_with_input(BenchmarkId::from_parameter(size), &size, |b, &size| {
            b.iter(|| {
                let rhs = vec![2.0f32; size as usize * size as usize];
                let source = MemorySource::<F32>::new(size, size, 1, pixels.clone()).unwrap();
                let div_op = Divide::<F32>::new(rhs, size, 1);
                let dyn_op = Box::new(OperationBridge::new(div_op, 1u32));
                let pipeline = PipelineBuilder::from_source(source)
                    .then(dyn_op)
                    .unwrap()
                    .build()
                    .unwrap();
                let mut sink = MemorySink::for_pipeline(&pipeline);
                RayonScheduler::new(1)
                    .unwrap()
                    .run(&pipeline, &mut sink)
                    .unwrap();
                black_box(sink.into_buffer())
            });
        });
    }

    group.finish();
}

criterion_group!(benches, bench_divide);
criterion_main!(benches);
