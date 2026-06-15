use criterion::{BenchmarkId, Criterion, black_box, criterion_group, criterion_main};
use viprs::{
    adapters::{
        pipeline::PipelineBuilder, scheduler::rayon_scheduler::RayonScheduler,
        sinks::memory::MemorySink, sources::memory::MemorySource,
    },
    domain::{format::F32, op::OperationBridge, ops::convolution::ConvOp},
    ports::scheduler::TileScheduler,
};

fn box_3x3_kernel() -> Vec<Vec<f64>> {
    let weight = 1.0 / 9.0;
    vec![
        vec![weight, weight, weight],
        vec![weight, weight, weight],
        vec![weight, weight, weight],
    ]
}

fn bench_conv(c: &mut Criterion) {
    let mut group = c.benchmark_group("conv_f32_box3x3");

    for &size in &[512u32, 2048, 8192] {
        let pixels = vec![1.0f32; size as usize * size as usize];

        group.bench_with_input(BenchmarkId::from_parameter(size), &size, |b, &size| {
            b.iter(|| {
                let source = MemorySource::<F32>::new(size, size, 1, pixels.clone()).unwrap();
                let op = ConvOp::<F32>::new(box_3x3_kernel()).unwrap();
                let pipeline = PipelineBuilder::from_source(source)
                    .then(Box::new(OperationBridge::new(op, 1)))
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
}

criterion_group!(benches, bench_conv);
criterion_main!(benches);
