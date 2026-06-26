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
        ops::convolution::{ConvSepH, ConvSepV},
    },
    ports::scheduler::TileScheduler,
};

fn bench_convsep(c: &mut Criterion) {
    let kernel = vec![0.25, 0.5, 0.25];
    let mut group = c.benchmark_group("convsep_f32");

    for &size in &[512u32, 2048, 8192] {
        let pixel_count = (size as usize) * (size as usize);
        let pixels = vec![1.0f32; pixel_count];

        group.bench_with_input(BenchmarkId::from_parameter(size), &size, |b, &size| {
            b.iter(|| {
                let source = MemorySource::<F32>::new(size, size, 1, pixels.clone()).unwrap();
                let h_bridge =
                    OperationBridge::new(ConvSepH::<F32>::new(kernel.clone()).unwrap(), 1u32);
                let v_bridge =
                    OperationBridge::new(ConvSepV::<F32>::new(kernel.clone()).unwrap(), 1u32);

                let pipeline = PipelineBuilder::from_source(source)
                    .then(Box::new(h_bridge))
                    .unwrap()
                    .then(Box::new(v_bridge))
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

criterion_group!(benches, bench_convsep);
criterion_main!(benches);
