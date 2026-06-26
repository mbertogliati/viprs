#![allow(missing_docs)]
use criterion::{BenchmarkId, Criterion, black_box, criterion_group, criterion_main};
use viprs::{
    adapters::{
        pipeline::internal::PipelinePlan, scheduler::rayon_scheduler::RayonScheduler,
        sinks::memory::MemorySink, sources::memory::MemorySource,
    },
    domain::{
        colorspace::{ColorspaceId, Lch},
        format::F32,
    },
    ports::scheduler::TileScheduler,
};

fn bench_lab_to_lch(c: &mut Criterion) {
    let mut group = c.benchmark_group("lab_to_lch_f32");

    for &size in &[512u32, 2048, 8192] {
        let pixels = vec![50.0_f32; size as usize * size as usize * 3];

        group.bench_with_input(BenchmarkId::from_parameter(size), &size, |b, &size| {
            b.iter(|| {
                let source = MemorySource::<F32>::new(size, size, 3, pixels.clone()).unwrap();
                let pipeline = PipelinePlan::from_source(source)
                    .with_colorspace(ColorspaceId::Lab)
                    .plan_colourspace::<Lch>()
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

criterion_group!(benches, bench_lab_to_lch);
criterion_main!(benches);
