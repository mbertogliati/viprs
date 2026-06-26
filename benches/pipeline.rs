#![allow(missing_docs)]
/// Benchmark: complete pipeline — MemorySource → Linear → Invert → MemorySink.
///
/// Exercises the full Source → multi-op → Sink path via RayonScheduler.
/// Three sizes are required by P6 (GUIDELINES.md): 512×512, 2048×2048, 8192×8192.
/// Both operations are pixel-local (U8 format), so DemandHint::ThinStrip governs
/// tile geometry.
use criterion::{BenchmarkId, Criterion, black_box, criterion_group, criterion_main};
use viprs::{
    adapters::{
        pipeline::internal::PipelinePlan, scheduler::rayon_scheduler::RayonScheduler,
        sinks::memory::MemorySink, sources::memory::MemorySource,
    },
    domain::format::U8,
    ports::scheduler::TileScheduler,
};

fn bench_pipeline(c: &mut Criterion) {
    let mut group = c.benchmark_group("pipeline_linear_invert_u8");

    for &size in &[512u32, 2048, 8192] {
        let pixel_count = (size as usize) * (size as usize);
        let pixels = vec![100u8; pixel_count];

        group.bench_with_input(BenchmarkId::from_parameter(size), &size, |b, &size| {
            b.iter(|| {
                let source = MemorySource::<U8>::new(size, size, 1, pixels.clone()).unwrap();
                // Two-stage pipeline: linear(1, 10) then invert.
                let pipeline = PipelinePlan::from_source(source)
                    .plan_linear(1.0, 10.0)
                    .unwrap()
                    .plan_invert()
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

criterion_group!(benches, bench_pipeline);
criterion_main!(benches);
