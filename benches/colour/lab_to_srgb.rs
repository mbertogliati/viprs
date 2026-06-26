#![allow(missing_docs)]
/// Benchmark: LabToSRgb — convert a F32 CIE L*a*b* image to sRGB (U8 output).
///
/// Measures the full pipeline path: MemorySource<F32> → LabToSRgb → MemorySink<U8>
/// via RayonScheduler. The conversion is pixel-local (no halo, no resampling), so
/// throughput scales linearly with pixel count and thread count.
use criterion::{BenchmarkId, Criterion, black_box, criterion_group, criterion_main};
use viprs::{
    adapters::{
        pipeline::internal::PipelinePlan, scheduler::rayon_scheduler::RayonScheduler,
        sinks::memory::MemorySink, sources::memory::MemorySource,
    },
    domain::{
        colorspace::{ColorspaceId, SRgb},
        format::F32,
    },
    ports::scheduler::TileScheduler,
};

fn bench_lab_to_srgb(c: &mut Criterion) {
    let mut group = c.benchmark_group("lab_to_srgb_f32");

    for &size in &[512u32, 2048, 8192] {
        let pixels = vec![50.0f32; size as usize * size as usize * 3];

        group.bench_with_input(BenchmarkId::from_parameter(size), &size, |b, &size| {
            b.iter(|| {
                let source = MemorySource::<F32>::new(size, size, 3, pixels.clone()).unwrap();
                let pipeline = PipelinePlan::from_source(source)
                    .with_colorspace(ColorspaceId::Lab)
                    .plan_colourspace::<SRgb>()
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

criterion_group!(benches, bench_lab_to_srgb);
criterion_main!(benches);
