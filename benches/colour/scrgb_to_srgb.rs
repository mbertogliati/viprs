#![allow(missing_docs)]
use criterion::{BenchmarkId, Criterion, black_box, criterion_group, criterion_main};
use viprs::{
    adapters::{
      pipeline::ImagePipeline, scheduler::rayon_scheduler::RayonScheduler,
      sinks::memory::MemorySink, sources::memory::MemorySource,
    },
    domain::{
        colorspace::{ColorspaceId, SRgb},
        format::F32,
    },
    ports::scheduler::TileScheduler,
};

fn bench_scrgb_to_srgb(c: &mut Criterion) {
    let mut group = c.benchmark_group("scrgb_to_srgb_f32");

    for &size in &[512u32, 2048, 8192] {
        let pixels = vec![0.5_f32; size as usize * size as usize * 3];

        group.bench_with_input(BenchmarkId::from_parameter(size), &size, |b, &size| {
            b.iter(|| {
                let source = MemorySource::<F32>::new(size, size, 3, pixels.clone()).unwrap();
                let pipeline = ImagePipeline::from_source(source)
                    .with_colorspace(ColorspaceId::ScRgb)
                    .colourspace::<SRgb>()
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

criterion_group!(benches, bench_scrgb_to_srgb);
criterion_main!(benches);
