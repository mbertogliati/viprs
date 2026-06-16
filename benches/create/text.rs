#![allow(missing_docs)]
use criterion::{BenchmarkId, Criterion, black_box, criterion_group, criterion_main};
use viprs::{
    adapters::{
        pipeline::PipelineBuilder, scheduler::rayon_scheduler::RayonScheduler,
        sinks::memory::MemorySink, sources::TextSource,
    },
    ports::scheduler::TileScheduler,
};

fn bench_text(c: &mut Criterion) {
    let mut group = c.benchmark_group("create_text_rgba");
    let scheduler = RayonScheduler::new(RayonScheduler::default_threads()).unwrap();

    for &(size, font_size) in &[(512u32, 32.0f32), (2048, 96.0), (8192, 256.0)] {
        group.bench_with_input(BenchmarkId::from_parameter(size), &size, |b, &_size| {
            b.iter(|| {
                let source = TextSource::new(
                    "viprs text source benchmark",
                    font_size,
                    [255, 255, 255, 255],
                    None::<&str>,
                )
                .unwrap();
                let pipeline = PipelineBuilder::from_source(source).build().unwrap();
                let mut sink = MemorySink::for_pipeline(&pipeline).unwrap();
                scheduler.run(&pipeline, &mut sink).unwrap();
                black_box(sink.into_buffer())
            });
        });
    }

    group.finish();
}

criterion_group!(benches, bench_text);
criterion_main!(benches);
