#![allow(missing_docs)]
/// Benchmark: Rotate270<U8> — rotate image 270° clockwise.
///
/// Measures the full pipeline path: MemorySource → Rotate270 → MemorySink via
/// RayonScheduler. Like Rotate90, this transposes geometry while preserving bytes
/// per sample and band count.
use criterion::{BenchmarkId, Criterion, black_box, criterion_group, criterion_main};
use viprs::{
    adapters::{
      pipeline::ImagePipeline, scheduler::rayon_scheduler::RayonScheduler,
      sinks::memory::MemorySink, sources::memory::MemorySource,
    },
    domain::format::U8,
    ports::scheduler::TileScheduler,
};

fn bench_rotate270(c: &mut Criterion) {
    let mut group = c.benchmark_group("rotate270_u8");

    for &size in &[512u32, 2048, 8192] {
        let pixel_count = (size as usize) * (size as usize);
        let pixels = vec![128u8; pixel_count];

        group.bench_with_input(BenchmarkId::from_parameter(size), &size, |b, &size| {
            b.iter(|| {
                let source = MemorySource::<U8>::new(size, size, 1, pixels.clone()).unwrap();
                let pipeline = ImagePipeline::from_source(source)
                    .rotate270()
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

criterion_group!(benches, bench_rotate270);
criterion_main!(benches);
