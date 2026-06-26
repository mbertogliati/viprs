#![allow(missing_docs)]
/// Benchmark: ExtractArea<U8> — crop a half-size region from the image center.
///
/// Measures the full pipeline path: MemorySource → ExtractArea → MemorySink via
/// RayonScheduler. The extracted region starts at (size/4, size/4) with
/// dimensions (size/2, size/2), covering a quarter of the total pixel area.
///
/// Note: process_region is currently a copy_from_slice (see B-021).
/// When B-021 (zero-copy structural ops) is resolved, this benchmark provides the
/// baseline to quantify the improvement.
use criterion::{BenchmarkId, Criterion, black_box, criterion_group, criterion_main};
use viprs::{
    adapters::{
        pipeline::ImagePipeline, scheduler::rayon_scheduler::RayonScheduler,
        sinks::memory::MemorySink, sources::memory::MemorySource,
    },
    domain::format::U8,
    ports::scheduler::TileScheduler,
};

fn bench_extract_area(c: &mut Criterion) {
    let mut group = c.benchmark_group("extract_area_u8");

    for &size in &[512u32, 2048, 8192] {
        let pixel_count = (size as usize) * (size as usize);
        let pixels = vec![128u8; pixel_count];

        group.bench_with_input(BenchmarkId::from_parameter(size), &size, |b, &size| {
            b.iter(|| {
                let source = MemorySource::<U8>::new(size, size, 1, pixels.clone()).unwrap();
                // Crop the center half: start at (size/4, size/4), size/2 × size/2.
                let x = size / 4;
                let y = size / 4;
                let w = size / 2;
                let h = size / 2;
                let pipeline = ImagePipeline::from_source(source)
                    .extract_area(x, y, w, h)
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

criterion_group!(benches, bench_extract_area);
criterion_main!(benches);
