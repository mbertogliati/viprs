#![allow(missing_docs)]
/// Benchmark: Rotate90<U8> — rotate image 90° clockwise.
///
/// Measures the full pipeline path: MemorySource → Rotate90 → MemorySink via
/// RayonScheduler. process_region performs a transpose-with-reversal copy — O(pixels)
/// work with no extra allocation.
///
/// Note: Rotate90 allocates per-node buffers sized to tile_h × tile_w for input
/// and tile_w × tile_h for output, as declared by NodeSpec. This is the baseline
/// cost for any geometry-changing structural op.
use criterion::{BenchmarkId, Criterion, black_box, criterion_group, criterion_main};
use viprs::{
    adapters::{
        pipeline::internal::PipelinePlan, scheduler::rayon_scheduler::RayonScheduler,
        sinks::memory::MemorySink, sources::memory::MemorySource,
    },
    domain::format::U8,
    ports::scheduler::TileScheduler,
};

fn bench_rotate90(c: &mut Criterion) {
    let mut group = c.benchmark_group("rotate90_u8");

    for &size in &[512u32, 2048, 8192] {
        let pixel_count = (size as usize) * (size as usize);
        let pixels = vec![128u8; pixel_count];

        group.bench_with_input(BenchmarkId::from_parameter(size), &size, |b, &size| {
            b.iter(|| {
                let source = MemorySource::<U8>::new(size, size, 1, pixels.clone()).unwrap();
                let pipeline = PipelinePlan::from_source(source)
                    .rotate90()
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

criterion_group!(benches, bench_rotate90);
criterion_main!(benches);
