#![allow(missing_docs)]
/// Benchmark: MapLut — lookup table pixel mapping.
///
/// Measures the full pipeline path: MemorySource → MapLut → MemorySink via RayonScheduler.
/// Uses an identity LUT (lut[i] = i) to isolate memory bandwidth and dispatch overhead.
use criterion::{BenchmarkId, Criterion, black_box, criterion_group, criterion_main};
use viprs::domain::ops::lut::map_lut::MapLut;
use viprs::{
    adapters::{
        pipeline::ImagePipeline, scheduler::rayon_scheduler::RayonScheduler,
        sinks::memory::MemorySink, sources::memory::MemorySource,
    },
    domain::format::U8,
    domain::op::OperationBridge,
    ports::scheduler::TileScheduler,
};

fn bench_map_lut(c: &mut Criterion) {
    let mut group = c.benchmark_group("map_lut_u8");
    for &size in &[512u32, 2048, 8192] {
        let pixels = vec![128u8; size as usize * size as usize];
        group.bench_with_input(BenchmarkId::from_parameter(size), &size, |b, &size| {
            b.iter(|| {
                let source = MemorySource::<U8>::new(size, size, 1, pixels.clone()).unwrap();
                let op = Box::new(OperationBridge::new(MapLut::identity(), 1u32));
                let pipeline = ImagePipeline::from_source(source)
                    .then(op)
                    .unwrap()
                    .build()
                    .unwrap();
                let mut sink = MemorySink::for_pipeline(&pipeline).unwrap();
                RayonScheduler::new(1)
                    .unwrap()
                    .run(&pipeline, &mut sink)
                    .unwrap();
                black_box(sink.into_buffer())
            });
        });
    }
    group.finish();
}

criterion_group!(benches, bench_map_lut);
criterion_main!(benches);
