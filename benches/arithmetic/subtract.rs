#![allow(missing_docs)]
/// Benchmark: Subtract<U8> — element-wise subtraction of a constant rhs buffer.
///
/// Measures the full pipeline path: MemorySource → Subtract → MemorySink via RayonScheduler.
/// rhs is sized to one tile (tile_w × tile_h × 1 band). ThinStrip tiles are
/// image_width × 16 for images ≤ 10 000 px wide.
use criterion::{BenchmarkId, Criterion, black_box, criterion_group, criterion_main};
use viprs::domain::ops::arithmetic::subtract::Subtract;
use viprs::{
    adapters::{
        pipeline::internal::PipelineBuilder, scheduler::rayon_scheduler::RayonScheduler,
        sinks::memory::MemorySink, sources::memory::MemorySource,
    },
    domain::op::OperationBridge,
    domain::{format::U8, image::DemandHint},
    ports::scheduler::TileScheduler,
};

fn tile_samples(size: u32) -> usize {
    let tw = DemandHint::ThinStrip.tile_width(size) as usize;
    let th = DemandHint::ThinStrip.tile_height(size, size) as usize;
    tw * th
}

fn bench_subtract(c: &mut Criterion) {
    let mut group = c.benchmark_group("subtract_u8");

    for &size in &[512u32, 2048, 8192] {
        let pixel_count = (size as usize) * (size as usize);
        // Source pixels ≥ rhs to avoid wrapping underflow in U8.
        let pixels = vec![200u8; pixel_count];
        let rhs = vec![1u8; tile_samples(size)];

        group.bench_with_input(BenchmarkId::from_parameter(size), &size, |b, &size| {
            b.iter(|| {
                let source = MemorySource::<U8>::new(size, size, 1, pixels.clone()).unwrap();
                let sub_op = Subtract::<U8>::new(rhs.clone());
                let dyn_op = Box::new(OperationBridge::new(sub_op, 1u32));
                let pipeline = PipelineBuilder::from_source(source)
                    .then(dyn_op)
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

criterion_group!(benches, bench_subtract);
criterion_main!(benches);
