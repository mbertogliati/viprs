#![allow(missing_docs)]
/// Benchmark: Conv2d<F32> with a 3×3 box filter.
///
/// Measures the full pipeline path: MemorySource<F32> → Conv2d (3×3 box) → MemorySink
/// via RayonScheduler. SmallTile tiles are 128×128.
use criterion::{BenchmarkId, Criterion, black_box, criterion_group, criterion_main};
use viprs::{
    adapters::{
      pipeline::ImagePipeline, scheduler::rayon_scheduler::RayonScheduler,
      sinks::memory::MemorySink, sources::memory::MemorySource,
    },
    domain::format::F32,
    ports::scheduler::TileScheduler,
};

fn box_3x3_kernel() -> Vec<Vec<f64>> {
    let w = 1.0f64 / 9.0;
    vec![vec![w, w, w], vec![w, w, w], vec![w, w, w]]
}

fn bench_conv2d(c: &mut Criterion) {
    let mut group = c.benchmark_group("conv2d_f32_box3x3");

    for &size in &[512u32, 2048, 8192] {
        let pixel_count = (size as usize) * (size as usize);
        let pixels = vec![1.0f32; pixel_count];

        group.bench_with_input(BenchmarkId::from_parameter(size), &size, |b, &size| {
            b.iter(|| {
                let source = MemorySource::<F32>::new(size, size, 1, pixels.clone()).unwrap();
                let pipeline = ImagePipeline::from_source(source)
                    .conv2d(box_3x3_kernel())
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

criterion_group!(benches, bench_conv2d);
criterion_main!(benches);
