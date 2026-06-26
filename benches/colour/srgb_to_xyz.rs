#![allow(missing_docs)]
/// Benchmark: SRgbToXyz — convert a U8 sRGB image to CIE XYZ (F32 output).
///
/// Measures the full pipeline path: MemorySource<U8> → SRgbToXyz → MemorySink<F32>
/// via RayonScheduler. The conversion is pixel-local (no halo, no resampling), so
/// throughput scales linearly with pixel count and thread count.
use criterion::{BenchmarkId, Criterion, black_box, criterion_group, criterion_main};
use viprs::{
    adapters::{
        pipeline::ImagePipeline, scheduler::rayon_scheduler::RayonScheduler,
        sinks::memory::MemorySink, sources::memory::MemorySource,
    },
    domain::{
        colorspace::{ColorspaceId, Xyz},
        format::U8,
    },
    ports::scheduler::TileScheduler,
};

fn bench_srgb_to_xyz(c: &mut Criterion) {
    let mut group = c.benchmark_group("srgb_to_xyz_u8");

    for &size in &[512u32, 2048, 8192] {
        let pixels = vec![128u8; size as usize * size as usize * 3];

        group.bench_with_input(BenchmarkId::from_parameter(size), &size, |b, &size| {
            b.iter(|| {
                let source = MemorySource::<U8>::new(size, size, 3, pixels.clone()).unwrap();
                let pipeline = ImagePipeline::from_source(source)
                    .with_colorspace(ColorspaceId::SRgb)
                    .colourspace::<Xyz>()
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

criterion_group!(benches, bench_srgb_to_xyz);
criterion_main!(benches);
