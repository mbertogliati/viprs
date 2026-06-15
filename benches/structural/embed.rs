/// Benchmark: Embed<U8> — insert a source image into a larger canvas.
///
/// Measures the full pipeline path: MemorySource → Embed → MemorySink via
/// RayonScheduler. process_region iterates every output pixel, copying from the
/// source when in-bounds and writing zeros for the border (ExtendMode::Black).
///
/// Three source sizes are benchmarked; the destination canvas is always 2× in each
/// dimension (512→1024, 2048→4096, 8192→16384), with the source centred at offset
/// (src_size/2, src_size/2).
use criterion::{BenchmarkId, Criterion, black_box, criterion_group, criterion_main};
use viprs::{
    adapters::{
        pipeline::PipelineBuilder, scheduler::rayon_scheduler::RayonScheduler,
        sinks::memory::MemorySink, sources::memory::MemorySource,
    },
    domain::{format::U8, ops::structural::embed::ExtendMode},
    ports::scheduler::TileScheduler,
};

fn bench_embed(c: &mut Criterion) {
    let mut group = c.benchmark_group("embed_black_u8");

    for &src_size in &[512u32, 2048, 8192] {
        let dst_size = src_size * 2;
        let x_off = src_size / 2;
        let y_off = src_size / 2;
        let pixel_count = (src_size as usize) * (src_size as usize);
        let pixels = vec![128u8; pixel_count * 3]; // 3-band RGB

        group.bench_with_input(
            BenchmarkId::from_parameter(src_size),
            &src_size,
            |b, &src_size| {
                b.iter(|| {
                    let source =
                        MemorySource::<U8>::new(src_size, src_size, 3, pixels.clone()).unwrap();
                    let pipeline = PipelineBuilder::from_source(source)
                        .embed(
                            dst_size,
                            dst_size,
                            x_off,
                            y_off,
                            src_size,
                            src_size,
                            ExtendMode::Black,
                        )
                        .unwrap()
                        .build()
                        .unwrap();
                    let mut sink = MemorySink::for_pipeline(&pipeline);
                    RayonScheduler::new(RayonScheduler::default_threads())
                        .unwrap()
                        .run(&pipeline, &mut sink)
                        .unwrap();
                    black_box(sink.into_buffer())
                });
            },
        );
    }

    group.finish();
}

criterion_group!(benches, bench_embed);
criterion_main!(benches);
