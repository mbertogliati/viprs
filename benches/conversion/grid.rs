/// Benchmark: Grid<U8> — rearrange a vertical strip of four frames into a 2×2 grid.
///
/// Measures the full pipeline path: MemorySource → Grid → MemorySink via RayonScheduler.
/// Output sizes follow the standard 512 / 2048 / 8192 dimensions.
use criterion::{BenchmarkId, Criterion, black_box, criterion_group, criterion_main};
use viprs::{
    adapters::{
        pipeline::PipelineBuilder, scheduler::rayon_scheduler::RayonScheduler,
        sinks::memory::MemorySink, sources::memory::MemorySource,
    },
    domain::format::U8,
    ports::scheduler::TileScheduler,
};

fn bench_grid(c: &mut Criterion) {
    let mut group = c.benchmark_group("grid_u8");

    for &output_size in &[512u32, 2048, 8192] {
        let src_width = output_size / 2;
        let tile_height = output_size / 2;
        let frame_count = 4u32;
        let input_height = tile_height * frame_count;
        let pixels = vec![127u8; (src_width * input_height) as usize];

        group.bench_with_input(
            BenchmarkId::from_parameter(output_size),
            &output_size,
            |b, &_output_size| {
                b.iter(|| {
                    let source =
                        MemorySource::<U8>::new(src_width, input_height, 1, pixels.clone())
                            .unwrap();
                    let pipeline = PipelineBuilder::from_source(source)
                        .grid(tile_height, 2)
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

criterion_group!(benches, bench_grid);
criterion_main!(benches);
