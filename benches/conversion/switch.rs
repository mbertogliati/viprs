use criterion::{BenchmarkId, Criterion, black_box, criterion_group, criterion_main};
use viprs::{
    PipelineBuilder, SwitchOp, U8,
    adapters::{
        scheduler::rayon_scheduler::RayonScheduler, sinks::memory::MemorySink,
        sources::memory::MemorySource,
    },
    ports::scheduler::TileScheduler,
};

const CHOICE_COUNT: u32 = 3;
const CANDIDATE_BANDS: u32 = 3;

fn make_pixels(size: u32) -> Vec<u8> {
    let pixel_count = size as usize * size as usize;
    let combined_bands = 1 + CHOICE_COUNT as usize * CANDIDATE_BANDS as usize;
    let mut pixels = Vec::with_capacity(pixel_count * combined_bands);

    for idx in 0..pixel_count {
        pixels.push((idx % CHOICE_COUNT as usize) as u8);
        for choice in 0..CHOICE_COUNT as usize {
            for band in 0..CANDIDATE_BANDS as usize {
                pixels.push(((idx + choice * 17 + band * 31) % 251) as u8);
            }
        }
    }

    pixels
}

fn bench_switch(c: &mut Criterion) {
    let mut group = c.benchmark_group("switch_u8");

    for &size in &[512u32, 2048, 8192] {
        let pixels = make_pixels(size);

        group.bench_with_input(BenchmarkId::from_parameter(size), &size, |b, &size| {
            b.iter(|| {
                let op = SwitchOp::<U8>::new(CHOICE_COUNT, CANDIDATE_BANDS);
                let source =
                    MemorySource::<U8>::new(size, size, op.combined_bands(), pixels.clone())
                        .unwrap();
                let pipeline = PipelineBuilder::from_source(source)
                    .then(Box::new(op))
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
        });
    }

    group.finish();
}

criterion_group!(benches, bench_switch);
criterion_main!(benches);
