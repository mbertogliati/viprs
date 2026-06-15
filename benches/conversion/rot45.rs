use criterion::{BenchmarkId, Criterion, black_box, criterion_group, criterion_main};
use viprs::{
    adapters::{
        pipeline::PipelineBuilder, scheduler::rayon_scheduler::RayonScheduler,
        sinks::memory::MemorySink, sources::memory::MemorySource,
    },
    domain::{format::U8, ops::conversion::rot45::Angle45},
    ports::scheduler::TileScheduler,
};

const ODD_SQUARE_SIZES: [u32; 3] = [513, 2049, 8193];
const BENCH_ANGLES: [(Angle45, &str); 2] = [(Angle45::D45, "d45"), (Angle45::D135, "d135")];

fn must<T, E: std::fmt::Display>(result: Result<T, E>, context: &str) -> T {
    match result {
        Ok(value) => value,
        Err(error) => panic!("{context}: {error}"),
    }
}

fn bench_rot45(c: &mut Criterion) {
    let mut group = c.benchmark_group("conversion_rot45_u8");
    let scheduler = must(
        RayonScheduler::new(RayonScheduler::default_threads()),
        "create rayon scheduler",
    );

    for &size in &ODD_SQUARE_SIZES {
        let pixels = (0..size as usize * size as usize)
            .map(|idx| (idx % 251) as u8)
            .collect::<Vec<_>>();

        for &(angle, label) in &BENCH_ANGLES {
            group.bench_with_input(BenchmarkId::new(label, size), &size, |b, &size| {
                b.iter(|| {
                    let source = must(
                        MemorySource::<U8>::new(size, size, 1, pixels.clone()),
                        "create memory source",
                    );
                    let builder = must(
                        PipelineBuilder::from_source(source).rot45(angle),
                        "add rot45 operation",
                    );
                    let pipeline = must(builder.build(), "build pipeline");
                    let mut sink = MemorySink::for_pipeline(&pipeline);
                    must(scheduler.run(&pipeline, &mut sink), "run pipeline");
                    black_box(sink.into_buffer())
                });
            });
        }
    }

    group.finish();
}

criterion_group!(benches, bench_rot45);
criterion_main!(benches);
