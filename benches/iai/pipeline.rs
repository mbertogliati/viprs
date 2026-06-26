#![allow(missing_docs)]
/// iai-callgrind benchmark: deterministic instruction-counted pipeline benchmarks.
///
/// Unlike criterion (wall-clock, noisy), iai-callgrind measures instruction count,
/// cache misses, and branch mispredictions via Valgrind's callgrind tool.
/// Results are perfectly reproducible — no warm-up, no statistical noise.
///
/// Requirements:
///   - Linux (valgrind doesn't support macOS)
///   - `cargo install iai-callgrind-runner`
///   - `valgrind` installed (apt install valgrind)
///
/// Run:
///   cargo bench --bench iai_pipeline
///
/// Or via Docker (from macOS):
///   docker run --rm -v $(pwd):/src -w /src viprs-perf:arm64 \
///     cargo bench --bench iai_pipeline
use iai_callgrind::{library_benchmark, library_benchmark_group, main};
use viprs::adapters::scheduler::rayon_scheduler::RayonScheduler;
use viprs::adapters::sinks::memory::MemorySink;
use viprs::adapters::sources::memory::MemorySource;
use viprs::domain::format::U8;
use viprs::domain::image::DemandHint;
use viprs::domain::op::OperationBridge;
use viprs::domain::ops::arithmetic::add::Add;
use viprs::domain::ops::arithmetic::invert::Invert;
use viprs::ports::scheduler::TileScheduler;
use viprs_runtime::pipeline::internal::PipelineBuilder;

fn tile_samples(size: u32) -> usize {
    let tw = DemandHint::ThinStrip.tile_width(size) as usize;
    let th = DemandHint::ThinStrip.tile_height(size, size) as usize;
    tw * th
}

fn run_invert(size: u32) {
    let pixel_count = (size as usize) * (size as usize);
    let pixels = vec![128u8; pixel_count];

    let source = MemorySource::<U8>::new(size, size, 1, pixels).unwrap();
    let invert_op = Invert::<U8>::new();
    let dyn_op = Box::new(OperationBridge::new(invert_op, 1u32));
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
}

fn run_add(size: u32) {
    let pixel_count = (size as usize) * (size as usize);
    let pixels = vec![128u8; pixel_count];
    let rhs = vec![1u8; tile_samples(size)];

    let source = MemorySource::<U8>::new(size, size, 1, pixels).unwrap();
    let add_op = Add::<U8>::new(rhs);
    let dyn_op = Box::new(OperationBridge::new(add_op, 1u32));
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
}

#[library_benchmark]
#[bench::small(512)]
#[bench::medium(2048)]
#[bench::large(8192)]
fn bench_invert(size: u32) {
    run_invert(size);
}

#[library_benchmark]
#[bench::small(512)]
#[bench::medium(2048)]
#[bench::large(8192)]
fn bench_add(size: u32) {
    run_add(size);
}

library_benchmark_group!(
    name = pipeline_ops;
    benchmarks = bench_invert, bench_add
);

main!(library_benchmark_groups = pipeline_ops);
