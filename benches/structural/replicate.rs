#![allow(missing_docs)]
/// Benchmark: Replicate<U8> — tile the source image 2×2.
///
/// Source sizes [256, 1024, 4096] produce output sizes [512, 2048, 8192], keeping
/// the benchmark on the standard destination sizes without exceeding the usual 8192² cap.
use criterion::{BenchmarkId, Criterion, black_box, criterion_group, criterion_main};
use viprs::{
    adapters::{
        pipeline::internal::PipelinePlan, scheduler::rayon_scheduler::RayonScheduler,
        sinks::memory::MemorySink, sources::memory::MemorySource,
    },
    domain::format::U8,
    ports::scheduler::TileScheduler,
};

fn bench_replicate(c: &mut Criterion) {
    let mut group = c.benchmark_group("replicate_2x2_u8");

    for &src_size in &[256u32, 1024, 4096] {
        let dst_size = src_size * 2;
        let pixel_count = (src_size as usize) * (src_size as usize);
        let pixels = vec![128u8; pixel_count];

        group.bench_with_input(
            BenchmarkId::from_parameter(dst_size),
            &src_size,
            |b, &src_size| {
                b.iter(|| {
                    let source =
                        MemorySource::<U8>::new(src_size, src_size, 1, pixels.clone()).unwrap();
                    let pipeline = PipelinePlan::from_source(source)
                        .replicate(2, 2)
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
            },
        );
    }

    group.finish();
}

criterion_group!(benches, bench_replicate);
criterion_main!(benches);
