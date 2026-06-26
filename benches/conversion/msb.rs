#![allow(missing_docs)]
use criterion::{BenchmarkId, Criterion, black_box, criterion_group, criterion_main};
use viprs::{
  adapters::{
        scheduler::rayon_scheduler::RayonScheduler, sinks::memory::MemorySink,
        sources::memory::MemorySource,
    },
  domain::format::U32,
  pipeline::ImagePipeline,
  ports::scheduler::TileScheduler,
};

fn must<T, E: std::fmt::Display>(result: Result<T, E>, context: &str) -> T {
    match result {
        Ok(value) => value,
        Err(error) => panic!("{context}: {error}"),
    }
}

fn bench_msb(c: &mut Criterion) {
    let mut group = c.benchmark_group("conversion_msb_u32_to_u8");

    for &size in &[512u32, 2048, 8192] {
        let pixels = (0..size as usize * size as usize)
            .map(|index| (index as u32).wrapping_mul(1_103_515_245))
            .collect::<Vec<_>>();

        group.bench_with_input(BenchmarkId::from_parameter(size), &size, |b, &size| {
            b.iter(|| {
                let source = must(
                    MemorySource::<U32>::new(size, size, 1, pixels.clone()),
                    "create memory source",
                );
                let pipeline = must(
                  ImagePipeline::from_source(source).msb(),
                  "add msb operation",
                )
                .build()
                .unwrap();
                let mut sink = MemorySink::for_pipeline(&pipeline).unwrap();
                let scheduler = must(
                    RayonScheduler::new(RayonScheduler::default_threads()),
                    "create rayon scheduler",
                );
                must(scheduler.run(&pipeline, &mut sink), "run pipeline");
                black_box(sink.into_buffer())
            });
        });
    }

    group.finish();
}

criterion_group!(benches, bench_msb);
criterion_main!(benches);
