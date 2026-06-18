#![allow(missing_docs)]
use criterion::{BenchmarkId, Criterion, black_box, criterion_group, criterion_main};
use viprs::{
    adapters::{
        pipeline::PipelineBuilder, scheduler::rayon_scheduler::RayonScheduler,
        sinks::memory::MemorySink, sources::memory::MemorySource,
    },
    domain::{format::U8, ops::conversion::subsample::SubsampleBridge},
    ports::scheduler::TileScheduler,
};

fn bench_subsample(c: &mut Criterion) {
    for (group_name, xfac, yfac, point) in [
        ("subsample_u8_rgba_2x2_line", 2, 2, false),
        ("subsample_u8_rgba_32x32_point", 32, 32, true),
    ] {
        let mut group = c.benchmark_group(group_name);

        for &size in &[512u32, 2048, 8192] {
            let pixels = vec![128u8; size as usize * size as usize * 4];

            group.bench_with_input(BenchmarkId::from_parameter(size), &size, |b, &size| {
                b.iter(|| {
                    let source = MemorySource::<U8>::new(size, size, 4, pixels.clone()).unwrap();
                    let pipeline = PipelineBuilder::from_source(source)
                        .then(Box::new(
                            SubsampleBridge::<U8>::with_point(xfac, yfac, 4, point).unwrap(),
                        ))
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
}

criterion_group!(benches, bench_subsample);
criterion_main!(benches);
