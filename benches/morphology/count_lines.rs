#![allow(missing_docs)]
use criterion::{BenchmarkId, Criterion, black_box, criterion_group, criterion_main};
use viprs::{
    pipeline::OperationBridge,
    adapters::{
        pipeline::internal::PipelinePlan, scheduler::rayon_scheduler::RayonScheduler,
        sinks::memory::MemorySink, sources::memory::MemorySource,
    },
    domain::{format::U8, ops::morphology::count_lines::{CountLinesOp, CountLinesDirection}},
    ports::scheduler::TileScheduler,
};

fn bench_count_lines(c: &mut Criterion) {
    let mut group = c.benchmark_group("count_lines_u8");

    for &size in &[512u32, 2048, 8192] {
        let pixels: Vec<u8> = (0..size as usize * size as usize)
            .map(|idx| u8::from((idx / 32) % 2 == 0))
            .collect();

        group.bench_with_input(BenchmarkId::from_parameter(size), &size, |b, &size| {
            b.iter(|| {
                let source = MemorySource::<U8>::new(size, size, 1, pixels.clone()).unwrap();
                let pipeline = PipelinePlan::from_source(source)
                    .append_dyn_op(Box::new(OperationBridge::new(
                        CountLinesOp::new(size, size, CountLinesDirection::Horizontal),
                        1,
                    )))
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
        });
    }

    group.finish();
}

criterion_group!(benches, bench_count_lines);
criterion_main!(benches);
