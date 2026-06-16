#![allow(missing_docs)]
use criterion::{BenchmarkId, Criterion, black_box, criterion_group, criterion_main};
use viprs::{
    adapters::{
        pipeline::PipelineArena, scheduler::rayon_scheduler::RayonScheduler,
        sinks::memory::MemorySink, sources::memory::MemorySource,
    },
    domain::{format::U8, image::Image, ops::conversion::SmartcropOp},
    ports::scheduler::TileScheduler,
};

fn bench_smartcrop(c: &mut Criterion) {
    let mut group = c.benchmark_group("smartcrop_u8");

    for &size in &[512u32, 2048, 8192] {
        let mut pixels = vec![32u8; size as usize * size as usize];
        let block = (size / 8).max(1);
        let start = size / 3;
        for y in start..(start + block).min(size) {
            for x in start..(start + block).min(size) {
                let idx = y as usize * size as usize + x as usize;
                pixels[idx] = ((x ^ y) & 0xff) as u8;
            }
        }

        group.bench_with_input(BenchmarkId::from_parameter(size), &size, |b, &size| {
            b.iter(|| {
                let image = Image::<U8>::from_buffer(size, size, 1, pixels.clone()).unwrap();
                let op = SmartcropOp::analyze(&image, size / 2, size / 2);
                let source = MemorySource::<U8>::new(size, size, 1, pixels.clone()).unwrap();
                let mut arena = PipelineArena::with_source(Box::new(source));
                arena.add_view_node(Box::new(op.into_bridge(1)));
                let pipeline = arena.compile().unwrap();
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

criterion_group!(benches, bench_smartcrop);
criterion_main!(benches);
