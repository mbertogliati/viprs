#![allow(missing_docs)]
use criterion::{BenchmarkId, Criterion, black_box, criterion_group, criterion_main};
use viprs::{
    adapters::{
        pipeline::PipelineArena, scheduler::rayon_scheduler::RayonScheduler,
        sinks::memory::MemorySink, sources::memory::MemorySource,
    },
    domain::{format::BandFormatId, format::F32, ops::resample::MapImOp},
    ports::scheduler::TileScheduler,
};

fn source_pixels(size: u32) -> Vec<f32> {
    let mut pixels = Vec::with_capacity(size as usize * size as usize);
    for y in 0..size {
        for x in 0..size {
            pixels.push(((x + y) % 251) as f32);
        }
    }
    pixels
}

fn index_pixels(size: u32) -> Vec<f32> {
    let mut pixels = Vec::with_capacity(size as usize * size as usize * 2);
    for y in 0..size {
        for x in 0..size {
            pixels.push(x as f32);
            pixels.push(y as f32);
        }
    }
    pixels
}

fn bench_mapim(c: &mut Criterion) {
    let mut group = c.benchmark_group("mapim_f32_roots");
    for &size in &[512u32, 2048, 8192] {
        let source_pixels = source_pixels(size);
        let index_pixels = index_pixels(size);
        group.bench_with_input(BenchmarkId::from_parameter(size), &size, |b, &size| {
            b.iter(|| {
                let source =
                    MemorySource::<F32>::new(size, size, 1, source_pixels.clone()).unwrap();
                let index = MemorySource::<F32>::new(size, size, 2, index_pixels.clone()).unwrap();

                let mut arena = PipelineArena::with_source(Box::new(source));
                let index_root = arena.add_root_source(Box::new(index));
                let node = arena.add_node(Box::new(MapImOp::<F32>::new(
                    size,
                    size,
                    1,
                    size,
                    size,
                    BandFormatId::F32,
                )));
                arena.connect_root_to_slot(0, node, 0).unwrap();
                arena.connect_root_to_slot(index_root, node, 1).unwrap();
                let pipeline = arena.compile().unwrap();

                let mut sink = MemorySink::for_pipeline(&pipeline).unwrap();
                RayonScheduler::new(RayonScheduler::default_threads())
                    .unwrap()
                    .run(&pipeline, &mut sink)
                    .unwrap();
                black_box(sink.into_buffer());
            });
        });
    }
    group.finish();
}

criterion_group!(benches, bench_mapim);
criterion_main!(benches);
