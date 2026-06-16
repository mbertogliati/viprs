#![allow(missing_docs)]
use criterion::{BenchmarkId, Criterion, black_box, criterion_group, criterion_main};
use viprs::{
    domain::reducers::HoughLineReducer,
    adapters::{
        pipeline::PipelineBuilder, scheduler::rayon_scheduler::RayonScheduler,
        sinks::memory::MemorySink, sources::memory::MemorySource,
    },
    domain::{format::U8, op::OperationBridge, ops::arithmetic::Linear},
    ports::scheduler::ReducingScheduler,
};

fn make_edge_pixels(size: u32) -> Vec<u8> {
    let mut pixels = vec![0u8; size as usize * size as usize];

    for y in 0..size as usize {
        let diagonal = y * size as usize + y;
        pixels[diagonal] = 255;

        let vertical = y * size as usize + size as usize / 2;
        pixels[vertical] = 255;
    }

    pixels
}

fn bench_hough_line(c: &mut Criterion) {
    let mut group = c.benchmark_group("hough_line_reducer");
    let scheduler = RayonScheduler::new(RayonScheduler::default_threads()).unwrap();

    for &size in &[512u32, 2048, 8192] {
        let pixels = make_edge_pixels(size);

        group.bench_with_input(BenchmarkId::from_parameter(size), &size, |b, &size| {
            b.iter(|| {
                let source = MemorySource::<U8>::new(size, size, 1, pixels.clone()).unwrap();
                let pipeline = PipelineBuilder::from_source(source)
                    .then(Box::new(OperationBridge::new_pixel_local(
                        Linear::<U8>::new(1, 0).unwrap(),
                        1,
                    )))
                    .unwrap()
                    .build()
                    .unwrap();
                let sink = MemorySink::for_pipeline(&pipeline).unwrap();
                let reducer = HoughLineReducer::new(180, 256, size, size, 0.0);

                let hough = scheduler
                    .run_with_reducer::<U8, HoughLineReducer>(&pipeline, &sink, &reducer)
                    .unwrap();

                black_box(hough);
                black_box(sink.into_buffer());
            });
        });
    }

    group.finish();
}

criterion_group!(benches, bench_hough_line);
criterion_main!(benches);
