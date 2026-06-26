#![allow(missing_docs)]
use std::f32::consts::PI;
use viprs_runtime::pipeline::internal::PipelinePlan;

use criterion::{BenchmarkId, Criterion, black_box, criterion_group, criterion_main};
use viprs::{
    adapters::{
        scheduler::rayon_scheduler::RayonScheduler, sinks::memory::MemorySink,
        sources::memory::MemorySource,
    },
    domain::format::F32,
    domain::ops::arithmetic::PolarOp,
    pipeline::OperationBridge,
    ports::scheduler::TileScheduler,
};

fn make_pixels(size: u32) -> Vec<f32> {
    let pixel_count = size as usize * size as usize;
    let mut pixels = Vec::with_capacity(pixel_count * 2);
    for idx in 0..pixel_count {
        let re = (idx % 257) as f32 - 128.0;
        let im = (((idx * 7) % 257) as f32 - 128.0) * (PI / 256.0);
        pixels.push(re);
        pixels.push(im);
    }
    pixels
}

fn bench_polar(c: &mut Criterion) {
    let mut group = c.benchmark_group("polar_f32");

    for &size in &[512u32, 2048, 8192] {
        let pixels = make_pixels(size);

        group.bench_with_input(BenchmarkId::from_parameter(size), &size, |b, &size| {
            b.iter(|| {
                let source = MemorySource::<F32>::new(size, size, 2, pixels.clone()).unwrap();
                let pipeline = PipelinePlan::from_source(source)
                    .append_dyn_op(Box::new(OperationBridge::new_pixel_local(PolarOp, 2)))
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

criterion_group!(benches, bench_polar);
criterion_main!(benches);
