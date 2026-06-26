#![allow(missing_docs)]
use criterion::{BenchmarkId, Criterion, black_box, criterion_group, criterion_main};
use viprs::{
    adapters::{
        pipeline::internal::PipelinePlan, scheduler::rayon_scheduler::RayonScheduler,
        sinks::memory::MemorySink, sources::memory::MemorySource,
    },
    domain::{format::F32, op::OperationBridge, ops::create::FrequencyMaskOp},
    ports::scheduler::TileScheduler,
};

fn bench_case<MakeOp>(
    group: &mut criterion::BenchmarkGroup<'_, criterion::measurement::WallTime>,
    name: &str,
    size: u32,
    make_op: MakeOp,
) where
    MakeOp: Fn(u32) -> FrequencyMaskOp<F32>,
{
    let scheduler = RayonScheduler::new(RayonScheduler::default_threads()).unwrap();
    let pixels = vec![0.0f32; size as usize * size as usize];

    group.bench_with_input(BenchmarkId::new(name, size), &size, |b, &size| {
        b.iter(|| {
            let source = MemorySource::<F32>::new(size, size, 1, pixels.clone()).unwrap();
            let pipeline = PipelinePlan::from_source(source)
                .append_dyn_op(Box::new(OperationBridge::new_pixel_local(make_op(size), 1)))
                .unwrap()
                .compile()
                .unwrap();
            let mut sink = MemorySink::for_pipeline(&pipeline).unwrap();
            scheduler.run(&pipeline, &mut sink).unwrap();
            black_box(sink.into_buffer())
        });
    });
}

pub fn bench_frequency_mask(c: &mut Criterion) {
    let mut group = c.benchmark_group("create_frequency_mask_f32");

    for &size in &[512u32, 2048, 8192] {
        bench_case(&mut group, "gaussian_band_optical", size, |size| {
            FrequencyMaskOp::<F32>::mask_gaussian_band(size, size, 0.35, 0.18, 0.12, 0.4)
                .unwrap()
                .with_optical(true)
                .with_nodc(true)
        });
        bench_case(&mut group, "butterworth_ring_reject", size, |size| {
            FrequencyMaskOp::<F32>::mask_butterworth_ring(size, size, 2.5, 0.42, 0.6, 0.14)
                .unwrap()
                .with_reject(true)
        });
        bench_case(&mut group, "fractal", size, |size| {
            FrequencyMaskOp::<F32>::mask_fractal(size, size, 2.7).unwrap()
        });
    }

    group.finish();
}

criterion_group!(benches, bench_frequency_mask);
criterion_main!(benches);
