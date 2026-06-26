#![allow(missing_docs)]
use criterion::{BenchmarkId, Criterion, criterion_group, criterion_main};
use viprs::{
    adapters::{
        pipeline::internal::PipelineBuilder, scheduler::rayon_scheduler::RayonScheduler,
        sinks::memory::MemorySink, sources::memory::MemorySource,
    },
    domain::{
        format::F32,
        op::OperationBridge,
        ops::create::{MaskButterworthOp, MaskFractalOp, MaskGaussianOp, MaskIdealOp},
    },
    ports::scheduler::TileScheduler,
};

fn bench_mask_family<O>(
    group: &mut criterion::BenchmarkGroup<'_, criterion::measurement::WallTime>,
    name: &str,
    size: u32,
    op: O,
) where
    O: viprs::domain::op::PixelLocalOp<Input = F32, Output = F32> + Copy + Send + Sync + 'static,
{
    let scheduler = RayonScheduler::new(RayonScheduler::default_threads()).unwrap();
    let pixels = vec![0.0f32; size as usize * size as usize];
    group.bench_with_input(BenchmarkId::new(name, size), &size, |b, &_size| {
        b.iter(|| {
            let source = MemorySource::<F32>::new(size, size, 1, pixels.clone()).unwrap();
            let pipeline = PipelineBuilder::from_source(source)
                .then(Box::new(OperationBridge::new_pixel_local(op, 1)))
                .unwrap()
                .build()
                .unwrap();
            let mut sink = MemorySink::for_pipeline(&pipeline).unwrap();
            scheduler.run(&pipeline, &mut sink).unwrap();
            criterion::black_box(sink.into_buffer())
        });
    });
}

fn bench_masks(c: &mut Criterion) {
    let mut group = c.benchmark_group("create_masks_f32");

    for &size in &[512u32, 2048, 8192] {
        bench_mask_family(
            &mut group,
            "ideal",
            size,
            MaskIdealOp::<F32>::mask_ideal(size, size, 0.35).unwrap(),
        );
        bench_mask_family(
            &mut group,
            "gaussian",
            size,
            MaskGaussianOp::<F32>::mask_gaussian(size, size, 0.35, 0.5).unwrap(),
        );
        bench_mask_family(
            &mut group,
            "butterworth",
            size,
            MaskButterworthOp::<F32>::mask_butterworth(size, size, 2.0, 0.35, 0.5).unwrap(),
        );
        bench_mask_family(
            &mut group,
            "fractal",
            size,
            MaskFractalOp::<F32>::mask_fractal(size, size, 2.5).unwrap(),
        );
    }

    group.finish();
}

criterion_group!(benches, bench_masks);
criterion_main!(benches);
