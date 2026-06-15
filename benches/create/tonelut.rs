use criterion::{BenchmarkId, Criterion, black_box, criterion_group, criterion_main};
use viprs::{
    adapters::{
        pipeline::PipelineBuilder, scheduler::rayon_scheduler::RayonScheduler,
        sinks::memory::MemorySink, sources::memory::MemorySource,
    },
    domain::{format::F32, op::OperationBridge, ops::create::TonelutOp},
    ports::scheduler::TileScheduler,
};

fn bench_case<MakeOp>(
    group: &mut criterion::BenchmarkGroup<'_, criterion::measurement::WallTime>,
    name: &str,
    size: u32,
    make_op: MakeOp,
) where
    MakeOp: Fn(u32) -> TonelutOp<F32>,
{
    let scheduler = RayonScheduler::new(RayonScheduler::default_threads()).unwrap();
    let pixels = vec![0.0f32; size as usize];

    group.bench_with_input(BenchmarkId::new(name, size), &size, |b, &size| {
        b.iter(|| {
            let source = MemorySource::<F32>::new(size, 1, 1, pixels.clone()).unwrap();
            let pipeline = PipelineBuilder::from_source(source)
                .then(Box::new(OperationBridge::new_pixel_local(make_op(size), 1)))
                .unwrap()
                .build()
                .unwrap();
            let mut sink = MemorySink::for_pipeline(&pipeline);
            scheduler.run(&pipeline, &mut sink).unwrap();
            black_box(sink.into_buffer())
        });
    });
}

pub fn bench_tonelut(c: &mut Criterion) {
    let mut group = c.benchmark_group("create_tonelut_f32");

    for &size in &[512u32, 2048, 8192] {
        bench_case(&mut group, "contrast_curve", size, |size| {
            TonelutOp::<F32>::new(
                size - 1,
                size - 1,
                0.0,
                100.0,
                0.18,
                0.5,
                0.84,
                18.0,
                -9.0,
                14.0,
            )
            .unwrap()
        });
        bench_case(&mut group, "warm_split_tone", size, |size| {
            TonelutOp::<F32>::new(
                size - 1,
                size - 1,
                0.0,
                100.0,
                0.22,
                0.52,
                0.86,
                -12.0,
                8.0,
                22.0,
            )
            .unwrap()
        });
    }

    group.finish();
}

criterion_group!(benches, bench_tonelut);
criterion_main!(benches);
