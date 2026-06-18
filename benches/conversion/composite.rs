#![allow(missing_docs)]
use criterion::{BenchmarkId, Criterion, black_box, criterion_group, criterion_main};
use viprs::{
    adapters::{
        pipeline::PipelineArena, scheduler::rayon_scheduler::RayonScheduler,
        sinks::memory::MemorySink, sources::memory::MemorySource,
    },
    domain::{
        format::F32,
        op::OperationBridge,
        ops::conversion::{BlendMode, CompositeOp, CopyOp},
    },
    ports::{scheduler::TileScheduler, source::DynImageSource},
};

fn bench_composite(c: &mut Criterion) {
    let mut group = c.benchmark_group("composite_over_f32");

    for &size in &[512u32, 2048, 8192] {
        let pixel_count = size as usize * size as usize;
        let pixels = vec![0.25f32; pixel_count * 4];

        group.bench_with_input(BenchmarkId::from_parameter(size), &size, |b, &size| {
            b.iter(|| {
                let source = MemorySource::<F32>::new(size, size, 4, pixels.clone()).unwrap();

                let mut arena =
                    PipelineArena::with_source(Box::new(source) as Box<dyn DynImageSource>);
                let base = arena.add_node(Box::new(OperationBridge::new_pixel_local(
                    CopyOp::<F32>::default(),
                    4,
                )));
                let overlay = arena.add_node(Box::new(OperationBridge::new_pixel_local(
                    CopyOp::<F32>::default(),
                    4,
                )));
                let composite = arena.add_node(Box::new(
                    CompositeOp::<F32>::new(BlendMode::Over, false, 4)
                        .expect("composite op configuration"),
                ));

                arena.connect(base, overlay).unwrap();
                arena.connect(base, composite).unwrap();
                arena.connect_to_slot(overlay, composite, 1).unwrap();

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

criterion_group!(benches, bench_composite);
criterion_main!(benches);
