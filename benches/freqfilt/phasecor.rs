use criterion::{BenchmarkId, Criterion, black_box, criterion_group, criterion_main};
use viprs::{
    adapters::{
        pipeline::PipelineArena, scheduler::rayon_scheduler::RayonScheduler,
        sinks::memory::MemorySink, sources::memory::MemorySource,
    },
    domain::{
        format::F32,
        op::OperationBridge,
        ops::{conversion::CopyOp, freqfilt::PhasecorOp},
    },
    ports::{scheduler::TileScheduler, source::DynImageSource},
};

fn bench_phasecor(c: &mut Criterion) {
    let mut group = c.benchmark_group("phasecor_f32");

    for &size in &[512u32, 2048, 8192] {
        let pixel_count = size as usize * size as usize;
        let pixels = (0..pixel_count)
            .flat_map(|index| {
                let re = (index % 23) as f32 * 0.125 + 1.0;
                let im = (index % 7) as f32 * 0.25 + 0.5;
                [re, im]
            })
            .collect::<Vec<_>>();

        group.bench_with_input(BenchmarkId::from_parameter(size), &size, |b, &size| {
            b.iter(|| {
                let source = MemorySource::<F32>::new(size, size, 2, pixels.clone()).unwrap();
                let mut arena =
                    PipelineArena::with_source(Box::new(source) as Box<dyn DynImageSource>);
                let base = arena.add_node(Box::new(OperationBridge::new_pixel_local(
                    CopyOp::<F32>::default(),
                    2,
                )));
                let branch = arena.add_node(Box::new(OperationBridge::new_pixel_local(
                    CopyOp::<F32>::default(),
                    2,
                )));
                let phasecor = arena.add_node(Box::new(PhasecorOp::<F32>::new()));

                arena.connect(base, branch).unwrap();
                arena.connect(base, phasecor).unwrap();
                arena.connect_to_slot(branch, phasecor, 1).unwrap();

                let pipeline = arena.compile().unwrap();
                let mut sink = MemorySink::for_pipeline(&pipeline);
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

criterion_group!(benches, bench_phasecor);
criterion_main!(benches);
