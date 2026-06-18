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
        ops::{conversion::CopyOp, freqfilt::FreqMultOp},
    },
    ports::{scheduler::TileScheduler, source::DynImageSource},
};

fn bench_freqmult(c: &mut Criterion) {
    let mut group = c.benchmark_group("freqmult_f32");

    for &size in &[512u32, 2048, 8192] {
        let pixel_count = size as usize * size as usize;
        let pixels = (0..pixel_count)
            .flat_map(|index| {
                let re = (index % 17) as f32 * 0.125 + 1.0;
                let im = (index % 11) as f32 * 0.0625;
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
                let freqmult = arena.add_node(Box::new(FreqMultOp::<F32>::new()));

                arena.connect(base, branch).unwrap();
                arena.connect(base, freqmult).unwrap();
                arena.connect_to_slot(branch, freqmult, 1).unwrap();

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

criterion_group!(benches, bench_freqmult);
criterion_main!(benches);
