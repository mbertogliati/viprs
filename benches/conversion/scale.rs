#![allow(missing_docs)]
use criterion::{BenchmarkId, Criterion, black_box, criterion_group, criterion_main};
use viprs::{
    adapters::{
        scheduler::rayon_scheduler::RayonScheduler, sinks::memory::MemorySink,
        sources::memory::MemorySource,
    },
    domain::{
        format::{U8, U16},
        image::{DemandHint, Region, Tile, TileMut},
        op::{DynOperation, Op, OperationBridge},
        ops::conversion::{ScaleMode, ScaleOp},
        reducers::stats::StatsReducer,
    },
    pipeline::internal::PipelinePlan,
    ports::scheduler::{ReducingScheduler, TileScheduler},
};

fn must<T, E: std::fmt::Display>(result: Result<T, E>, context: &str) -> T {
    match result {
        Ok(value) => value,
        Err(error) => panic!("{context}: {error}"),
    }
}

struct PassThroughU16;

impl Op for PassThroughU16 {
    type Input = U16;
    type Output = U16;
    type State = ();

    fn demand_hint(&self) -> DemandHint {
        DemandHint::ThinStrip
    }

    fn required_input_region(&self, output: &Region) -> Region {
        *output
    }

    fn start(&self) {}

    #[inline]
    fn process_region(&self, _state: &mut (), input: &Tile<U16>, output: &mut TileMut<U16>) {
        output.data.copy_from_slice(input.data);
    }
}

fn bytes_to_u16(bytes: &[u8]) -> Vec<u16> {
    bytemuck::cast_slice(bytes).to_vec()
}

fn bench_scale(c: &mut Criterion) {
    let mut group = c.benchmark_group("conversion_scale_u16_to_u8");
    let scheduler = must(
        RayonScheduler::new(RayonScheduler::default_threads()),
        "create rayon scheduler",
    );

    for &size in &[512u32, 2048, 8192] {
        let pixels = (0..size as usize * size as usize)
            .map(|index| ((index * 257) % (u16::MAX as usize + 1)) as u16)
            .collect::<Vec<_>>();

        group.bench_with_input(BenchmarkId::from_parameter(size), &size, |b, &size| {
            b.iter(|| {
                let source = must(
                    MemorySource::<U16>::new(size, size, 1, pixels.clone()),
                    "create memory source",
                );
                let input_pipeline = must(
                    PipelinePlan::from_source(source)
                        .append_dyn_op(Box::new(OperationBridge::new(PassThroughU16, 1))),
                    "add identity operation",
                )
                .compile()
                .unwrap();
                let stats_sink = MemorySink::for_pipeline(&input_pipeline).unwrap();
                let stats = must(
                    scheduler.run_with_reducer::<U16, StatsReducer>(
                        &input_pipeline,
                        &stats_sink,
                        &StatsReducer::new(1),
                    ),
                    "reduce scale stats",
                );
                let intermediate = bytes_to_u16(&stats_sink.into_buffer());

                let scale_source = must(
                    MemorySource::<U16>::new(size, size, 1, intermediate),
                    "create scale source",
                );
                let op: Box<dyn DynOperation> = Box::new(OperationBridge::new_pixel_local(
                    ScaleOp::<U16, U8>::from_stats(&stats, ScaleMode::Linear),
                    1,
                ));
                let scale_pipeline = must(
                    PipelinePlan::from_source(scale_source).append_dyn_op(op),
                    "add scale operation",
                )
                .compile()
                .unwrap();
                let mut sink = MemorySink::for_pipeline(&scale_pipeline).unwrap();
                must(
                    scheduler.run(&scale_pipeline, &mut sink),
                    "run scale pipeline",
                );
                black_box(sink.into_buffer())
            });
        });
    }

    group.finish();
}

criterion_group!(benches, bench_scale);
criterion_main!(benches);
