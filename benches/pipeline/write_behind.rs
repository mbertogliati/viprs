use std::{thread, time::Duration};

use criterion::{
    BatchSize, BenchmarkId, Criterion, Throughput, black_box, criterion_group, criterion_main,
};
use viprs::{
    adapters::{
        pipeline::{CompiledPipeline, PipelineBuilder},
        scheduler::rayon_scheduler::RayonScheduler,
        sinks::{
            double_buffer::DoubleBufferSink,
            file_sink::{FileSink, FileSinkWriter},
        },
        sources::BlackSource,
    },
    domain::{
        error::ViprsError,
        format::U8,
        image::{DemandHint, Region, Tile, TileMut},
        op::{Op, OperationBridge},
    },
    ports::scheduler::TileScheduler,
    ports::sink::ConcurrentSink,
};

const STANDARD_SIZES: [u32; 3] = [512, 2048, 8192];
const BANDS: u32 = 3;
const BYTES_PER_SAMPLE: usize = 1;
const ROWS_PER_BUFFER: u32 = 64;
const WRITE_LATENCY_US: u64 = 200;

struct ThinStripCopy;

impl Op for ThinStripCopy {
    type Input = U8;
    type Output = U8;
    type State = ();

    fn demand_hint(&self) -> DemandHint {
        DemandHint::ThinStrip
    }

    fn required_input_region(&self, region: &Region) -> Region {
        *region
    }

    fn start(&self) {}

    #[inline]
    fn process_region(&self, _: &mut (), input: &Tile<U8>, output: &mut TileMut<U8>) {
        output.data.copy_from_slice(input.data);
    }
}

struct SimulatedEncodeWriter {
    latency: Duration,
    bytes_written: usize,
}

impl SimulatedEncodeWriter {
    fn new(latency: Duration) -> Self {
        Self {
            latency,
            bytes_written: 0,
        }
    }
}

impl FileSinkWriter for SimulatedEncodeWriter {
    fn write_region(&mut self, _region: Region, data: &[u8]) -> Result<(), ViprsError> {
        thread::sleep(self.latency);
        self.bytes_written += data.len();
        Ok(())
    }

    fn finish(&mut self) -> Result<(), ViprsError> {
        black_box(self.bytes_written);
        Ok(())
    }
}

fn make_pipeline(size: u32) -> CompiledPipeline {
    let source = BlackSource::new(size, size, BANDS);
    PipelineBuilder::from_source(source)
        .then(Box::new(OperationBridge::new(ThinStripCopy, BANDS)))
        .unwrap()
        .build()
        .unwrap()
}

fn run_with_file_sink(
    scheduler: &RayonScheduler,
    pipeline: &CompiledPipeline,
    latency: Duration,
) -> Result<(), ViprsError> {
    let sink = FileSink::new(
        pipeline.width,
        pipeline.height,
        pipeline.output_bands,
        BYTES_PER_SAMPLE,
        Box::new(SimulatedEncodeWriter::new(latency)),
    );
    scheduler.run_concurrent(pipeline, &sink)?;
    ConcurrentSink::finish(Box::new(sink))
}

fn run_with_double_buffer_sink(
    scheduler: &RayonScheduler,
    pipeline: &CompiledPipeline,
    latency: Duration,
) -> Result<(), ViprsError> {
    let sink = DoubleBufferSink::new(
        pipeline.width,
        pipeline.height,
        pipeline.output_bands,
        BYTES_PER_SAMPLE,
        ROWS_PER_BUFFER,
        Box::new(SimulatedEncodeWriter::new(latency)),
    )?;
    scheduler.run_concurrent(pipeline, &sink)?;
    ConcurrentSink::finish(Box::new(sink))
}

fn bench_write_behind(c: &mut Criterion) {
    let scheduler = RayonScheduler::new(RayonScheduler::default_threads()).unwrap();
    let latency = Duration::from_micros(WRITE_LATENCY_US);
    let mut group = c.benchmark_group("pipeline_write_behind_u8");
    group.sample_size(10);
    group.warm_up_time(Duration::from_millis(250));
    group.measurement_time(Duration::from_secs(2));

    for &size in &STANDARD_SIZES {
        let pipeline = make_pipeline(size);
        let bytes = size as u64 * size as u64 * BANDS as u64 * BYTES_PER_SAMPLE as u64;
        group.throughput(Throughput::Bytes(bytes));

        group.bench_with_input(BenchmarkId::new("file_sink", size), &size, |b, _| {
            b.iter_batched(
                || (),
                |_| {
                    run_with_file_sink(&scheduler, &pipeline, latency).unwrap();
                },
                BatchSize::SmallInput,
            );
        });

        group.bench_with_input(BenchmarkId::new("double_buffer", size), &size, |b, _| {
            b.iter_batched(
                || (),
                |_| {
                    run_with_double_buffer_sink(&scheduler, &pipeline, latency).unwrap();
                },
                BatchSize::SmallInput,
            );
        });
    }

    group.finish();
}

criterion_group!(benches, bench_write_behind);
criterion_main!(benches);
