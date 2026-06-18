#![allow(missing_docs)]
use criterion::{BenchmarkId, Criterion, black_box, criterion_group, criterion_main};
use viprs::{
    adapters::{
        pipeline::PipelineBuilder, scheduler::rayon_scheduler::RayonScheduler,
        sinks::memory::MemorySink, sources::memory::MemorySource,
    },
    domain::{format::F32, op::OperationBridge, ops::create::InvertlutOp},
    ports::scheduler::TileScheduler,
};

fn make_curve_table(rows: usize) -> Vec<f64> {
    let mut table = Vec::with_capacity(rows * 4);

    for index in 0..rows {
        let t = if rows == 1 {
            0.0
        } else {
            index as f64 / (rows - 1) as f64
        };
        table.extend_from_slice(&[t, t.powf(0.85), t * t, t.sqrt()]);
    }

    table
}

pub fn bench_invertlut(c: &mut Criterion) {
    let mut group = c.benchmark_group("create_invertlut_f32");
    let scheduler = RayonScheduler::new(RayonScheduler::default_threads()).unwrap();
    let rows = 17usize;
    let cols = 4usize;
    let bands = (cols - 1) as u32;
    let table = make_curve_table(rows);

    for &size in &[512u32, 2048, 8192] {
        let pixels = vec![0.0f32; size as usize * bands as usize];
        group.bench_with_input(
            BenchmarkId::new("triple_band_curve", size),
            &size,
            |b, &size| {
                b.iter(|| {
                    let source = MemorySource::<F32>::new(size, 1, bands, pixels.clone()).unwrap();
                    let pipeline = PipelineBuilder::from_source(source)
                        .then(Box::new(OperationBridge::new_pixel_local(
                            InvertlutOp::<F32>::new(&table, rows, cols, size).unwrap(),
                            bands,
                        )))
                        .unwrap()
                        .build()
                        .unwrap();
                    let mut sink = MemorySink::for_pipeline(&pipeline).unwrap();
                    scheduler.run(&pipeline, &mut sink).unwrap();
                    black_box(sink.into_buffer())
                });
            },
        );
    }

    group.finish();
}

criterion_group!(benches, bench_invertlut);
criterion_main!(benches);
