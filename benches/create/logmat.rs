#![allow(missing_docs)]
use criterion::{BenchmarkId, Criterion, black_box, criterion_group, criterion_main};
use viprs::{
    adapters::{
        pipeline::ImagePipeline, scheduler::rayon_scheduler::RayonScheduler,
        sinks::memory::MemorySink, sources::memory::MemorySource,
    },
    domain::{
        format::F32,
        op::OperationBridge,
        ops::create::{LogmatOp, LogmatPrecision},
    },
    ports::scheduler::TileScheduler,
};

fn sigma_for_target_width(target: u32, min_ampl: f64) -> f64 {
    let mut low = 0.000_001f64;
    let mut high = target as f64;

    for _ in 0..48 {
        let sigma = (low + high) * 0.5;
        match LogmatOp::<F32>::new(sigma, min_ampl) {
            Ok(op) if op.width() < target => {
                low = sigma;
            }
            Ok(_) | Err(_) => {
                high = sigma;
            }
        }
    }

    low
}

fn bench_case(
    group: &mut criterion::BenchmarkGroup<'_, criterion::measurement::WallTime>,
    name: &str,
    target_width: u32,
    min_ampl: f64,
    separable: bool,
    precision: LogmatPrecision,
) {
    let scheduler = RayonScheduler::new(RayonScheduler::default_threads()).unwrap();
    let sigma = sigma_for_target_width(target_width, min_ampl);
    let template = LogmatOp::<F32>::new(sigma, min_ampl)
        .unwrap()
        .with_separable(separable)
        .with_precision(precision);
    let pixels = vec![0.0f32; template.width() as usize * template.height() as usize];

    group.bench_with_input(
        BenchmarkId::new(name, target_width),
        &target_width,
        |b, &_target_width| {
            b.iter(|| {
                let op = LogmatOp::<F32>::new(sigma, min_ampl)
                    .unwrap()
                    .with_separable(separable)
                    .with_precision(precision);
                let source = MemorySource::<F32>::new(
                    template.width(),
                    template.height(),
                    1,
                    pixels.clone(),
                )
                .unwrap();
                let pipeline = ImagePipeline::from_source(source)
                    .then(Box::new(OperationBridge::new_pixel_local(op, 1)))
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

pub fn bench_logmat(c: &mut Criterion) {
    let mut group = c.benchmark_group("create_logmat_f32");

    for &size in &[512u32, 2048, 8192] {
        bench_case(
            &mut group,
            "float_kernel",
            size,
            0.05,
            false,
            LogmatPrecision::Float,
        );
        bench_case(
            &mut group,
            "integer_separable",
            size,
            0.1,
            true,
            LogmatPrecision::Integer,
        );
    }

    group.finish();
}

criterion_group!(benches, bench_logmat);
criterion_main!(benches);
