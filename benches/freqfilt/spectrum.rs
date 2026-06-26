#![allow(missing_docs)]
use criterion::{BenchmarkId, Criterion, black_box, criterion_group, criterion_main};
use viprs::{
    adapters::{
      pipeline::ImagePipeline, scheduler::rayon_scheduler::RayonScheduler,
      sinks::memory::MemorySink, sources::memory::MemorySource,
    },
    domain::{format::F32, ops::freqfilt::SpectrumOp},
    ports::scheduler::TileScheduler,
};

fn bench_spectrum(c: &mut Criterion) {
    let mut group = c.benchmark_group("spectrum_f32");

    for &size in &[512u32, 2048, 8192] {
        let pixel_count = size as usize * size as usize;
        let pixels = (0..pixel_count)
            .flat_map(|index| {
                let re = (index % 19) as f32 * 0.25 - 2.0;
                let im = (index % 13) as f32 * 0.125;
                [re, im]
            })
            .collect::<Vec<_>>();

        group.bench_with_input(BenchmarkId::from_parameter(size), &size, |b, &size| {
            b.iter(|| {
                let source = MemorySource::<F32>::new(size, size, 2, pixels.clone()).unwrap();
                let pipeline = ImagePipeline::from_source(source)
                    .then(Box::new(SpectrumOp::<F32>::new()))
                    .unwrap()
                    .build()
                    .unwrap();
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

criterion_group!(benches, bench_spectrum);
criterion_main!(benches);
