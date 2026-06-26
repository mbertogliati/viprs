#![allow(missing_docs)]
use criterion::{BenchmarkId, Criterion, black_box, criterion_group, criterion_main};
use viprs::{
    adapters::{
      pipeline::ImagePipeline, scheduler::rayon_scheduler::RayonScheduler,
      sinks::memory::MemorySink, sources::memory::MemorySource,
    },
    domain::{
        format::{U8, U16},
        op::OperationBridge,
        ops::colour::{
            CicpColourPrimaries, CicpMatrixCoefficients, CicpProfile, CicpToScRgb,
            CicpTransferCharacteristics,
        },
    },
    ports::scheduler::TileScheduler,
};

fn make_u8_pixels(size: u32) -> Vec<u8> {
    let pixel_count = size as usize * size as usize * 3;
    (0..pixel_count)
        .map(|index| ((index * 29 + index / 3 * 7) % 256) as u8)
        .collect()
}

fn make_u16_pixels(size: u32) -> Vec<u16> {
    let pixel_count = size as usize * size as usize * 3;
    (0..pixel_count)
        .map(|index| ((index * 977 + index / 3 * 131) % 65_536) as u16)
        .collect()
}

fn hlg_bt2020_profile() -> CicpProfile {
    CicpProfile::new(
        CicpColourPrimaries::Bt2020,
        CicpTransferCharacteristics::Hlg,
        CicpMatrixCoefficients::RgbIdentity,
        true,
    )
}

fn pq_display_p3_profile() -> CicpProfile {
    CicpProfile::new(
        CicpColourPrimaries::Smpte432,
        CicpTransferCharacteristics::Pq,
        CicpMatrixCoefficients::RgbIdentity,
        true,
    )
}

fn bench_cicp_to_scrgb_u8(c: &mut Criterion) {
    let mut group = c.benchmark_group("cicp_to_scrgb_u8_hlg_bt2020");
    let scheduler = RayonScheduler::new(RayonScheduler::default_threads()).unwrap();
    let profile = hlg_bt2020_profile();

    for &size in &[512_u32, 2048, 8192] {
        let pixels = make_u8_pixels(size);
        group.bench_with_input(BenchmarkId::from_parameter(size), &size, |b, &size| {
            b.iter(|| {
                let source = MemorySource::<U8>::new(size, size, 3, pixels.clone()).unwrap();
                let pipeline = ImagePipeline::from_source(source)
                    .then(Box::new(OperationBridge::new_pixel_local(
                        CicpToScRgb::<U8>::new(profile),
                        3,
                    )))
                    .unwrap()
                    .build()
                    .unwrap();
                let mut sink = MemorySink::for_pipeline(&pipeline).unwrap();
                scheduler.run(&pipeline, &mut sink).unwrap();
                black_box(sink.into_buffer());
            });
        });
    }

    group.finish();
}

fn bench_cicp_to_scrgb_u16(c: &mut Criterion) {
    let mut group = c.benchmark_group("cicp_to_scrgb_u16_pq_display_p3");
    let scheduler = RayonScheduler::new(RayonScheduler::default_threads()).unwrap();
    let profile = pq_display_p3_profile();

    for &size in &[512_u32, 2048, 8192] {
        let pixels = make_u16_pixels(size);
        group.bench_with_input(BenchmarkId::from_parameter(size), &size, |b, &size| {
            b.iter(|| {
                let source = MemorySource::<U16>::new(size, size, 3, pixels.clone()).unwrap();
                let pipeline = ImagePipeline::from_source(source)
                    .then(Box::new(OperationBridge::new_pixel_local(
                        CicpToScRgb::<U16>::new(profile),
                        3,
                    )))
                    .unwrap()
                    .build()
                    .unwrap();
                let mut sink = MemorySink::for_pipeline(&pipeline).unwrap();
                scheduler.run(&pipeline, &mut sink).unwrap();
                black_box(sink.into_buffer());
            });
        });
    }

    group.finish();
}

criterion_group!(benches, bench_cicp_to_scrgb_u8, bench_cicp_to_scrgb_u16);
criterion_main!(benches);
