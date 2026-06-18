#![allow(missing_docs)]
use bytemuck::{cast_slice, cast_slice_mut};
use criterion::{BenchmarkId, Criterion, black_box, criterion_group, criterion_main};
use viprs::domain::{
    format::U8,
    image::Region,
    op::DynOperation,
    ops::colour::{UhdrGainMapMetadata, UhdrToScRgb},
};

fn base_pixels(size: u32) -> Vec<u8> {
    let sample_count = size as usize * size as usize * 3;
    (0..sample_count)
        .map(|idx| ((idx * 29 + idx / 3 * 7) % 256) as u8)
        .collect()
}

fn mono_gainmap(size: u32) -> Vec<u8> {
    let sample_count = size as usize * size as usize;
    (0..sample_count)
        .map(|idx| ((idx * 17 + idx / 11) % 256) as u8)
        .collect()
}

fn rgb_gainmap(size: u32) -> Vec<u8> {
    let sample_count = size as usize * size as usize * 3;
    (0..sample_count)
        .map(|idx| ((idx * 13 + idx / 5 * 19) % 256) as u8)
        .collect()
}

fn metadata() -> UhdrGainMapMetadata {
    UhdrGainMapMetadata {
        gamma: [1.0, 1.2, 1.4],
        min_content_boost: [1.0, 1.0, 1.0],
        max_content_boost: [2.0, 4.0, 8.0],
        offset_hdr: [0.0, 0.0, 0.0],
        offset_sdr: [0.0, 0.0, 0.0],
    }
}

// Criterion-only baseline: the current xtask/libvips runner only exposes single-input image ops,
// while `uhdr2scrgb` requires a base image, a gainmap, and metadata to define the conversion.
fn bench_uhdr2scrgb(c: &mut Criterion) {
    let mono_op = UhdrToScRgb::<U8>::new(metadata(), 1)
        .unwrap_or_else(|err| panic!("failed to construct mono uhdr op: {err}"));
    let rgb_op = UhdrToScRgb::<U8>::new(metadata(), 3)
        .unwrap_or_else(|err| panic!("failed to construct rgb uhdr op: {err}"));

    {
        let mut mono_group = c.benchmark_group("uhdr2scrgb_u8_mono");
        for &size in &[512_u32, 2048, 8192] {
            let region = Region::new(0, 0, size, size);
            let base = base_pixels(size);
            let mono = mono_gainmap(size);
            let mut mono_output = vec![0.0f32; size as usize * size as usize * 3];
            let input_regions = [region, region];

            mono_group.bench_with_input(BenchmarkId::from_parameter(size), &size, |b, _| {
                b.iter(|| {
                    let mut state = ();
                    mono_op.dyn_process_region_multi(
                        &mut state,
                        &[cast_slice(&base), cast_slice(&mono)],
                        cast_slice_mut(&mut mono_output),
                        &input_regions,
                        region,
                    );
                    black_box(&mono_output);
                });
            });
        }
        mono_group.finish();
    }

    let mut rgb_group = c.benchmark_group("uhdr2scrgb_u8_rgb");
    for &size in &[512_u32, 2048, 8192] {
        let region = Region::new(0, 0, size, size);
        let base = base_pixels(size);
        let rgb = rgb_gainmap(size);
        let mut rgb_output = vec![0.0f32; size as usize * size as usize * 3];
        let input_regions = [region, region];

        rgb_group.bench_with_input(BenchmarkId::from_parameter(size), &size, |b, _| {
            b.iter(|| {
                let mut state = ();
                rgb_op.dyn_process_region_multi(
                    &mut state,
                    &[cast_slice(&base), cast_slice(&rgb)],
                    cast_slice_mut(&mut rgb_output),
                    &input_regions,
                    region,
                );
                black_box(&rgb_output);
            });
        });
    }
    rgb_group.finish();
}

criterion_group!(benches, bench_uhdr2scrgb);
criterion_main!(benches);
