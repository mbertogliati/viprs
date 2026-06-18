#![allow(missing_docs)]
/// Benchmark: Premultiply<U8> and Unpremultiply<U8> — alpha premultiplication.
///
/// Measures `process_region` directly (not via the full pipeline), because
/// `PipelineBuilder` does not yet have a `premultiply()` convenience method
/// (blocked on B-50 which addresses multi-band-count ops in the bridge).
///
/// The benchmark covers the pixel-path hot loop for three RGBA image sizes.
use criterion::{BenchmarkId, Criterion, black_box, criterion_group, criterion_main};
use viprs::domain::{
    format::U8,
    image::{Region, Tile, TileMut},
    op::Op,
    ops::structural::{Premultiply, Unpremultiply},
};

fn bench_premultiply(c: &mut Criterion) {
    let mut group = c.benchmark_group("premultiply_u8");

    for &size in &[512u32, 2048, 8192] {
        let bands = 4u32; // RGBA
        let pixel_count = (size as usize) * (size as usize);
        // Vary alpha across pixels to avoid branch prediction shortcuts.
        let pixels: Vec<u8> = (0..pixel_count * bands as usize)
            .map(|i| {
                let band = i % bands as usize;
                if band == 3 { (i % 256) as u8 } else { 200u8 }
            })
            .collect();

        let region = Region::new(0, 0, size, size);
        let op = Premultiply::<U8>::new(bands);
        let mut out = vec![0u8; pixel_count * bands as usize];

        group.bench_with_input(BenchmarkId::from_parameter(size), &size, |b, _| {
            b.iter(|| {
                let input = Tile::<U8>::new(region, bands, &pixels);
                let mut output = TileMut::<U8>::new(region, bands, &mut out);
                let mut state = ();
                op.process_region(&mut state, &input, &mut output);
                black_box(&out);
            });
        });
    }

    group.finish();
}

fn bench_unpremultiply(c: &mut Criterion) {
    let mut group = c.benchmark_group("unpremultiply_u8");

    for &size in &[512u32, 2048, 8192] {
        let bands = 4u32;
        let pixel_count = (size as usize) * (size as usize);
        // Premultiplied data: colour bands scaled by alpha.
        let pixels: Vec<u8> = (0..pixel_count * bands as usize)
            .map(|i| {
                let band = i % bands as usize;
                let alpha = ((i / bands as usize) % 256) as u8;
                if band == 3 {
                    alpha
                } else {
                    (100u32 * alpha as u32 / 255) as u8
                }
            })
            .collect();

        let region = Region::new(0, 0, size, size);
        let op = Unpremultiply::<U8>::new(bands);
        let mut out = vec![0u8; pixel_count * bands as usize];

        group.bench_with_input(BenchmarkId::from_parameter(size), &size, |b, _| {
            b.iter(|| {
                let input = Tile::<U8>::new(region, bands, &pixels);
                let mut output = TileMut::<U8>::new(region, bands, &mut out);
                let mut state = ();
                op.process_region(&mut state, &input, &mut output);
                black_box(&out);
            });
        });
    }

    group.finish();
}

criterion_group!(benches, bench_premultiply, bench_unpremultiply);
criterion_main!(benches);
