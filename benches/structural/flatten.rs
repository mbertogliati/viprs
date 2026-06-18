#![allow(missing_docs)]
/// Benchmark: Flatten<U8> — alpha-composite RGBA onto a solid background.
///
/// Measures `process_region` directly to isolate the RGBA -> RGB hot loop.
use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion};
use viprs::domain::{
    format::U8,
    image::{Region, Tile, TileMut},
    op::Op,
    ops::structural::Flatten,
};

fn bench_flatten(c: &mut Criterion) {
    let mut group = c.benchmark_group("flatten_u8");

    for &size in &[512u32, 2048, 8192] {
        let in_bands = 4u32; // RGBA
        let out_bands = 3u32; // RGB
        let pixel_count = (size as usize) * (size as usize);

        // Vary alpha across pixels to avoid branch prediction shortcuts.
        let pixels: Vec<u8> = (0..pixel_count * in_bands as usize)
            .map(|i| {
                let band = i % in_bands as usize;
                if band == 3 {
                    (i % 256) as u8
                } else {
                    200u8
                }
            })
            .collect();

        let in_region = Region::new(0, 0, size, size);
        let out_region = Region::new(0, 0, size, size);
        let background = vec![0u8; out_bands as usize];
        let op = Flatten::<U8>::new(in_bands, background);
        let mut out = vec![0u8; pixel_count * out_bands as usize];

        group.bench_with_input(BenchmarkId::from_parameter(size), &size, |b, _| {
            b.iter(|| {
                let input = Tile::<U8>::new(in_region, in_bands, &pixels);
                let mut output = TileMut::<U8>::new(out_region, out_bands, &mut out);
                let mut state = ();
                op.process_region(&mut state, &input, &mut output);
                black_box(&out);
            });
        });
    }

    group.finish();
}

criterion_group!(benches, bench_flatten);
criterion_main!(benches);
