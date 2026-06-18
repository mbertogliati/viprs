#![allow(missing_docs)]
use criterion::{BenchmarkId, Criterion, black_box, criterion_group, criterion_main};
use viprs::domain::draw::DrawOp;
use viprs::domain::format::U8;
use viprs::domain::image::{Region, TileMut};
use viprs::domain::ops::draw::DrawFloodOp;

fn make_pixels(size: u32) -> Vec<u8> {
    let mut pixels = vec![0u8; size as usize * size as usize];
    let width = size as usize;
    for x in 0..width {
        pixels[x] = 1;
        pixels[(width - 1) * width + x] = 1;
    }
    for y in 0..width {
        pixels[y * width] = 1;
        pixels[y * width + width - 1] = 1;
    }
    pixels
}

fn bench_draw_flood(c: &mut Criterion) {
    let mut group = c.benchmark_group("draw_flood_u8");

    for &size in &[512u32, 2048, 8192] {
        let pixels = make_pixels(size);
        let op = DrawFloodOp::<U8>::new((size / 2) as i32, (size / 2) as i32, vec![255]).unwrap();

        group.bench_with_input(BenchmarkId::from_parameter(size), &size, |b, &size| {
            b.iter(|| {
                let mut working = pixels.clone();
                let mut tile = TileMut::new(Region::new(0, 0, size, size), 1, &mut working);
                op.draw(&mut tile);
                black_box(working);
            });
        });
    }

    group.finish();
}

criterion_group!(benches, bench_draw_flood);
criterion_main!(benches);
