#![allow(missing_docs)]
use criterion::{BenchmarkId, Criterion, black_box, criterion_group, criterion_main};
use viprs::domain::{
    draw::DrawOp,
    format::U8,
    image::{Region, TileMut},
    ops::draw::DrawLineOp,
};

fn bench_draw_line(c: &mut Criterion) {
    let mut group = c.benchmark_group("draw_line");

    for &size in &[512_u32, 2048, 8192] {
        group.bench_with_input(BenchmarkId::from_parameter(size), &size, |b, &size| {
            let op = DrawLineOp::<U8>::new(
                0,
                (size / 2) as i32,
                size as i32 - 1,
                (size / 2) as i32,
                vec![255],
            )
            .unwrap();
            b.iter(|| {
                let mut pixels = vec![0_u8; size as usize * size as usize];
                let mut tile = TileMut::new(Region::new(0, 0, size, size), 1, &mut pixels);
                op.draw(&mut tile);
                black_box(pixels);
            });
        });
    }

    group.finish();
}

criterion_group!(benches, bench_draw_line);
criterion_main!(benches);
