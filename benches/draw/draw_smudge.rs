#![allow(missing_docs)]
use criterion::{BenchmarkId, Criterion, black_box, criterion_group, criterion_main};
use viprs::domain::draw::DrawOp;
use viprs::domain::format::U8;
use viprs::domain::image::{Region, TileMut};
use viprs::domain::ops::draw::DrawSmudgeOp;

fn bench_draw_smudge(c: &mut Criterion) {
    let mut group = c.benchmark_group("draw_smudge");

    for &size in &[512_u32, 2048, 8192] {
        group.bench_with_input(BenchmarkId::from_parameter(size), &size, |b, &size| {
            let op = DrawSmudgeOp::<U8>::new(
                size,
                size,
                (size / 4) as i32,
                (size / 4) as i32,
                size / 2,
                size / 2,
            );

            b.iter(|| {
                let mut pixels = vec![64_u8; size as usize * size as usize];
                let mut tile = TileMut::new(Region::new(0, 0, size, size), 1, &mut pixels);
                op.draw(&mut tile);
                black_box(pixels);
            });
        });
    }

    group.finish();
}

criterion_group!(benches, bench_draw_smudge);
criterion_main!(benches);
