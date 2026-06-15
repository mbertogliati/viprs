use criterion::{BenchmarkId, Criterion, black_box, criterion_group, criterion_main};
use viprs::domain::{
    draw::DrawOp,
    format::U8,
    image::{Region, TileMut},
    ops::draw::DrawCircleOp,
};

fn bench_draw_circle(c: &mut Criterion) {
    let mut group = c.benchmark_group("draw_circle");

    for &size in &[512_u32, 2048, 8192] {
        group.bench_with_input(BenchmarkId::from_parameter(size), &size, |b, &size| {
            let op = DrawCircleOp::<U8>::new(
                (size / 2) as i32,
                (size / 2) as i32,
                size / 4,
                vec![255],
                false,
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

criterion_group!(benches, bench_draw_circle);
criterion_main!(benches);
