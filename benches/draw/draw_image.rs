#![allow(missing_docs)]
use criterion::{BenchmarkId, Criterion, black_box, criterion_group, criterion_main};
use viprs::domain::draw::DrawOp;
use viprs::domain::format::U8;
use viprs::domain::image::{Region, TileMut};
use viprs::domain::ops::draw::{DrawImageMode, DrawImageOp};

fn make_sub(size: u32) -> Vec<u8> {
    vec![32_u8; (size / 2) as usize * (size / 2) as usize]
}

fn bench_draw_image(c: &mut Criterion) {
    let mut group = c.benchmark_group("draw_image");

    for &size in &[512_u32, 2048, 8192] {
        let sub = make_sub(size);
        group.bench_with_input(BenchmarkId::from_parameter(size), &size, |b, &size| {
            let op = DrawImageOp::<U8>::new(
                size / 2,
                size / 2,
                1,
                sub.clone(),
                (size / 4) as i32,
                (size / 4) as i32,
                DrawImageMode::Set,
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

criterion_group!(benches, bench_draw_image);
criterion_main!(benches);
