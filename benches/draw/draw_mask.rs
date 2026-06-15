use criterion::{BenchmarkId, Criterion, black_box, criterion_group, criterion_main};
use viprs::{DrawMaskOp, DrawOp, Region, TileMut, U8};

fn make_mask(size: u32) -> Vec<u8> {
    vec![128_u8; (size / 2) as usize * (size / 2) as usize]
}

fn bench_draw_mask(c: &mut Criterion) {
    let mut group = c.benchmark_group("draw_mask");

    for &size in &[512_u32, 2048, 8192] {
        let mask = make_mask(size);
        group.bench_with_input(BenchmarkId::from_parameter(size), &size, |b, &size| {
            let op = DrawMaskOp::<U8>::new(
                size / 2,
                size / 2,
                mask.clone(),
                vec![255],
                (size / 4) as i32,
                (size / 4) as i32,
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

criterion_group!(benches, bench_draw_mask);
criterion_main!(benches);
