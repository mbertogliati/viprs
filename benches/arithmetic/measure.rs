use criterion::{BenchmarkId, Criterion, black_box, criterion_group, criterion_main};
use viprs::domain::{
    format::U8,
    image::{Region, Tile},
    ops::arithmetic::MeasureOp,
    reducer::TileReducer,
};

fn bench_measure(c: &mut Criterion) {
    let mut group = c.benchmark_group("measure_u8");

    for &size in &[512u32, 2048, 8192] {
        let pixel_count = size as usize * size as usize;
        let pixels = vec![96u8; pixel_count];
        let region = Region::new(0, 0, size, size);
        let reducer = MeasureOp::new(1);

        group.bench_with_input(BenchmarkId::from_parameter(size), &size, |b, _| {
            b.iter(|| {
                let tile = Tile::<U8>::new(region, 1, &pixels);
                let partial = reducer.reduce_tile(&tile, &region);
                let result = <MeasureOp as TileReducer<U8>>::finalize(&reducer, partial);
                black_box(result);
            });
        });
    }

    group.finish();
}

criterion_group!(benches, bench_measure);
criterion_main!(benches);
