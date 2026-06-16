#![allow(missing_docs)]
#[path = "../common/mod.rs"]
mod common;

use criterion::{BenchmarkId, Criterion, black_box, criterion_group, criterion_main};
use viprs::domain::{
    format::U8,
    image::{Region, Tile},
    ops::mosaicing::{GlobalBalanceReducer, TileOverlap},
    reducer::TileReducer,
};

fn bench_global_balance(c: &mut Criterion) {
    let mut group = c.benchmark_group("global_balance_u8");

    for &size in &common::STANDARD_SIZES {
        let overlap_width = (size / 4).max(32);
        let left_region = Region::new(0, 0, size, size);
        let right_region = Region::new(size as i32 - overlap_width as i32, 0, size, size);
        let overlap = Region::new(size as i32 - overlap_width as i32, 0, overlap_width, size);
        let reducer = GlobalBalanceReducer::new(
            vec![left_region, right_region],
            vec![TileOverlap {
                lhs: 0,
                rhs: 1,
                region: overlap,
            }],
            3,
        )
        .unwrap();
        let left_pixels = vec![96u8; left_region.pixel_count() * 3];
        let right_pixels = vec![144u8; right_region.pixel_count() * 3];
        let left_tile = Tile::<U8>::new(left_region, 3, &left_pixels);
        let right_tile = Tile::<U8>::new(right_region, 3, &right_pixels);

        group.bench_with_input(BenchmarkId::from_parameter(size), &size, |b, _| {
            b.iter(|| {
                let partial = <GlobalBalanceReducer as TileReducer<U8>>::combine(
                    &reducer,
                    reducer.reduce_tile(&left_tile, &left_region),
                    reducer.reduce_tile(&right_tile, &right_region),
                );
                black_box(<GlobalBalanceReducer as TileReducer<U8>>::finalize(
                    &reducer, partial,
                ));
            });
        });
    }

    group.finish();
}

criterion_group!(benches, bench_global_balance);
criterion_main!(benches);
