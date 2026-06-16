#![allow(missing_docs)]
#[path = "../common/mod.rs"]
mod common;

use criterion::{Criterion, criterion_group, criterion_main};
use viprs::domain::{format::U8, ops::structural::unpremultiply::Unpremultiply};

fn bench_unpremultiply(c: &mut Criterion) {
    common::bench_direct_op_with_regions::<Unpremultiply<U8>, U8, U8, _, _, _>(
        c,
        "unpremultiply_u8_rgba",
        4,
        4,
        || Unpremultiply::<U8>::new(4),
        common::direct_tile_regions::<Unpremultiply<U8>>,
        |samples| {
            (0..samples)
                .map(|i| {
                    if i % 4 == 3 {
                        ((i / 4) % 256) as u8
                    } else {
                        96u8
                    }
                })
                .collect()
        },
    );
}

criterion_group!(benches, bench_unpremultiply);
criterion_main!(benches);
