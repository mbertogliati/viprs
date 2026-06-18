#![allow(missing_docs)]
#[path = "../common/mod.rs"]
mod common;

use criterion::{Criterion, criterion_group, criterion_main};
use viprs::domain::{format::U8, ops::mosaicing::RemosaicOp};

fn bench_remosaic(c: &mut Criterion) {
    common::bench_direct_op_with_regions(
        c,
        "remosaic_u8",
        3,
        3,
        || RemosaicOp::<U8>::new(1.15).unwrap(),
        common::direct_tile_regions,
        |len| vec![96u8; len],
    );
}

criterion_group!(benches, bench_remosaic);
criterion_main!(benches);
