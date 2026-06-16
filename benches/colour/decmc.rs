#![allow(missing_docs)]
#[path = "../common/mod.rs"]
mod common;

use criterion::{Criterion, criterion_group, criterion_main};
use viprs::domain::{format::F32, ops::colour::decmc::DECMC};

fn bench_decmc(c: &mut Criterion) {
    common::bench_direct_op_with_regions::<DECMC, F32, F32, _, _, _>(
        c,
        "decmc_f32",
        6,
        1,
        || DECMC,
        common::direct_tile_regions::<DECMC>,
        |samples| vec![0.5f32; samples],
    );
}

criterion_group!(benches, bench_decmc);
criterion_main!(benches);
