#![allow(missing_docs)]
#[path = "../common/mod.rs"]
mod common;

use criterion::{Criterion, criterion_group, criterion_main};
use viprs::domain::ops::arithmetic::ComplexConjOp;

fn bench_complex_conj(c: &mut Criterion) {
    common::bench_direct_op_with_regions(
        c,
        "complex_conj_f32",
        2,
        2,
        || ComplexConjOp,
        common::direct_tile_regions,
        |samples| {
            (0..samples)
                .map(|index| index as f32 * 0.125 - 32.0)
                .collect::<Vec<_>>()
        },
    );
}

criterion_group!(benches, bench_complex_conj);
criterion_main!(benches);
