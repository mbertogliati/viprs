#![allow(missing_docs)]
#[path = "../common/mod.rs"]
mod common;

use criterion::{Criterion, criterion_group, criterion_main};
use viprs::domain::ops::arithmetic::ComplexRealOp;

fn bench_complex_real(c: &mut Criterion) {
    common::bench_direct_op_with_regions(
        c,
        "complex_real_f32",
        2,
        1,
        || ComplexRealOp::new(2),
        common::direct_tile_regions,
        |samples| {
            (0..samples)
                .map(|index| index as f32 * 0.0625 - 16.0)
                .collect::<Vec<_>>()
        },
    );
}

criterion_group!(benches, bench_complex_real);
criterion_main!(benches);
