#![allow(missing_docs)]
#[path = "../common/mod.rs"]
mod common;

use criterion::{Criterion, criterion_group, criterion_main};
use viprs::domain::ops::arithmetic::ComplexImagOp;

fn bench_complex_imag(c: &mut Criterion) {
    common::bench_direct_op_with_regions(
        c,
        "complex_imag_f32",
        2,
        1,
        || ComplexImagOp::new(2),
        common::direct_tile_regions,
        |samples| {
            (0..samples)
                .map(|index| index as f32 * 0.0625 - 16.0)
                .collect::<Vec<_>>()
        },
    );
}

criterion_group!(benches, bench_complex_imag);
criterion_main!(benches);
