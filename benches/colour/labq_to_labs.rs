#![allow(missing_docs)]
#[path = "../common/mod.rs"]
mod common;

use criterion::{Criterion, criterion_group, criterion_main};
use viprs::domain::{
    format::{I16, U8},
    ops::colour::labq_to_labs::LabQToLabS,
};

fn bench_labq_to_labs(c: &mut Criterion) {
    common::bench_direct_op_with_regions::<LabQToLabS, U8, I16, _, _, _>(
        c,
        "labq_to_labs",
        4,
        3,
        || LabQToLabS,
        common::direct_tile_regions::<LabQToLabS>,
        |samples| vec![128u8; samples],
    );
}

criterion_group!(benches, bench_labq_to_labs);
criterion_main!(benches);
