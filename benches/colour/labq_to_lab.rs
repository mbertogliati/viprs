#[path = "../common/mod.rs"]
mod common;

use criterion::{Criterion, criterion_group, criterion_main};
use viprs::domain::{
    format::{F32, U8},
    ops::colour::labq_to_lab::LabQToLab,
};

fn bench_labq_to_lab(c: &mut Criterion) {
    common::bench_direct_op_with_regions::<LabQToLab, U8, F32, _, _, _>(
        c,
        "labq_to_lab",
        4,
        3,
        || LabQToLab,
        common::direct_tile_regions::<LabQToLab>,
        |samples| vec![128u8; samples],
    );
}

criterion_group!(benches, bench_labq_to_lab);
criterion_main!(benches);
