#![allow(missing_docs)]
#[path = "../common/mod.rs"]
mod common;

use criterion::{Criterion, criterion_group, criterion_main};
use viprs::domain::{
    format::{F32, I16},
    ops::colour::lab_to_labs::LabToLabS,
};

fn bench_lab_to_labs(c: &mut Criterion) {
    common::bench_direct_op_with_regions::<LabToLabS, F32, I16, _, _, _>(
        c,
        "lab_to_labs",
        3,
        3,
        || LabToLabS,
        common::direct_tile_regions::<LabToLabS>,
        |samples| vec![32.0f32; samples],
    );
}

criterion_group!(benches, bench_lab_to_labs);
criterion_main!(benches);
