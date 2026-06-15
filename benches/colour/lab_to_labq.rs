#[path = "../common/mod.rs"]
mod common;

use criterion::{Criterion, criterion_group, criterion_main};
use viprs::domain::{
    format::{F32, U8},
    ops::colour::lab_to_labq::LabToLabQ,
};

fn bench_lab_to_labq(c: &mut Criterion) {
    common::bench_direct_op_with_regions::<LabToLabQ, F32, U8, _, _, _>(
        c,
        "lab_to_labq",
        3,
        4,
        || LabToLabQ,
        common::direct_tile_regions::<LabToLabQ>,
        |samples| vec![42.0f32; samples],
    );
}

criterion_group!(benches, bench_lab_to_labq);
criterion_main!(benches);
