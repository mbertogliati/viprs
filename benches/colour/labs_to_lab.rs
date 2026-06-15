#[path = "../common/mod.rs"]
mod common;

use criterion::{Criterion, criterion_group, criterion_main};
use viprs::domain::{
    format::{F32, I16},
    ops::colour::labs_to_lab::LabSToLab,
};

fn bench_labs_to_lab(c: &mut Criterion) {
    common::bench_direct_op_with_regions::<LabSToLab, I16, F32, _, _, _>(
        c,
        "labs_to_lab",
        3,
        3,
        || LabSToLab,
        common::direct_tile_regions::<LabSToLab>,
        |samples| vec![1024i16; samples],
    );
}

criterion_group!(benches, bench_labs_to_lab);
criterion_main!(benches);
