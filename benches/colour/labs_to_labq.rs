#![allow(missing_docs)]
#[path = "../common/mod.rs"]
mod common;

use criterion::{Criterion, criterion_group, criterion_main};
use viprs::domain::{
    format::{I16, U8},
    ops::colour::labs_to_labq::LabSToLabQ,
};

fn bench_labs_to_labq(c: &mut Criterion) {
    common::bench_direct_op_with_regions::<LabSToLabQ, I16, U8, _, _, _>(
        c,
        "labs_to_labq",
        3,
        4,
        || LabSToLabQ,
        common::direct_tile_regions::<LabSToLabQ>,
        |samples| vec![12_000i16; samples],
    );
}

criterion_group!(benches, bench_labs_to_labq);
criterion_main!(benches);
