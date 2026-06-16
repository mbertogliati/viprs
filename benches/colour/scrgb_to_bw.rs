#![allow(missing_docs)]
#[path = "../common/mod.rs"]
mod common;

use criterion::{Criterion, criterion_group, criterion_main};
use viprs::domain::{format::F32, ops::colour::scrgb_to_bw::ScRgbToBw};

fn bench_scrgb_to_bw(c: &mut Criterion) {
    common::bench_direct_op_with_regions::<ScRgbToBw, F32, F32, _, _, _>(
        c,
        "scrgb_to_bw",
        3,
        1,
        || ScRgbToBw,
        common::direct_tile_regions::<ScRgbToBw>,
        |samples| vec![0.75f32; samples],
    );
}

criterion_group!(benches, bench_scrgb_to_bw);
criterion_main!(benches);
