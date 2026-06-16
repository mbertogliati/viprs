#![allow(missing_docs)]
#[path = "../common/mod.rs"]
mod common;

use criterion::{Criterion, criterion_group, criterion_main};
use viprs::domain::{format::U8, ops::histogram::hist_match::HistMatchOp};

fn bench_hist_match(c: &mut Criterion) {
    let cum: Vec<u64> = (0u64..256u64).collect();
    common::bench_direct_op_with_regions::<HistMatchOp<U8>, U8, U8, _, _, _>(
        c,
        "hist_match_u8",
        1,
        1,
        || HistMatchOp::<U8>::from_cumulative_hists(&cum, &cum).unwrap(),
        common::direct_tile_regions::<HistMatchOp<U8>>,
        |samples| (0..samples).map(|i| (i % 256) as u8).collect(),
    );
}

criterion_group!(benches, bench_hist_match);
criterion_main!(benches);
