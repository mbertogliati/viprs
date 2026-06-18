#![allow(missing_docs)]
#[path = "../common/mod.rs"]
mod common;

use criterion::{BenchmarkId, Criterion, black_box, criterion_group, criterion_main};
use viprs::domain::ops::mosaicing::{ChkpairOp, TiePointPair};

fn synthetic_pairs(size: u32) -> Vec<TiePointPair> {
    let stride = (size / 8).max(32) as f64;
    let max = f64::from(size.saturating_sub(64));
    let mut pairs = Vec::with_capacity(18);
    let mut y = 32.0;
    while y <= max && pairs.len() < 16 {
        let mut x = 32.0;
        while x <= max && pairs.len() < 16 {
            pairs.push(TiePointPair::from_xy(x, y, x + 3.0, y - 2.0));
            x += stride;
        }
        y += stride;
    }
    pairs.push(TiePointPair::from_xy(48.0, 48.0, 128.0, 96.0));
    pairs.push(TiePointPair::from_xy(
        max.max(48.0),
        max.max(48.0),
        12.0,
        14.0,
    ));
    pairs
}

fn bench_chkpair(c: &mut Criterion) {
    let mut group = c.benchmark_group("chkpair");
    let op = ChkpairOp::new(1.0).unwrap();

    for &size in &common::STANDARD_SIZES {
        let pairs = synthetic_pairs(size);
        group.bench_with_input(BenchmarkId::from_parameter(size), &size, |b, _| {
            b.iter(|| black_box(op.filter(&pairs).unwrap()));
        });
    }

    group.finish();
}

criterion_group!(benches, bench_chkpair);
criterion_main!(benches);
